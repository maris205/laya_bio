#!/usr/bin/env python3
"""Admit a bounded, full-input three-task JEV-style SFT experiment."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from laya_biocpt_data import sha, digest, read_lines, write_lines, write_json, key, kmers

TASKS=['promoter','structural_class','fluorescence']
TYPE_IDS={'choice':0,'score':1,'noul':2}


class Representation:
    def __init__(self, cpt_data):
        self.tokenizer=AutoTokenizer.from_pretrained(cpt_data/'representation/base_tokenizer')
        entries=json.loads((cpt_data/'representation/new_tokens.json').read_text())
        self.sources={m:Tokenizer.from_file(str(cpt_data/f'representation/source_tokenizers/{m}.json')) for m in ['dna','protein']}
        self.maps={m:{e['source_id']:e['expanded_id'] for e in entries if e['modality']==m} for m in self.sources}
    def text(self,s):
        return self.tokenizer(s,add_special_tokens=False)['input_ids']
    def biological(self,s,m):
        e=self.sources[m].encode(s)
        assert ''.join(e.tokens)==s and '[UNK]' not in e.tokens
        return [self.maps[m][i] for i in e.ids]
    def prepare(self,row):
        tok=self.tokenizer
        head=[tok.cls_token_id]+self.text(row['primitive']+' question: '+row['question'])+[tok.sep_token_id]
        options=[]
        for i,label in enumerate(row['choices']):
            text=f'level {i}: {label}' if row['primitive']=='score' else label
            options.append([tok.mask_token_id]+self.text(' '+text))
        body=[tok.sep_token_id]+self.text(('DNA' if row['modality']=='dna' else 'Protein')+' sequence:')+self.biological(row['sequence'],row['modality'])+[tok.sep_token_id]
        return {**row,'head_ids':head,'option_ids':options,'body_ids':body,'length':len(head)+sum(map(len,options))+len(body),'qtype':TYPE_IDS[row['primitive']]}


def interpolate(value, anchors):
    """Continuous target distribution with exact expected value in train range."""
    v=float(np.clip(value,anchors[0],anchors[-1])); out=np.zeros(len(anchors),dtype=np.float64)
    j=int(np.searchsorted(anchors,v,side='right'))
    if j==0:out[0]=1
    elif j==len(anchors):out[-1]=1
    else:
        w=(v-anchors[j-1])/(anchors[j]-anchors[j-1]);out[j-1]=1-w;out[j]=w
    return out.tolist()


def subset(rows,n):
    """Hash sample; preserve original class proportions and all observed classes."""
    if len(rows)<n:raise ValueError(f'Insufficient admitted rows: {len(rows)} < {n}')
    if rows[0]['primitive']=='score':return sorted(rows,key=lambda r:digest('jev-sft-v1:'+r['id']))[:n]
    pools=defaultdict(list)
    for r in rows:pools[r['label']].append(r)
    labels=sorted(pools);quota={k:max(1,int(n*len(pools[k])/len(rows))) for k in labels}
    while sum(quota.values())<n:
        k=max(labels,key=lambda k:(n*len(pools[k])/len(rows)-quota[k],-k));quota[k]+=1
    while sum(quota.values())>n:
        k=max((k for k in labels if quota[k]>1),key=lambda k:quota[k]-n*len(pools[k])/len(rows));quota[k]-=1
    return [r for k in labels for r in sorted(pools[k],key=lambda r:digest('jev-sft-v1:'+r['id']))[:quota[k]]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--cpt-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--train-per-task',type=int,default=8192)
    p.add_argument('--max-length',type=int,default=512)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'status.json',{'status':'preparing'})
    rep=Representation(a.cpt_root/'data'); sources={}; pools={'train':[],'dev':[]};protected=set(); excluded=[]
    def source(path):sources[str(path)]=sha(path);return path
    formal=a.workspace/'artifacts/laya_formal_data';manifest=json.loads(source(formal/'manifest.json').read_text())
    for native,task,primitive in [('promoter_detection','promoter','noul'),('fold_class','structural_class','choice')]:
        for split in ['calibration','test']:
            path=source(formal/f'{native}_{split}.jsonl')
            assert sha(path)==manifest['tasks'][native]['files'][split]['sha256']
            # Input-only reservation; targets are neither copied nor consulted.
            for r in read_lines(path):protected.add((r['modality'],key(r['sequence'],r['modality'])))
        for split,native_split in [('train','train'),('dev','selection_dev')]:
            path=source(formal/f'{native}_{native_split}.jsonl');assert sha(path)==manifest['tasks'][native]['files'][native_split]['sha256']
            for r in read_lines(path):
                label=r['label'];choices=r['choices']
                if task=='promoter':
                    label=int(r['choices'][r['label']]=='promoter');choices=['false: not a promoter','true: promoter']
                pools[split].append({'id':r['id'],'task':task,'primitive':primitive,'modality':r['modality'],'sequence':r['sequence'],
                  'label':label,'choices':choices,'question':'Is this DNA sequence a promoter?' if task=='promoter' else 'Which structural class describes this protein?',
                  'group_id':r['modality']+':'+key(r['sequence'],r['modality']),'split':split,'source_split':native_split})
    gfp_bank=set()
    for split,native in [('train','train'),('dev','valid')]:
        path=source(a.workspace/f'data/05_task_extension_sources/tape/fluorescence/fluorescence_{native}.json')
        for i,r in enumerate(json.loads(path.read_text())):
            seq=r['primary'].upper();v=float(r['log_fluorescence'][0]);assert np.isfinite(v)
            gfp_bank.update(kmers(seq,15))
            pools[split].append({'id':f'fluorescence:{native}:{i}','task':'fluorescence','primitive':'score','modality':'protein','sequence':seq,
               'value':v,'label':0,'choices':['pending']*5,'question':'What is the log-fluorescence level of this GFP variant?',
               'group_id':'protein:'+key(seq,'protein'),'split':split,'source_split':native})
    # Admission for the new task covers the entire saved CPT sample (including
    # unused sequences), not just the subset actually presented to the model.
    overlaps=[];cpt_checked=0
    for split in ['train','validation']:
        path=source(a.cpt_root/f'data/raw_{split}.jsonl')
        for r in read_lines(path):
            if r['modality']!='protein':continue
            cpt_checked+=1
            if any(x in gfp_bank for x in kmers(r['content'],15)):overlaps.append({'id':r['id'],'split':split})
    admission={'status':'pass' if not overlaps else 'failed','new_task':'fluorescence','GFP_native_train_and_valid_15mers':len(gfp_bank),
       'CPT_protein_snapshots_checked':cpt_checked,'overlapping_CPT_rows':overlaps,'test_targets_used':False,'GFP_test_file_read':False,
       'limitation':'15mer exclusion does not establish family independence. Native GFP train/valid share a parent.'}
    write_json(a.output/'cpt_admission.json',admission)
    if overlaps:raise ValueError('New task overlaps the existing CPT snapshot; do not launch from this checkpoint.')
    heldout=protected|{(r['modality'],key(r['sequence'],r['modality'])) for r in pools['dev']}
    targets=defaultdict(set)
    for rows in pools.values():
        for r in rows:targets[(r['task'],r['group_id'])].add(str(r.get('value',r['label'])))
    filtered={};availability={};seen=set()
    for split in ['dev','train']:
        filtered[split]=[]
        for r in pools[split]:
            entity=(r['modality'],key(r['sequence'],r['modality']));dupe=(r['task'],r['group_id']);reason=None
            if entity in protected:reason='protected_formal_calibration_or_test_sequence'
            elif split=='train' and entity in heldout:reason='entity_in_development'
            elif len(targets[dupe])>1:reason='conflicting_targets'
            elif dupe in seen:reason='duplicate_entity'
            if reason:excluded.append({'id':r['id'],'task':r['task'],'reason':reason});continue
            seen.add(dupe);filtered[split].append(r)
    # Choose Score anchors from the actual hash-selected training rows only.
    score_rows=subset([r for r in filtered['train'] if r['task']=='fluorescence'],a.train_per_task)
    values=np.array([r['value'] for r in score_rows]);anchors=np.linspace(values.min(),values.max(),5).tolist()
    spec={'anchors':anchors,'train_mean':float(values.mean()),'train_median':float(np.median(values)),'train_std':float(values.std()),
          'train_min':float(values.min()),'train_max':float(values.max()),'fit_entities':len(values),'target':'linear interpolation between neighboring fixed native-unit anchors',
          'loss':'0.5 soft-target cross-entropy + 0.5 standardized MSE of expected native score','calibration_fit':False}
    for split in filtered:
        encoded=[]
        for r in filtered[split]:
            if r['primitive']=='score':
                r['anchors']=anchors;r['choices']=[f'log fluorescence = {x:.8g}' for x in anchors]
                r['target_probs']=interpolate(r['value'],anchors);r['label']=int(np.argmax(r['target_probs']))
            r=rep.prepare(r)
            if r['length']>a.max_length:excluded.append({'id':r['id'],'task':r['task'],'reason':'full_input_over_budget','length':r['length']});continue
            encoded.append(r)
        filtered[split]=encoded;availability[split]=dict(Counter(r['task'] for r in encoded))
    chosen={}
    for split in ['train','dev']:
        chosen[split]=[]
        for task in TASKS:
            pool=[r for r in filtered[split] if r['task']==task]
            pool=subset(pool,a.train_per_task) if split=='train' else pool
            chosen[split].extend(sorted(pool,key=lambda r:digest('jev-order-v1:'+r['id'])))
    assert {r['id'] for r in chosen['train'] if r['task']=='fluorescence'}=={r['id'] for r in score_rows},'Anchor fit subset changed after input eligibility.'
    assert not {r['group_id'] for r in chosen['train']}&{r['group_id'] for r in chosen['dev']}
    outputs={}
    for split,rows in chosen.items():
        path=a.output/f'{split}.jsonl';write_lines(path,rows)
        outputs[split]={'file':path.name,'sha256':sha(path),'rows':len(rows),'by_task':dict(Counter(r['task'] for r in rows)),
         'lengths':{t:{'max':max(r['length'] for r in rows if r['task']==t),'median':float(np.median([r['length'] for r in rows if r['task']==t]))} for t in TASKS},
         'classes':{t:dict(Counter(r['label'] for r in rows if r['task']==t)) for t in TASKS}}
    write_lines(a.output/'exclusions.jsonl',excluded)
    result={'status':'complete','tasks':TASKS,'outputs':outputs,'source_sha256':sources,'score_spec':spec,'max_length':a.max_length,
        'train_per_task':a.train_per_task,'available_before_cap':availability,'exclusions':dict(Counter(r['task']+':'+r['reason'] for r in excluded)),
        'cpt_data_manifest_sha256':sha(a.cpt_root/'data/manifest.json'),'cpt_checkpoint_sha256':sha(a.cpt_root/'round/cpt/model.safetensors'),
        'test_inference':False,'test_targets_used':False,'all_inputs_untruncated':True,'train_dev_exact_RC_overlap':0,
        'limitations':['One-seed engineering decision round; reused development sets.','GFP native split shares parent protein.','No full homology-independence claim.','8192 labels per task is a bounded data budget, not full-data saturation.']}
    write_json(a.output/'manifest.json',result);write_json(a.output/'status.json',{'status':'complete'})
    print(json.dumps({k:result[k] for k in ['outputs','available_before_cap','exclusions','score_spec']},indent=2),flush=True)

if __name__=='__main__':main()
