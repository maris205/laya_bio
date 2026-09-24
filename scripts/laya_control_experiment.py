"""Full-BPE fixed-class readout and trained text-only controls, without CPT/test.

B1 retains the pretrained Laya encoder, two contextual head blocks, type
embedding and readout MLP. It pools CLS and replaces the scalar candidate scorer
with learned 2/7-class output matrices. Candidate label strings are absent.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import shutil

import torch
from torch import nn
from safetensors.torch import load_file

import laya_formal_experiment as core
from laya_direct_bpe import DirectBPE, digest
import laya_direct_experiment as direct


class FixedClassModel(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.encoder=base.encoder
        self.head=base.head
        self.type_emb=base.type_emb
        self.readout=nn.Sequential(*list(base.scorer.children())[:-1])
        d=base.encoder.config.hidden_size
        self.classifiers=nn.ModuleDict({'2':nn.Linear(d,2),'7':nn.Linear(d,7)})

    def forward(self,input_ids,attention_mask,marker_pos,marker_mask,qtype):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        h=h+self.type_emb(qtype)[:,None,:]
        if self.head is not None:
            pad=~attention_mask.bool()
            for layer in self.head.layers:h=layer(h,src_key_padding_mask=pad)
        pooled=self.readout(h[:,0])
        sizes=marker_mask.sum(-1)
        if not torch.all((sizes==2)|(sizes==7)):raise ValueError('Unknown task class count')
        logits=pooled.new_full(marker_mask.shape,-1e4)
        for k in (2,7):
            selected=sizes==k
            if selected.any():logits[selected,:k]=self.classifiers[str(k)](pooled[selected])
        logits=logits.float().masked_fill(~marker_mask,-1e4)
        return logits,logits.new_zeros((len(input_ids),2))


def make_items(rows,rep,kind,builder,seed,training=False):
    if kind=='text_only':
        empty=[dict(row,sequence='') for row in rows]
        return direct.make_items(empty,rep,'full_bpe',builder,seed,shuffle=training)
    items=[]
    for row in rows:
        prefix=f"Task context: {row['context']}\nSequence: "
        prefix_ids=rep.base(prefix.replace(rep.base.mask_token,' '),add_special_tokens=False)['input_ids']
        ids=[rep.base.cls_token_id]+prefix_ids+rep.sequence_ids(row['sequence'],row['kind'])+[rep.base.sep_token_id]
        if len(ids)>1024:raise ValueError('Fixed-head input exceeds context budget')
        items.append({'ids':ids,'markers':[0]*len(row['choices']),'qtype':0,'label':row['label'],
                      'id':row['id'],'task':row['task'],'n_tokens':len(ids),'full_tokens':len(ids),'truncated':False})
    return items


def fresh_reload(checkpoint,cfg,build_model,device,kind):
    if kind=='text_only':return core.fresh_reload(checkpoint,cfg,build_model,device)
    base=build_model(cfg,encoder_dir=str(checkpoint/'encoder'))
    model=FixedClassModel(base)
    model.load_state_dict(load_file(str(checkpoint/'model.safetensors')),strict=True)
    return model.to(device).eval()


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['b1','text_only'],required=True)
    p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--eligible-ids',default='artifacts/laya_formal_data/eligible_ids.json')
    p.add_argument('--smoke',action='store_true')
    return p.parse_args()


def main():
    a=parse_args()
    if (a.output_dir/'summary.json').exists():raise FileExistsError('Completed control already exists')
    core.set_seed(a.seed)
    rep=DirectBPE()
    rows=direct.load_rows(a)
    names={}
    for split,values in rows.items():
        for row in values:
            if row['task'] in names and names[row['task']]!=row['choices']:
                raise ValueError('Canonical class order changed')
            names[row['task']]=row['choices']
    _,build_model,builder=core.import_laya('vendor/laya')
    items={s:make_items(values,rep,a.kind,builder,a.seed,s=='train') for s,values in rows.items()}
    item_stats={s:direct.stats(values) for s,values in items.items()}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'kind':a.kind,'seed':a.seed,'item_stats':item_stats},indent=2),flush=True)
    source=core.ROOT/'artifacts/laya_model'
    cfg=json.loads((source/'rl_agent_config.json').read_text())
    cfg.update(max_len=1024,head_max_len=256,gradient_checkpointing=True,control_type=a.kind,class_names=names)
    device=torch.device('cuda')
    base=core.load_model(source,cfg,build_model,device,trainable=True)
    expansion=direct.initialize(base,rep)
    core.set_seed(a.seed)
    if a.kind=='b1':
        model=FixedClassModel(base).to(device)
        del base
        probe=direct.gradient_probe(model,rep,items['train'],device)
    else:
        model=base
        probe=None
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    core.set_seed(a.seed)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    training=core.train_model(model,items['train'],rep.expanded,device,2 if a.smoke else 3005,
                             8,4,2e-5,.01,a.seed,25,3,.05)
    training['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    training['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    predictions={s:core.evaluate(model,items[s],rep.expanded,device,16) for s in ('selection_dev','calibration')}
    groups=defaultdict(list)
    for r in predictions['calibration']:groups[r['task']].append(r)
    temperatures={t:core.fit_metric_temperature([r['logits'] for r in v],[r['label'] for r in v],
                                               n_classes=v[0]['n_classes']) for t,v in groups.items()}
    temp_values={t:v['temperature'] for t,v in temperatures.items()}
    metrics={s:{'raw':core.metric_from_records(v),'calibrated':core.metric_from_records(v,temp_values)}
             for s,v in predictions.items()}
    if any(not torch.isfinite(torch.tensor(r['logits'])).all() for v in predictions.values() for r in v):
        raise FloatingPointError('Non-finite control prediction')
    checkpoint=a.output_dir/'checkpoint'
    core.save_checkpoint(model,rep.expanded,cfg,checkpoint)
    shutil.copytree(rep.root,checkpoint/'representation')
    disk_rep=DirectBPE(checkpoint/'representation')
    disk_items=make_items(rows['selection_dev'],disk_rep,a.kind,builder,a.seed)
    if disk_items!=items['selection_dev']:raise ValueError('Control reload IDs/labels mismatch')
    del model
    if a.kind=='text_only':del base
    gc.collect();torch.cuda.empty_cache()
    reloaded=fresh_reload(checkpoint,cfg,build_model,device,a.kind)
    reloaded_predictions=core.evaluate(reloaded,disk_items,disk_rep.expanded,device,16)
    match=len(reloaded_predictions)==len(predictions['selection_dev']) and all(
        x['id']==y['id'] and len(x['logits'])==len(y['logits']) and
        max(abs(a-b) for a,b in zip(x['logits'],y['logits']))<1e-5
        for x,y in zip(reloaded_predictions,predictions['selection_dev']))
    if not match:raise ValueError('Control checkpoint reload changed logits')
    for split,records in predictions.items():
        (a.output_dir/f'{split}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary={'formal':not a.smoke,'smoke_only':a.smoke,'kind':a.kind,'condition':'full_bpe','seed':a.seed,
             'no_cpt':True,'test_access':False,'evaluated_splits':['selection_dev','calibration'],
             'class_names':names,'n_rows':{s:len(v) for s,v in rows.items()},'item_stats':item_stats,
             'training':training,'trainable_parameters':trainable,'embedding_trainable':True,
             'embedding_receives_sequence_supervision':a.kind=='b1','expansion':expansion,
             'new_embedding_gradient_probe':probe,'evaluation':metrics,'calibration_temperature':temperatures,
             'calibration_temperature_fit_on':'calibration','checkpoint':str(checkpoint),
             'checkpoint_reload_logits_match':match,'checkpoint_reload_input_ids_match':True,
             'representation_sha256':digest(rep.root/'metadata.json'),
             'selection_policy':'final fixed-budget checkpoint',
             'b1_architecture':'pretrained encoder + contextual head + readout MLP; CLS pool; new 2/7-class output matrices' if a.kind=='b1' else None}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
