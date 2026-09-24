"""Inference-only development diagnostics; original labels are reference labels.

Residue shuffling is an out-of-distribution intervention, not a label-preserving
biological transformation. Original calibration temperatures remain fixed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

import laya_formal_experiment as core
from laya_direct_bpe import DirectBPE, digest
from laya_direct_experiment import make_items

ROOT=core.ROOT
SEEDS=(20260922,20260923,20260924)


def perturbed(row,variant):
    row=dict(row)
    if variant=='sequence_removed':row['sequence']=''
    elif variant.startswith('residue_shuffle_'):
        raw=row['sequence'];characters=list(raw)
        seed=int.from_bytes(hashlib.sha256(f"diagnostic-v1:{variant}:{row['id']}".encode()).digest()[:8],'big')
        random.Random(seed).shuffle(characters)
        row['sequence']=''.join(characters)
        if Counter(raw)!=Counter(row['sequence']):raise ValueError('Shuffle changed residue composition')
    return row


def permutation(row):
    order=list(range(len(row['choices'])))
    seed=int.from_bytes(hashlib.sha256(f"candidate-diagnostic-v1:{row['id']}".encode()).digest()[:8],'big')
    random.Random(seed).shuffle(order)
    if order==list(range(len(order))):order=order[1:]+order[:1]
    return order


def canonicalize(records,orders):
    result=[]
    for record in records:
        order=orders[record['id']]
        logits=[0.]*len(order)
        for position,canonical in enumerate(order):logits[canonical]=record['logits'][position]
        item=dict(record,logits=logits,label=order[record['label']])
        values=np.asarray(logits,dtype=float);exponent=np.exp(values-values.max())
        item['probs']=(exponent/exponent.sum()).tolist()
        result.append(item)
    return result


def compare(reference,changed):
    if [(r['id'],r['label'],r['task']) for r in reference]!=[(r['id'],r['label'],r['task']) for r in changed]:
        raise ValueError('Diagnostic membership/labels changed')
    out={}
    for task in sorted({r['task'] for r in reference}):
        pairs=[(a,b) for a,b in zip(reference,changed) if a['task']==task]
        p=np.asarray([a['probs'] for a,b in pairs]);q=np.asarray([b['probs'] for a,b in pairs])
        if not np.isfinite(p).all() or not np.isfinite(q).all():raise FloatingPointError('Non-finite diagnostic')
        middle=(p+q)/2
        js=.5*np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(middle,1e-300))+
                     q*np.log(np.maximum(q,1e-300)/np.maximum(middle,1e-300)),axis=1)
        out[task]={'n':len(pairs),'prediction_agreement':float((p.argmax(1)==q.argmax(1)).mean()),
                   'mean_probability_l1':float(np.abs(p-q).sum(1).mean()),
                   'mean_max_probability_difference':float(np.abs(p-q).max(1).mean()),
                   'mean_js_divergence':float(js.mean())}
    return out


def run_one(run_dir,out,batch_size=16):
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    saved=json.loads((run_dir/'summary.json').read_text())
    if not saved['formal'] or saved['test_access'] or not saved['checkpoint_reload_logits_match']:
        raise ValueError('Source checkpoint failed protocol gate')
    rep=DirectBPE(run_dir/'checkpoint/representation')
    eligible=core.load_eligible_ids(saved['eligible_id_filter'])
    rows=[r for r in core.load_split(ROOT/'artifacts/laya_formal_data','both','selection_dev') if r['id'] in eligible]
    _,build_model,builder=core.import_laya('vendor/laya')
    cfg=json.loads((run_dir/'checkpoint/rl_agent_config.json').read_text())
    model=core.fresh_reload(run_dir/'checkpoint',cfg,build_model,torch.device('cuda'))
    tokenizer=rep.expanded if saved['condition']=='full_bpe' else rep.base
    base_items=make_items(rows,rep,saved['condition'],builder,saved['seed'])
    baseline=core.evaluate(model,base_items,tokenizer,torch.device('cuda'),batch_size)
    previous=[json.loads(x) for x in (run_dir/'selection_dev_predictions.jsonl').read_text().splitlines()]
    if len(previous)!=len(baseline) or any(a['id']!=b['id'] or max(abs(x-y) for x,y in zip(a['logits'],b['logits']))>1e-5
                                          for a,b in zip(previous,baseline)):
        raise ValueError('Unperturbed predictions do not reproduce source result')
    temps={t:v['temperature'] for t,v in saved['calibration_temperature'].items()}
    result={'condition':saved['condition'],'seed':saved['seed'],'split':'selection_dev','test_access':False,
            'no_training':True,'reference_labels_not_assumed_valid_under_sequence_intervention':True,
            'source_summary_sha256':digest(run_dir/'summary.json'),
            'baseline_logits_reproduced':True,'temperature_refitted':False,
            'original':{'raw':core.metric_from_records(baseline),'calibrated':core.metric_from_records(baseline,temps)},
            'variants':{}}
    out.mkdir(parents=True,exist_ok=True)
    for variant in ('sequence_removed','residue_shuffle_0','residue_shuffle_1','residue_shuffle_2','candidate_permutation'):
        if variant=='candidate_permutation':
            items=[];orders={}
            for row in rows:
                order=permutation(row);orders[row['id']]=order
                ids,markers=rep.build(row,saved['condition'],builder,order)
                items.append({'ids':ids,'markers':markers,'label':order.index(row['label']),
                              'task':row['task'],'id':row['id']})
            predictions=canonicalize(core.evaluate(model,items,tokenizer,torch.device('cuda'),batch_size),orders)
            unchanged=None
        else:
            changed=[perturbed(row,variant) for row in rows]
            unchanged=sum(a['sequence']==b['sequence'] for a,b in zip(rows,changed))
            items=make_items(changed,rep,saved['condition'],builder,saved['seed'])
            predictions=core.evaluate(model,items,tokenizer,torch.device('cuda'),batch_size)
        result['variants'][variant]={'reference_label_metrics':{'raw':core.metric_from_records(predictions),
                                     'calibrated':core.metric_from_records(predictions,temps)},
                                    'comparison_to_original':compare(baseline,predictions),
                                    'unchanged_sequence_count':unchanged}
        (out/f'{variant}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        print(json.dumps({'condition':saved['condition'],'seed':saved['seed'],'variant':variant,
                          'comparison':result['variants'][variant]['comparison_to_original']}),flush=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    del model
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/laya_controls/sequence_diagnostics')
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--conditions',nargs='+',choices=['raw','full_bpe'],default=['full_bpe','raw'])
    args=p.parse_args()
    for seed in args.seeds:
        for condition in args.conditions:
            run_one(ROOT/f'artifacts/laya_direct_legacy/{condition}_seed{seed}',
                    args.output_dir/f'{condition}_seed{seed}')
    print('All requested development diagnostics completed.',flush=True)


if __name__=='__main__':main()
