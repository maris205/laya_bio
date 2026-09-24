#!/usr/bin/env python3
"""Summarize typed pilot evidence with training-only constant baselines."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, matthews_corrcoef, accuracy_score, roc_auc_score, average_precision_score
from Bio.Align import PairwiseAligner
import laya_multitask_data as data
import laya_multitask_experiment as exp


def baselines(train,dev):
    task_info={};predictions=[];support={}
    for task in data.TASKS:
        tr=[r for r in train if r['task']==task];de=[r for r in dev if r['task']==task];r=tr[0]
        spec={'primitive':r['primitive'],'n_outputs':len(data.LOCATIONS) if r['primitive']=='multilabel' else len(r['choices'])}
        if r['primitive']=='multilabel':
            probs=np.array([x['labels'] for x in tr]).mean(0)
            support[task]={'train_positive_counts':np.array([x['labels'] for x in tr]).sum(0).tolist(),
                           'dev_positive_counts':np.array([x['labels'] for x in de]).sum(0).tolist()}
            for row in de:
                for i,p in enumerate(probs):
                    predictions.append({'id':row['id']+':'+str(i),'entity_id':row['id'],'task':task,'label_index':i,
                                        'label':int(row['labels'][i]),'probs':[float(1-p),float(p)]})
        else:
            counts=np.bincount([x['label'] for x in tr],minlength=spec['n_outputs']);probs=counts/counts.sum()
            support[task]={'train_class_counts':counts.tolist(),
                           'dev_class_counts':np.bincount([x['label'] for x in de],minlength=spec['n_outputs']).tolist()}
            if r['primitive']=='score':
                values=np.array([x['value'] for x in tr]);spec.update(anchors=r['anchors'],train_min=float(values.min()),train_max=float(values.max()))
            for row in de:
                rec={'id':row['id'],'entity_id':row['id'],'task':task,'label':row['label'],'probs':probs.tolist()}
                if r['primitive']=='score':rec.update(value=row['value'],score_prediction=float(values.mean()))
                predictions.append(rec)
        task_info[task]=spec
    result=exp.metrics(predictions,task_info)
    tr=np.array([r['value'] for r in train if r['primitive']=='score']);de=np.array([r['value'] for r in dev if r['primitive']=='score'])
    median=float(np.median(tr));result['fluorescence']['training_median']={'mae':float(mean_absolute_error(de,np.full(len(de),median))),
        'rmse':float(np.sqrt(mean_squared_error(de,np.full(len(de),median)))),'constant':median}
    aligner=PairwiseAligner(mode='global',match_score=1.,mismatch_score=-1.,open_gap_score=-2.,extend_gap_score=-.5)
    aligned={}
    for split,pool in [('train',train),('dev',dev)]:
        selected=[r for r in pool if r['task']=='homology_pair']
        scores=[aligner.score(r['inputs'][0]['sequence'],r['inputs'][1]['sequence']) / max(len(x['sequence']) for x in r['inputs']) for r in selected]
        aligned[split]=(np.array([r['label'] for r in selected]),np.array(scores))
    y,scores=aligned['train'];thresholds=np.unique(np.quantile(scores,np.linspace(0,1,101)))
    threshold=max(thresholds,key=lambda t:(matthews_corrcoef(y,scores>=t),float(t)))
    y,scores=aligned['dev'];prediction=scores>=threshold
    result['homology_alignment_baseline']={'description':'Global affine-gap alignment score / maximum sequence length; not a probability. Threshold maximizes train MCC over 101 train-score quantiles.',
        'match':1.,'mismatch':-1.,'gap_open':-2.,'gap_extend':-.5,'train_selected_threshold':float(threshold),
        'dev_accuracy':float(accuracy_score(y,prediction)),'dev_mcc':float(matthews_corrcoef(y,prediction)),
        'dev_auroc':float(roc_auc_score(y,scores)),'dev_auprc':float(average_precision_score(y,scores))}
    return {'description':'Training-only class/prevalence/mean constants; no sequence input. Score also reports the training median for MAE.',
            'metrics':result,'support':support}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--candidate',type=Path);p.add_argument('--shared-heads',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sets={s:[json.loads(line) for line in (a.data_dir/(s+'.jsonl')).open()] for s in ['train','dev']}
    result={'status':'engineering_pilot_not_confirmatory','constant_baselines':baselines(sets['train'],sets['dev'])}
    for kind,path in [('candidate',a.candidate),('shared_heads',a.shared_heads)]:
        if path:result[kind]=json.loads((path/'summary.json').read_text())
    if 'candidate' in result and 'shared_heads' in result:
        assert result['candidate']['data_manifest_sha256']==result['shared_heads']['data_manifest_sha256']
        assert result['candidate']['entity_presentations']==result['shared_heads']['entity_presentations']
        result['matched_entity_exposure']=True
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['constant_baselines']['metrics'],indent=2))

if __name__=='__main__':main()
