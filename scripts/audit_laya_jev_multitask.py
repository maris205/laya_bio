#!/usr/bin/env python3
"""Independent file, prediction and schedule audit of the JEV decision round."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score,log_loss,mean_absolute_error,mean_squared_error
from safetensors import safe_open
import torch
from laya_biocpt_data import sha,digest,read_lines,write_json,key,kmers
from laya_jev_multitask_data import Representation

TASKS=['promoter','structural_class','fluorescence']
RUNS=['cpt_joint','no_cpt_joint','cpt_promoter','cpt_structural_class','cpt_fluorescence']


def load(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cpt-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root/'round';assert load(root/'status.json')['status']=='complete'
    recipe=load(root/'manifest.json');manifest=load(a.root/'data/manifest.json');spec=manifest['score_spec'];hashed=0
    for name,h in recipe['code_sha256'].items():assert sha(root/'frozen_code'/name)==h;hashed+=1
    assert sha(a.root/'data/manifest.json')==recipe['data_manifest_sha256']
    assert sha(a.cpt_root/'round/cpt/model.safetensors')==recipe['cpt_checkpoint_sha256']
    assert sha(a.cpt_root.parent/'laya_model/model.safetensors')==recipe['initial_model_sha256']
    assert sha(a.cpt_root/'data/manifest.json')==manifest['cpt_data_manifest_sha256']
    for path,h in manifest['source_sha256'].items():assert sha(Path(path))==h;hashed+=1
    gfp_bank=set()
    for path in manifest['source_sha256']:
        if Path(path).name in ['fluorescence_train.json','fluorescence_valid.json']:
            for r in load(Path(path)):gfp_bank.update(kmers(r['primary'].upper(),15))
    cpt_protein_rows=0
    for split in ['train','validation']:
        for row in read_lines(a.cpt_root/f'data/raw_{split}.jsonl'):
            if row['modality']=='protein':
                cpt_protein_rows+=1
                assert not any(k in gfp_bank for k in kmers(row['content'],15))
    admission=load(a.root/'data/cpt_admission.json')
    assert admission['status']=='pass' and admission['GFP_native_train_and_valid_15mers']==len(gfp_bank)
    assert admission['CPT_protein_snapshots_checked']==cpt_protein_rows
    rows={split:read_lines(a.root/'data'/manifest['outputs'][split]['file']) for split in ['train','dev']}
    for split in rows:assert sha(a.root/'data'/manifest['outputs'][split]['file'])==manifest['outputs'][split]['sha256'];hashed+=1
    assert not {r['group_id'] for r in rows['train']}&{r['group_id'] for r in rows['dev']}
    representation=Representation(a.cpt_root/'data');nrows=0
    for split,pool in rows.items():
        assert len({r['id'] for r in pool})==len(pool)
        assert dict(Counter(r['task'] for r in pool))==manifest['outputs'][split]['by_task']
        for r in pool:
            assert r['group_id']==r['modality']+':'+key(r['sequence'],r['modality'])
            rebuilt=representation.prepare(r)
            for field in ['head_ids','body_ids','option_ids','length','qtype']:assert r[field]==rebuilt[field]
            assert r['length']<=manifest['max_length'];nrows+=1
            if r['primitive']=='score':
                q=np.asarray(r['target_probs']);assert np.all(q>=0);np.testing.assert_allclose(q.sum(),1,atol=1e-12,rtol=0)
                assert np.count_nonzero(q)>0 and np.count_nonzero(q)<=2
                np.testing.assert_allclose(q@np.asarray(spec['anchors']),np.clip(r['value'],spec['train_min'],spec['train_max']),atol=1e-12,rtol=0)
    values=np.array([r['value'] for r in rows['train'] if r['task']=='fluorescence'])
    np.testing.assert_allclose([values.mean(),np.median(values),values.std(),values.min(),values.max()],
       [spec['train_mean'],spec['train_median'],spec['train_std'],spec['train_min'],spec['train_max']],atol=1e-12,rtol=0)
    np.testing.assert_allclose(spec['anchors'],np.linspace(values.min(),values.max(),5),atol=1e-12,rtol=0)
    # Replay the frozen sampler and candidate augmentation; the actual trainer
    # also asserts exactly three exposures for every visited entity.
    sys.path.insert(0,str(root/'frozen_code'))
    modspec=importlib.util.spec_from_file_location('frozen_jev_train',root/'frozen_code/laya_jev_multitask_train.py')
    frozen=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(frozen)
    full_schedule=list(frozen.schedule(rows['train']));assert len(full_schedule)==1152
    replay_exposures=Counter(r['id'] for _,_,_,rr in full_schedule for r in rr)
    assert set(replay_exposures)=={r['id'] for r in rows['train']} and set(replay_exposures.values())=={3}
    order_hash={t:hashlib.sha256() for t in TASKS}
    for _,_,task,rr in full_schedule:
        for r in rr:order_hash[task].update((r['id']+'\n').encode())
    checks=0;prediction_files=0;roundoff=0.;configs={};initial_predictions={};retained=[]
    for name in RUNS:
        folder=root/name;cfg=load(folder/'run_config.json');configs[name]=cfg;status=load(folder/'status.json')
        assert status['status']=='complete' and status['checkpoint_reload_exact']
        assert cfg['data_manifest_sha256']==recipe['data_manifest_sha256']
        assert cfg['initial_model_sha256']==recipe['initial_model_sha256']
        assert cfg['cpt_checkpoint_sha256']==(recipe['cpt_checkpoint_sha256'] if cfg['arm']=='cpt' else None)
        active=TASKS if cfg['task']=='joint' else [cfg['task']]
        expected=[r for r in full_schedule if r[2] in active];trace=read_lines(folder/'training_trace.jsonl')
        assert len(trace)==len(expected)==status['updates']==cfg['updates']
        assert status['per_entity_exposure_histogram']=={'3':8192*len(active)}
        assert status['presentations_by_task']=={t:24576 if t in active else 0 for t in TASKS}
        assert cfg['batch_order_sha256_by_task']=={t:h.hexdigest() for t,h in order_hash.items()}
        ph=hashlib.sha256()
        for index,((global_step,epoch,task,rr),record) in enumerate(zip(expected,trace),1):
            assert (record['step'],record['joint_schedule_step'],record['epoch'],record['task'])==(index,global_step,epoch,task)
            assert all(math.isfinite(record[k]) for k in ['loss','gradient_norm','encoder_gradient_norm','scorer_gradient_norm'])
            np.testing.assert_allclose(record['encoder_lr'],2e-5*frozen.lr_factor(global_step,1152),atol=1e-15,rtol=0)
            for row in rr:
                presented=frozen.render(row,epoch,True);ph.update(json.dumps(presented['order']).encode())
                assert presented['order'][presented['presented_label']]==row['label']
                assert all(presented['ids'][m]==representation.tokenizer.mask_token_id for m in presented['markers'])
        assert ph.hexdigest()==cfg['training_panel_order_sha256']
        if status['model_retained']:
            assert sha(folder/'model.safetensors')==status['checkpoint_sha256'];retained.append(name)
            with safe_open(str(folder/'model.safetensors'),framework='pt',device='cpu') as f:
                assert not any(k.startswith(('classifiers.','regressors.')) for k in f.keys())
        curves=read_lines(folder/'learning_curve.jsonl')
        assert {(r['epoch'],r['split']) for r in curves}=={(0,'dev'),(1,'dev'),(2,'dev'),(3,'dev'),(0,'train'),(3,'train')}
        initial_predictions[name]={}
        for point in curves:
            split=point['split'];path=folder/f'{split}_epoch{point["epoch"]}_predictions.jsonl';pred=read_lines(path);prediction_files+=1
            source=[]
            for t in active:
                pool=[r for r in rows[split] if r['task']==t]
                if split=='train':pool=sorted(pool,key=lambda r:digest('train-eval:'+r['id']))[:1024]
                source.extend(pool)
            assert [(r['id'],r['label'],r['group_id']) for r in pred]==[(r['id'],r['label'],r['group_id']) for r in source]
            by_id={r['id']:r for r in source}
            for task,metric in point['metrics'].items():
                rr=[r for r in pred if r['task']==task];truth=np.array([r['label'] for r in rr]);probs=np.array([r['probs'] for r in rr]);arg=probs.argmax(-1)
                assert np.all(np.isfinite(probs)) and np.all(probs>=0) and np.all(probs<=1)
                roundoff=max(roundoff,float(np.abs(probs.sum(1)-1).max()));np.testing.assert_allclose(probs.sum(1),1,atol=1e-6,rtol=0)
                assert np.array_equal(arg,[r['prediction'] for r in rr]);assert metric['rows']==len(rr)
                normalized=probs/probs.sum(1,keepdims=True)
                independent=[accuracy_score(truth,arg),f1_score(truth,arg,labels=list(range(probs.shape[1])),average='macro',zero_division=0),log_loss(truth,normalized,labels=list(range(probs.shape[1])))]
                np.testing.assert_allclose(independent,[metric['accuracy'],metric['macro_f1'],metric['nll']],atol=1e-7,rtol=1e-6)
                assert confusion_matrix(truth,arg,labels=list(range(probs.shape[1]))).tolist()==metric['confusion']
                for r in rr:
                    original=by_id[r['id']];out=r['output'];assert out['type']==original['primitive'] and out['probabilities']==r['probs'] and r['output_valid']
                    if out['type']=='choice':assert out['index']==r['prediction'] and out['answer']==original['choices'][r['prediction']]
                    elif out['type']=='noul':assert type(out['answer']) is bool and out['answer']==bool(r['prediction'])
                    else:
                        assert r['value']==original['value'];assert out['level']==r['prediction'] and out['units']=='log fluorescence'
                        np.testing.assert_allclose(r['score_prediction'],np.dot(r['probs'],spec['anchors']),atol=1e-12,rtol=0)
                        assert out['value']==r['score_prediction']
                        assert spec['train_min']-1e-5<=out['value']<=spec['train_max']+1e-5
                assert metric['output_valid_fraction']==1.
                if task=='fluorescence':
                    target=np.array([r['value'] for r in rr]);score=np.array([r['score_prediction'] for r in rr]);rho=spearmanr(target,score).statistic
                    expected_values=[np.sqrt(mean_squared_error(target,score)),mean_absolute_error(target,score),score.std()]
                    np.testing.assert_allclose(expected_values,[metric['rmse'],metric['mae'],metric['prediction_std']],atol=1e-12,rtol=1e-10)
                    if np.isfinite(rho):np.testing.assert_allclose(rho,metric['spearman'],atol=1e-12,rtol=0)
                    else:assert metric['spearman'] is None
                else:np.testing.assert_allclose(recall_score(truth,arg,labels=list(range(probs.shape[1])),average=None,zero_division=0),metric['per_class_recall'])
                if split=='dev' and point['epoch']==0:initial_predictions[name][task]=rr
                checks+=1
        reverse_path=folder/'dev_reversed_choice_predictions.jsonl'
        if reverse_path.exists():
            reverse=read_lines(reverse_path);base=[r for r in read_lines(folder/'dev_epoch3_predictions.jsonl') if r['task']=='structural_class']
            assert [(r['id'],r['label']) for r in reverse]==[(r['id'],r['label']) for r in base]
            rm=load(folder/'reversed_choice_metrics.json')['structural_class'];pr=np.array([r['probs'] for r in reverse]);yy=[r['label'] for r in reverse]
            np.testing.assert_allclose([accuracy_score(yy,pr.argmax(1)),f1_score(yy,pr.argmax(1),labels=list(range(7)),average='macro',zero_division=0)],
                                      [rm['accuracy'],rm['macro_f1']],atol=1e-12,rtol=0)
    assert len({c['initial_shared_head_sha256'] for c in configs.values()})==1
    assert configs['cpt_joint']['training_panel_order_sha256']==configs['no_cpt_joint']['training_panel_order_sha256']
    for task in TASKS:assert initial_predictions['cpt_joint'][task]==initial_predictions['cpt_'+task][task]
    resume=torch.load(root/'cpt_joint/resume.pt',map_location='cpu',weights_only=False,mmap=True)
    assert resume['completed_epochs']==3 and resume['completed_updates']==1152
    assert {'optimizer','torch_rng','cuda_rng','python_rng','numpy_rng'}.issubset(resume)
    assert {float(v['step']) for v in resume['optimizer']['state'].values()}=={1152.}
    result={'status':'pass','hashed_source_code_files':hashed,'encoded_rows_reconstructed':nrows,'metric_points_recomputed':checks,
       'prediction_files_checked':prediction_files,'max_saved_probability_sum_roundoff':roundoff,'continuous_training_targets_preserve_native_values':True,
       'GFP_CPT_15mer_admission_independently_recomputed':True,'CPT_protein_rows_rechecked':cpt_protein_rows,
       'all_typed_outputs_valid':True,'retained_model_hashes_verified':retained,'shared_head_initialization_identical_all_runs':True,
       'CPT_single_and_joint_initial_dev_predictions_identical':True,'frozen_schedules_and_panel_hashes_reproduced':True,
       'actual_trainer_exposure_histograms_match_three_epochs':True,'no_task_specific_heads_in_retained_models':True,
       'CPT_joint_resume_optimizer_step':1152,'CPT_joint_resume_parameter_states':len(resume['optimizer']['state']),
       'test_inference':False,'NLL_check':'FP32 probability sum roundoff normalized only for independent FP64 sklearn log_loss.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
