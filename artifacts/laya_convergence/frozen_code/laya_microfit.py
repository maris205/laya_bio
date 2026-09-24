#!/usr/bin/env python3
"""Memorization diagnostic on fixed 32-example training subsets; no dev/test read."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoTokenizer
import laya_multitask_experiment as exp
import laya_multitask_data as data


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def select_training_subset(rows, task, count=32):
    pool=[r for r in rows if r['task']==task]
    classes=sorted(set(r['label'] for r in pool));chosen=[]
    for index,label in enumerate(classes):
        n=count//len(classes)+int(index<count%len(classes))
        candidates=sorted([r for r in pool if r['label']==label],key=lambda r:data.sha('microfit-v1:'+r['id']))
        if len(candidates)<n:raise ValueError('Not enough examples in class')
        chosen.extend(candidates[:n])
    chosen.sort(key=lambda r:data.sha('microfit-order-v1:'+r['id']))
    assert len(chosen)==count and len({r['id'] for r in chosen})==count
    assert all(r['split']=='train' for r in chosen)
    return chosen


def canonicalize(records, entities, kind):
    if kind!='candidate':return records
    items={i['id']:i for e in entities for i in e['items']};out=[]
    for row in records:
        order=items[row['id']]['order'];p=[0.]*len(order)
        for j,k in enumerate(order):p[k]=row['probs'][j]
        out.append({**row,'probs':p,'label':order[row['label']]})
    return out


def diagnostic_metrics(records,task_info):
    result=exp.metrics(records,task_info)
    for task,metric in result.items():
        rr=[r for r in records if r['task']==task];p=np.array([r['probs'] for r in rr])
        metric['predicted_class_counts']=np.bincount(p.argmax(1),minlength=p.shape[1]).tolist()
        metric['probability_std_mean']=float(p.std(0).mean())
        for field in ['score_prediction','scalar_prediction']:
            if field in rr[0]:metric[field]['prediction_std']=float(np.std([r[field] for r in rr]))
    return result


def fit_passed(metric):
    # This is categorical memorization, not a generalization threshold.
    return metric['accuracy']>=31/32 and metric['nll']<=.15


def compare_training_prefix(current, reference):
    """Ignore timing, but require all recorded optimization values to replay exactly."""
    if len(current)!=len(reference):raise ValueError('Prefix lengths differ')
    keys=['step','loss','lr','gradient_norm','gradient_norms_by_module']
    mismatches=[x['step'] for x,y in zip(current,reference) if any(x[k]!=y[k] for k in keys)]
    return {'updates_compared':len(reference),'exact_optimization_trace_match':not mismatches,
            'mismatch_steps':mismatches,'max_loss_difference':max(abs(x['loss']-y['loss']) for x,y in zip(current,reference)),
            'max_gradient_norm_difference':max(abs(x['gradient_norm']-y['gradient_norm']) for x,y in zip(current,reference))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data-dir','model-dir','laya-repo','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--task',choices=['splice','fluorescence'],required=True)
    p.add_argument('--kind',choices=['candidate','shared_heads'],required=True)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--updates',type=int,default=128)
    p.add_argument('--seed',type=int,default=20260924)
    p.add_argument('--resample-choice-order',action='store_true')
    p.add_argument('--pooling',choices=['cls','mean'],default='cls')
    p.add_argument('--score-loss',choices=['joint','ce','mse'],default='joint')
    p.add_argument('--probe-readout',action='store_true')
    p.add_argument('--bypass-shared-head',action='store_true')
    p.add_argument('--native-readout',action='store_true')
    p.add_argument('--discard-checkpoint-after-verification',action='store_true')
    p.add_argument('--eval-every',type=int,default=0,help='Add periodic training-only evaluations; zero preserves the historical schedule')
    p.add_argument('--reference-run',type=Path,help='Require an exact replay of this completed run before extending its update budget')
    a=p.parse_args()
    if a.updates<1 or a.eval_every<0:p.error('Updates must be positive and eval-every nonnegative')
    if a.resample_choice_order and (a.task!='splice' or a.kind!='candidate'):p.error('Order-resampling arm is defined only for splice candidate scoring')
    if (a.pooling!='cls' or a.score_loss!='joint' or a.probe_readout or a.bypass_shared_head or a.native_readout) and (a.kind!='shared_heads' or a.task!='fluorescence'):p.error('Readout/loss diagnostics require GFP shared heads')
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(8);exp.core.set_seed(a.seed);device=torch.device('cuda')
    manifest_path=a.data_dir/'manifest.json';manifest=json.loads(manifest_path.read_text())
    train_path=a.data_dir/manifest['outputs']['train']['file'];assert sha_file(train_path)==manifest['outputs']['train']['sha256']
    rows=[json.loads(l) for l in train_path.read_text().splitlines()]
    selected=select_training_subset(rows,a.task)
    info={}
    for task in data.TASKS:
        rr=[r for r in rows if r['task']==task];r=rr[0]
        spec={'primitive':r['primitive'],'n_outputs':len(data.LOCATIONS) if r['primitive']=='multilabel' else len(r['choices'])}
        if r['primitive']=='score':
            values=np.array([x['value'] for x in rr]);spec.update(manifest['score_spec'],mean=float(values.mean()),std=max(1e-6,float(values.std())))
        info[task]=spec
    _,_,builder=exp.core.import_laya(str(a.laya_repo));tok=AutoTokenizer.from_pretrained(a.model_dir/'tokenizer')
    training=exp.make_items(selected,tok,builder,a.kind,a.seed,True,manifest['max_length'])
    canonical=exp.make_items(selected,tok,builder,a.kind,a.seed,False,manifest['max_length'])
    a.output.mkdir(parents=True)
    reference=json.loads((a.reference_run/'summary.json').read_text()) if a.reference_run else None
    reference_updates=reference['actual_updates'] if reference else -1
    if reference and (not reference['complete'] or a.updates<=reference_updates):raise ValueError('Reference must be a completed shorter run')
    config={'kind':a.kind,'task':a.task,'lr':a.lr,'updates':a.updates,'seed':a.seed,'effective_batch':32,'micro_batch':4,
            'schedule':'linear warmup 8 updates, then constant learning rate','weight_decay':.01,'clip_norm':1.,
            'choice_order_policy':'resample independently per example and update' if a.resample_choice_order else 'one fixed permutation per example',
            'pooling':a.pooling,'score_loss':a.score_loss,'probe_readout':a.probe_readout,
            'eval_every':a.eval_every,'reference_run':str(a.reference_run) if a.reference_run else None,
            'bypass_shared_head':a.bypass_shared_head,'native_readout':a.native_readout,
            'checkpoint_retained':not a.discard_checkpoint_after_verification,
            'classification_supervised':a.kind=='candidate' or a.score_loss!='mse',
            'scalar_supervised':a.task=='fluorescence' and a.kind=='shared_heads' and a.score_loss!='ce',
            'scalar_fit_threshold':'standardized training RMSE <= 0.15',
            'dropout':'original model train mode','precision':'BF16 autocast, FP32 weights',
            'fit_threshold':'training-panel accuracy >=31/32 AND NLL <=0.15; no early stopping',
            'dev_access':False,'test_access':False,'selection':'stratified by training label; SHA256 microfit-v1:id, deterministic class quotas',
            'selected_ids':[r['id'] for r in selected],'label_counts':dict(Counter(r['label'] for r in selected)),
            'subset_sha256':data.sha('\n'.join(json.dumps(r,sort_keys=True) for r in selected)),
            'manifest_sha256':sha_file(manifest_path),'script_sha256':sha_file(Path(__file__)),
            'train_source_sha256':sha_file(train_path),'task_info':info,'gpu':torch.cuda.get_device_name(),
            'limitations':['Training memorization diagnostic only; no generalization evaluation.',
                           'Subset is deliberately class-balanced; not representative prevalence.',
                           'Score bins, anchors and scalar normalization retain the prior 1,024-training-row specification.']}
    if reference:
        matched_keys=['kind','task','lr','seed','effective_batch','micro_batch','schedule','weight_decay','clip_norm',
                      'dropout','precision','pooling','score_loss','probe_readout','native_readout','bypass_shared_head',
                      'subset_sha256','manifest_sha256','train_source_sha256','task_info','selected_ids','choice_order_policy']
        for key in matched_keys:
            if config[key]!=reference[key]:raise ValueError('Reference configuration differs: '+key)
        config['reference_matched_configuration_keys']=matched_keys
    (a.output/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    model,cfg=exp.build(a.kind,a.model_dir,a.laya_repo,info,device,pooling=a.pooling,score_loss=a.score_loss,bypass_head=a.bypass_shared_head,native_readout=a.native_readout)
    config['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({'event':'loaded','task':a.task,'kind':a.kind,'lr':a.lr,'trainable_parameters':config['trainable_parameters']}),flush=True)
    history=[]
    def evaluate_step(step):
        records,_=exp.evaluate(model,a.kind,training,tok,device,info,8)
        records=canonicalize(records,training,a.kind)
        can,_=exp.evaluate(model,a.kind,canonical,tok,device,info,8)
        rec={'step':step,'training_panel':diagnostic_metrics(records,info)[a.task],
             'canonical_panel':diagnostic_metrics(can,info)[a.task]}
        if a.probe_readout and step in {0,reference_updates,a.updates}:
            import laya_readout_probe
            rec['readout_probe']=laya_readout_probe.probe(model,canonical,tok,device,info)
        history.append(rec)
        with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps({'event':'evaluation',**rec}),flush=True)
        return records,can
    evaluate_step(0)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01);rng=random.Random(a.seed)
    rows_by_id={r['id']:r for r in selected}
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();trace=[]
    for step in range(1,a.updates+1):
        model.train()
        pool=exp.resample_choice_batch(training,rows_by_id,tok,builder,a.seed,step,manifest['max_length']) if a.resample_choice_order else training
        chunk=pool.copy();rng.shuffle(chunk);items=[i for e in chunk for i in e['items']]
        optimizer.zero_grad(set_to_none=True);loss_value=0.;lr=a.lr*min(1.,step/8)
        for group in optimizer.param_groups:group['lr']=lr
        for offset in range(0,len(items),4):
            loss=exp.loss_values(model,a.kind,items[offset:offset+4],tok,device,info).sum()/32
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite microfit loss')
            loss.backward();loss_value+=float(loss.detach())
        grad={}
        if step in {1,32,64,reference_updates,a.updates}:
            for prefix in ['encoder','head','type_emb','scorer','readout','classifiers','regressors']:
                values=[p.grad.detach().norm().square() for n,p in model.named_parameters() if n.startswith(prefix+'.') and p.grad is not None]
                if values:grad[prefix]=float(torch.stack(values).sum().sqrt())
        norm=float(nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step()
        rec={'step':step,'loss':loss_value,'lr':lr,'gradient_norm':norm,'gradient_norms_by_module':grad,'seconds':time.monotonic()-started}
        trace.append(rec)
        with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        if step in {1,8,16,32,64,96,reference_updates,a.updates} or (a.eval_every and step%a.eval_every==0):
            _,step_predictions=evaluate_step(step)
            if step==reference_updates:
                old_trace=[json.loads(l) for l in (a.reference_run/'training.jsonl').read_text().splitlines()]
                check=compare_training_prefix(trace,old_trace)
                old_predictions=[json.loads(l) for l in (a.reference_run/'canonical_panel_predictions.jsonl').read_text().splitlines()]
                check['exact_prediction_match']=step_predictions==old_predictions
                check['exact_final_metrics_match']=history[-1]['canonical_panel']==reference['final']['canonical_panel']
                check['configuration_match']=True
                check['max_probability_difference']=max(abs(x-y) for r,s in zip(step_predictions,old_predictions) for x,y in zip(r['probs'],s['probs']))
                check['max_scalar_difference']=max([abs(r['scalar_prediction']-s['scalar_prediction']) for r,s in zip(step_predictions,old_predictions) if 'scalar_prediction' in r] or [0.])
                (a.output/'prefix_check.json').write_text(json.dumps(check,indent=2)+'\n')
                if not all(check[k] for k in ['exact_optimization_trace_match','exact_prediction_match','exact_final_metrics_match']):raise RuntimeError('Reference replay failed; refusing to extend training')
                (a.output/'prefix_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in step_predictions))
    seconds=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()/2**30
    fitted,can=evaluate_step(a.updates) if history[-1]['step']!=a.updates else (None,None)
    # Save final predictions once, including input-only counterfactual evaluation.
    fitted,_=exp.evaluate(model,a.kind,training,tok,device,info,8);fitted=canonicalize(fitted,training,a.kind)
    can,_=exp.evaluate(model,a.kind,canonical,tok,device,info,8)
    rotated=[{**r,'inputs':selected[(i+1)%len(selected)]['inputs']} for i,r in enumerate(selected)]
    rotated_items=exp.make_items(rotated,tok,builder,a.kind,a.seed,False,manifest['max_length'])
    rotated_predictions,_=exp.evaluate(model,a.kind,rotated_items,tok,device,info,8)
    for name,records in [('training_panel',fitted),('canonical_panel',can),('rotated_sequences',rotated_predictions)]:
        (a.output/(name+'_predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    checkpoint=a.output/'checkpoint';checkpoint.mkdir()
    save_file({k:v.detach().contiguous().cpu() for k,v in model.state_dict().items()},str(checkpoint/'model.safetensors'))
    (checkpoint/'rl_agent_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (checkpoint/'task_info.json').write_text(json.dumps(info,indent=2)+'\n')
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    model,_=exp.build(a.kind,a.model_dir,a.laya_repo,info,device,checkpoint)
    reload,_=exp.evaluate(model,a.kind,canonical,tok,device,info,8)
    assert len(reload)==len(can) and all(x['id']==y['id'] for x,y in zip(reload,can))
    prob_error=max(abs(x-y) for r,s in zip(reload,can) for x,y in zip(r['probs'],s['probs']))
    scalar_error=max([abs(r['scalar_prediction']-s['scalar_prediction']) for r,s in zip(reload,can) if 'scalar_prediction' in r] or [0.])
    assert prob_error<1e-6 and scalar_error<1e-6
    final=history[-1]
    summary={**config,'complete':True,'actual_updates':len(trace),'entity_presentations':32*len(trace),
        'training_and_periodic_eval_seconds':seconds,'peak_allocated_gib':peak,
        'initial':history[0],'final':final,'training_panel_fit_passed':fit_passed(final['training_panel']) if config['classification_supervised'] else None,
        'canonical_panel_fit_passed':fit_passed(final['canonical_panel']) if config['classification_supervised'] else None,
        'scalar_fit_passed':final['canonical_panel']['scalar_prediction']['rmse']/info[a.task]['std']<=.15 if a.task=='fluorescence' and config['scalar_supervised'] else None,
        'rotated_sequences':diagnostic_metrics(rotated_predictions,info)[a.task],
        'reload_max_probability_error':prob_error,'reload_max_scalar_error':scalar_error,
        'checkpoint_sha256':sha_file(checkpoint/'model.safetensors')}
    if a.discard_checkpoint_after_verification:(checkpoint/'model.safetensors').unlink()
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
