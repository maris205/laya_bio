#!/usr/bin/env python3
"""One shared Laya encoder/typed candidate scorer for three biological tasks."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file,save_file
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,mean_squared_error,mean_absolute_error
from scipy.stats import spearmanr
from laya_biocpt_data import sha,digest,read_lines,write_lines,write_json
import laya_biocpt_train as bio
from laya_jev_multitask_data import TASKS,interpolate

SEED=20260926
EPOCHS=3
BATCH=64


def append(path,record):
    with path.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')


class SharedDecision(nn.Module):
    """No task-specific parameters. One scorer evaluates all candidate markers."""
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder;d=encoder.config.hidden_size
        layer=nn.TransformerEncoderLayer(d,max(1,d//64),4*d,.1,batch_first=True,norm_first=True)
        self.head=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.type_emb=nn.Embedding(3,d)
        self.scorer=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
        self.head_checkpointing=True
    def forward(self,input_ids,attention_mask,marker_pos,marker_mask,qtype):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        h=h+self.type_emb(qtype)[:,None,:]
        for layer in self.head.layers:
            if self.training and self.head_checkpointing:
                h=checkpoint(layer,h,src_key_padding_mask=~attention_mask.bool(),use_reentrant=False)
            else:h=layer(h,src_key_padding_mask=~attention_mask.bool())
        selected=torch.gather(h,1,marker_pos[:,:,None].expand(-1,-1,h.size(-1)))
        logits=self.scorer(selected).squeeze(-1).float()
        return logits.masked_fill(~marker_mask,-1e4)


def build(model_dir,cpt_root,arm):
    # Biological row initialization is exactly the completed CPT comparison's.
    mlm,_,_=bio.build(model_dir,cpt_root/'data',cpt_root/'round/cpt/model.safetensors' if arm=='cpt' else None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED+7);model=SharedDecision(mlm.model)
    del mlm
    original=load_file(str(model_dir/'model.safetensors'))
    for name in ['head','type_emb','scorer']:
        module=getattr(model,name)
        module.load_state_dict({k.removeprefix(name+'.'):v for k,v in original.items() if k.startswith(name+'.')},strict=True)
    del original
    return model.to('cuda')


def tensor_hash(values):
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def render(row,epoch=0,training=False,reverse=False):
    order=list(range(len(row['choices'])))
    if row['primitive']=='choice':
        if reverse:order.reverse()
        elif training:random.Random(int(digest(f'jev-choice-v1:{SEED}:{epoch}:{row["id"]}')[:16],16)).shuffle(order)
    ids=row['head_ids'].copy();marks=[]
    for j in order:marks.append(len(ids));ids.extend(row['option_ids'][j])
    ids.extend(row['body_ids'])
    assert len(ids)==row['length']
    return {**row,'ids':ids,'markers':marks,'order':order,'presented_label':order.index(row['label'])}


def pack(rows,pad,device='cuda'):
    width=max(len(r['ids']) for r in rows);k=max(len(r['markers']) for r in rows)
    ids=torch.full((len(rows),width),pad,dtype=torch.long);att=torch.zeros_like(ids)
    marks=torch.zeros((len(rows),k),dtype=torch.long);mask=torch.zeros_like(marks,dtype=torch.bool)
    for i,r in enumerate(rows):
        ids[i,:len(r['ids'])]=torch.tensor(r['ids']);att[i,:len(r['ids'])]=1
        marks[i,:len(r['markers'])]=torch.tensor(r['markers']);mask[i,:len(r['markers'])]=True
    return {'input_ids':ids.to(device),'attention_mask':att.to(device),'marker_pos':marks.to(device),'marker_mask':mask.to(device),
            'qtype':torch.tensor([r['qtype'] for r in rows],dtype=torch.long,device=device)}


def losses(logits,rows,score_spec):
    labels=torch.tensor([r['presented_label'] for r in rows],device=logits.device)
    if rows[0]['primitive']!='score':return F.cross_entropy(logits,labels,reduction='none')
    targets=torch.tensor([r['target_probs'] for r in rows],device=logits.device)
    ce=-(targets*F.log_softmax(logits,dim=-1)).sum(-1)
    anchors=torch.tensor(score_spec['anchors'],device=logits.device)
    pred=logits.softmax(-1)@anchors
    true=torch.tensor([r['value'] for r in rows],device=logits.device)
    return .5*ce+.5*((pred-true)/score_spec['train_std']).square()


def summarize(records,score_spec):
    output={}
    for task in TASKS:
        rr=[r for r in records if r['task']==task]
        if not rr:continue
        y=np.array([r['label'] for r in rr]);p=np.array([r['probs'] for r in rr]);pred=p.argmax(-1)
        out={'rows':len(rr),'accuracy':float(accuracy_score(y,pred)),
             'macro_f1':float(f1_score(y,pred,labels=list(range(p.shape[1])),average='macro',zero_division=0)),
             'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-30,1)).mean()),
             'confusion':confusion_matrix(y,pred,labels=list(range(p.shape[1]))).tolist(),
             'output_valid_fraction':float(np.mean([r['output_valid'] for r in rr]))}
        if task=='fluorescence':
            values=np.array([r['value'] for r in rr]);estimate=np.array([r['score_prediction'] for r in rr])
            rho=spearmanr(values,estimate).statistic
            out.update(rmse=float(np.sqrt(mean_squared_error(values,estimate))),mae=float(mean_absolute_error(values,estimate)),
                spearman=float(rho) if np.isfinite(rho) else None,normalized_rmse=float(np.sqrt(mean_squared_error(values,estimate))/score_spec['train_std']),
                prediction_std=float(estimate.std()),
                training_mean_baseline_rmse=float(np.sqrt(np.mean((values-score_spec['train_mean'])**2))),
                training_median_baseline_mae=float(np.mean(np.abs(values-score_spec['train_median']))),
                outside_anchor_range_fraction=float(np.mean((values<score_spec['train_min'])|(values>score_spec['train_max']))),
                discrete_metrics_note='Accuracy/F1/NLL use nearest-anchor class only; continuous RMSE is primary.')
        else:
            mat=np.array(out['confusion']);out['per_class_recall']=(np.diag(mat)/np.maximum(1,mat.sum(1))).tolist()
        output[task]=out
    return output


def prediction_records(rows,logits,score_spec):
    records=[]
    for row,logit in zip(rows,logits):
        p=logit[:len(row['choices'])].float().softmax(-1).cpu().tolist()
        canonical=[0.]*len(p)
        for j,k in enumerate(row['order']):canonical[k]=p[j]
        answer=int(np.argmax(canonical));primitive=row['primitive']
        output={'type':primitive,'probabilities':canonical}
        if primitive=='noul':output['answer']=bool(answer)
        elif primitive=='choice':output.update(index=answer,answer=row['choices'][answer])
        else:output.update(level=answer,value=float(np.dot(canonical,score_spec['anchors'])),units='log fluorescence')
        record={'id':row['id'],'group_id':row['group_id'],'task':row['task'],'label':row['label'],'probs':canonical,'prediction':answer,'output':output,
                'output_valid':bool(np.isfinite(canonical).all() and abs(sum(canonical)-1)<1e-5 and 0<=answer<len(row['choices']))}
        if primitive=='score':record.update(value=row['value'],score_prediction=output['value'])
        records.append(record)
    return records


@torch.no_grad()
def evaluate(model,rows,pad,micro,spec,reverse=False):
    model.eval();records=[]
    for task in TASKS:
        pool=[r for r in rows if r['task']==task]
        for start in range(0,len(pool),micro):
            rr=[render(r,reverse=reverse) for r in pool[start:start+micro]]
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**pack(rr,pad))
            assert torch.isfinite(logits).all()
            records.extend(prediction_records(rr,logits,spec))
    return records


def schedule(rows):
    by={t:[r for r in rows if r['task']==t] for t in TASKS}
    assert len({len(v) for v in by.values()})==1
    assert all(len(v)%BATCH==0 for v in by.values())
    task_batches=len(by[TASKS[0]])//BATCH
    rng=random.Random(SEED+41);step=0
    for epoch in range(1,EPOCHS+1):
        for task in TASKS:rng.shuffle(by[task])
        for index in range(task_batches):
            order=TASKS.copy();rng.shuffle(order)
            for task in order:
                step+=1;yield step,epoch,task,by[task][index*BATCH:(index+1)*BATCH]


def lr_factor(step,total):
    warmup=max(1,int(.05*total))
    if step<=warmup:return step/warmup
    return .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))


def make_optimizer(model):
    groups=defaultdict(list)
    for name,p in model.named_parameters():
        rate=2e-5 if name.startswith('encoder.') else 1e-4
        decay=0. if p.ndim<2 or 'norm' in name.lower() or 'embeddings' in name or 'type_emb' in name else .01
        groups[(rate,decay)].append(p)
    return torch.optim.AdamW([{'params':params,'lr':rate,'base_lr':rate,'weight_decay':decay} for (rate,decay),params in groups.items()])


def train_step(model,opt,rows,pad,micro,spec):
    model.train();opt.zero_grad(set_to_none=True);loss_total=0.
    for start in range(0,len(rows),micro):
        rr=rows[start:start+micro]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**pack(rr,pad));loss=losses(logits,rr,spec).sum()/len(rows)
        loss.backward();loss_total+=float(loss.detach())
    encoder_norm=float(model.encoder.layers[0].mlp.Wi.weight.grad.norm())
    scorer_norm=float(model.scorer[-1].weight.grad.norm())
    gradient=float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
    return {'loss':loss_total,'gradient_norm':gradient,'encoder_gradient_norm':encoder_norm,'scorer_gradient_norm':scorer_norm}


@torch.no_grad()
def benchmark(model,rows,pad,micro):
    """Sequential single-request and batched full-panel runtime, including render/pack."""
    model.eval();result={}
    for task in TASKS:
        pool=sorted([r for r in rows if r['task']==task],key=lambda r:digest('bench:'+r['id']))[:64]
        for _ in range(3):
            with torch.autocast('cuda',dtype=torch.bfloat16):model(**pack([render(pool[0])],pad))
        elapsed=[]
        for row in pool:
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**pack([render(row)],pad))
            out.float().softmax(-1).cpu().tolist();torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
        torch.cuda.synchronize();started=time.perf_counter()
        for start in range(0,len(pool),micro):
            rr=[render(r) for r in pool[start:start+micro]]
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**pack(rr,pad))
            out.float().softmax(-1).cpu().tolist()
        torch.cuda.synchronize();total=time.perf_counter()-started
        result[task]={'requests':len(pool),'batch_size':micro,'single_request_p50_ms':float(np.median(elapsed)*1000),
          'single_request_p95_ms':float(np.quantile(elapsed,.95)*1000),'batched_requests_per_second':len(pool)/total,
          'mean_input_tokens':float(np.mean([r['length'] for r in pool])),'all_candidates_scored_in_one_encoder_call':True,
          'scope':'Cached biological/text tokenization; includes panel assembly, CPU packing, H2D, model, softmax, D2H; excludes file/network latency.'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-dir','cpt-root','data-dir','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',choices=['cpt','no_cpt'],required=True);p.add_argument('--task',choices=['joint']+TASKS,default='joint')
    p.add_argument('--micro',type=int,default=8);p.add_argument('--eval-micro',type=int,default=16)
    p.add_argument('--smoke',action='store_true');p.add_argument('--retain-model',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(8);bio.seed(SEED)
    manifest=json.loads((a.data_dir/'manifest.json').read_text());assert manifest['status']=='complete'
    rows={}
    for split in ['train'] if a.smoke else ['train','dev']:
        path=a.data_dir/manifest['outputs'][split]['file'];assert sha(path)==manifest['outputs'][split]['sha256'];rows[split]=read_lines(path)
    spec=manifest['score_spec'];pad=0
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.cpt_root/'data/representation/base_tokenizer');pad=tok.pad_token_id
    model=build(a.model_dir,a.cpt_root,a.arm)
    nparams=sum(p.numel() for p in model.parameters())
    shared_sha=tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')})
    orders={t:hashlib.sha256() for t in TASKS};panel_hash=hashlib.sha256();total=0
    for step,epoch,task,rr in schedule(rows['train']):
        for r in rr:orders[task].update((r['id']+'\n').encode())
        if a.task=='joint' or task==a.task:
            total+=1
            for r in rr:panel_hash.update(json.dumps(render(r,epoch,True)['order']).encode())
    config={'arm':a.arm,'task':a.task,'seed':SEED,'epochs':EPOCHS,'effective_batch':BATCH,'micro_batch':a.micro,'eval_micro_batch':a.eval_micro,
       'updates':total,'encoder_lr':2e-5,'typed_head_scorer_lr':1e-4,'schedule':'5% warmup; cosine to 10%; single tasks replay joint global LR positions',
       'initial_shared_head_sha256':shared_sha,'batch_order_sha256_by_task':{k:v.hexdigest() for k,v in orders.items()},'training_panel_order_sha256':panel_hash.hexdigest(),
       'data_manifest_sha256':sha(a.data_dir/'manifest.json'),'initial_model_sha256':sha(a.model_dir/'model.safetensors'),
       'cpt_checkpoint_sha256':manifest['cpt_checkpoint_sha256'] if a.arm=='cpt' else None,'trainable_parameters':nparams,
       'score_spec':spec,'head':'Original shared Laya 2-layer typed transformer and shared marker scorer; no task-specific heads; no temperature calibration.',
       'smoke_only':a.smoke,'test_inference':False,'model_retention':'Joint models retained; anchors may be removed after exact reload verification.'}
    if a.arm=='cpt':assert sha(a.cpt_root/'round/cpt/model.safetensors')==manifest['cpt_checkpoint_sha256']
    write_json(a.output/'run_config.json',config)
    write_json(a.output/'status.json',{'status':'running','step':0,'updates':total,'created_utc':datetime.now(timezone.utc).isoformat()})
    opt=make_optimizer(model);bio.seed(SEED+501)
    task_rows=lambda rr:[r for r in rr if a.task=='joint' or r['task']==a.task]
    train_eval=[]
    for task in TASKS:
        train_eval.extend(sorted([r for r in task_rows(rows['train']) if r['task']==task],key=lambda r:digest('train-eval:'+r['id']))[:1024])
    def measure(epoch,split,rr):
        records=evaluate(model,rr,pad,a.eval_micro,spec)
        write_lines(a.output/f'{split}_epoch{epoch}_predictions.jsonl',records)
        metric={'epoch':epoch,'split':split,'metrics':summarize(records,spec)};append(a.output/'learning_curve.jsonl',metric)
        print(json.dumps(metric),flush=True);return records
    if not a.smoke:
        measure(0,'dev',task_rows(rows['dev']));measure(0,'train',train_eval)
    executed=0;times=defaultdict(list);exposures=Counter();global_total=3*total if a.task!='joint' else total
    last_epoch=0
    for step,epoch,task,rr in schedule(rows['train']):
        if a.task!='joint' and task!=a.task:continue
        if a.smoke and task in times:continue
        for g in opt.param_groups:g['lr']=g['base_lr']*lr_factor(step,global_total)
        rendered=[render(r,epoch,True) for r in rr];torch.cuda.synchronize();t0=time.monotonic()
        result=train_step(model,opt,rendered,pad,a.micro,spec);torch.cuda.synchronize()
        elapsed=time.monotonic()-t0;times[task].append(elapsed);exposures.update(r['id'] for r in rr);executed+=1
        point={'step':executed,'joint_schedule_step':step,'epoch':epoch,'task':task,**result,'seconds':elapsed,'encoder_lr':next(g['lr'] for g in opt.param_groups if g['base_lr']==2e-5)}
        append(a.output/'training_trace.jsonl',point)
        if executed%16==0 or executed==1:
            write_json(a.output/'status.json',{'status':'running','step':executed,'updates':total,'epoch':epoch,'task':task});print(json.dumps(point),flush=True)
        if a.smoke and len(times)==3:break
        # Final task update of an epoch; single-task filters still reach it.
        per_epoch=total//EPOCHS
        if not a.smoke and executed%per_epoch==0:
            measure(epoch,'dev',task_rows(rows['dev']));last_epoch=epoch
    if not a.smoke:
        assert executed==total and last_epoch==EPOCHS
        assert set(exposures.values())=={EPOCHS}
        final_train=measure(EPOCHS,'train',train_eval)
        reverse_rows=[r for r in task_rows(rows['dev']) if r['primitive']=='choice']
        if reverse_rows:
            reversed_predictions=evaluate(model,reverse_rows,pad,a.eval_micro,spec,reverse=True)
            write_lines(a.output/'dev_reversed_choice_predictions.jsonl',reversed_predictions)
            write_json(a.output/'reversed_choice_metrics.json',summarize(reversed_predictions,spec))
        if a.task=='joint':write_json(a.output/'inference_benchmark.json',benchmark(model,rows['dev'],pad,a.eval_micro))
    # Save/reload compares a fixed train-only panel covering every trained task.
    probe=[]
    for task in TASKS:probe.extend([r for r in task_rows(rows['train']) if r['task']==task][:8])
    before=evaluate(model,probe,pad,a.eval_micro,spec)
    model_path=a.output/'model.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(model_path))
    model_sha=sha(model_path);peak=torch.cuda.max_memory_allocated()
    if not a.smoke and a.task=='joint':
        # A continuation checkpoint is kept for the primary CPT joint run only.
        if a.arm=='cpt':
            torch.save({'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
              'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'completed_epochs':EPOCHS,'completed_updates':executed,
              'schedule_seed':SEED+41,'continuation_note':'Completed fixed schedule; a future extension must declare its new LR schedule.'},a.output/'resume.pt')
    del opt,model;gc.collect();torch.cuda.empty_cache()
    model=build(a.model_dir,a.cpt_root,a.arm);model.load_state_dict(load_file(str(model_path)),strict=True)
    after=evaluate(model,probe,pad,a.eval_micro,spec);assert before==after,'Checkpoint predictions differ'
    if a.smoke or not a.retain_model:model_path.unlink()
    final={'status':'complete','updates':executed,'checkpoint_reload_exact':True,'checkpoint_sha256':model_sha,
       'model_retained':not a.smoke and a.retain_model,'peak_gpu_bytes':peak,'elapsed_seconds':time.monotonic()-started,
       'seconds_per_update_by_task':{t:float(np.mean(v)) for t,v in times.items()},'presentations_by_task':{t:sum(exposures[r['id']] for r in rows['train'] if r['task']==t) for t in TASKS},
       'per_entity_exposure_histogram':dict(Counter(exposures.values())),'completed_utc':datetime.now(timezone.utc).isoformat(),'test_inference':False}
    write_json(a.output/'status.json',final);print(json.dumps(final),flush=True)

if __name__=='__main__':main()
