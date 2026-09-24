#!/usr/bin/env python3
"""Train/evaluate a task-balanced typed Laya engineering pilot, train/dev only.

Candidate scoring uses the original three Laya type embeddings. Multilabel
examples expand into marginal Noul propositions with per-entity normalization.
The shared-head control uses one sigmoid panel and a native scalar Score head.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
import laya_formal_experiment as core
import laya_multitask_data as data

TYPE_IDS={'choice':0,'score':1,'noul':2,'multilabel':2}


def make_items(rows, tokenizer, builder, kind, seed=0, training=False, max_length=1024):
    entities=[]
    for row in rows:
        if kind=='candidate':
            items=[]
            for item in data.questions(row):
                order=list(range(len(item['choices'])))
                if training and item['primitive']=='choice':
                    random.Random(int(data.sha(f'{seed}:{item["id"]}')[:16],16)).shuffle(order)
                ids,markers=builder(tokenizer,item['state'],data.rubric(item),max_len=1000000,head_max_len=256,option_order=order)
                if len(ids)>max_length:raise ValueError('Full input exceeds limit: '+item['id'])
                items.append({**item,'ids':ids,'markers':markers,'qtype':TYPE_IDS[item['primitive']],
                              'label':order.index(item['label']),'order':order})
        else:
            question='Predict all annotated cellular locations of this eukaryotic protein.' if row['primitive']=='multilabel' else row['question']
            ids=[tokenizer.cls_token_id]+tokenizer(question+'\n'+data.state(row),add_special_tokens=False)['input_ids']+[tokenizer.sep_token_id]
            if len(ids)>max_length:raise ValueError('Fixed-head full input exceeds limit')
            items=[{**row,'ids':ids,'qtype':TYPE_IDS[row['primitive']],'entity_id':row['id'],'weight':1.0}]
        entities.append({'id':row['id'],'task':row['task'],'items':items})
    return entities


def collate(items,pad_id,device):
    n=len(items); length=max(len(x['ids']) for x in items)
    ids=torch.full((n,length),pad_id,dtype=torch.long)
    attention=torch.zeros_like(ids)
    for i,item in enumerate(items):
        ids[i,:len(item['ids'])]=torch.tensor(item['ids']);attention[i,:len(item['ids'])]=1
    return {'input_ids':ids.to(device),'attention_mask':attention.to(device),
            'qtype':torch.tensor([x['qtype'] for x in items],device=device,dtype=torch.long)}


def candidate_forward(model,batch,items,device):
    k=max(len(x['markers']) for x in items)
    pos=torch.zeros((len(items),k),dtype=torch.long,device=device)
    mask=torch.zeros((len(items),k),dtype=torch.bool,device=device)
    for i,item in enumerate(items):
        pos[i,:len(item['markers'])]=torch.tensor(item['markers'],device=device);mask[i,:len(item['markers'])]=True
    logits,_=model(**batch,marker_pos=pos,marker_mask=mask)
    return logits.masked_fill(~mask,-1e4)


class SharedHeads(nn.Module):
    def __init__(self,base,task_info):
        super().__init__();self.encoder=base.encoder;self.head=base.head;self.type_emb=base.type_emb
        self.readout=nn.Sequential(*list(base.scorer.children())[:-1])
        d=base.encoder.config.hidden_size
        self.classifiers=nn.ModuleDict({t:nn.Linear(d,s['n_outputs']) for t,s in task_info.items()})
        self.regressors=nn.ModuleDict({t:nn.Linear(d,1) for t,s in task_info.items() if s['primitive']=='score'})
    def forward(self,input_ids,attention_mask,qtype,task):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        h=h+self.type_emb(qtype)[:,None,:]
        if self.head is not None:
            for layer in self.head.layers:h=layer(h,src_key_padding_mask=~attention_mask.bool())
        h=self.readout(h[:,0]);scalar=self.regressors[task](h).squeeze(-1).float() if task in self.regressors else None
        return self.classifiers[task](h).float(),scalar


def build(kind,model_dir,laya_repo,task_info,device,checkpoint=None):
    _,build_model,_=core.import_laya(str(laya_repo))
    cfg=json.loads((model_dir/'rl_agent_config.json').read_text());cfg.update(gradient_checkpointing=True)
    base=core.load_model(model_dir,cfg,build_model,torch.device('cpu'),trainable=True)
    base.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=base if kind=='candidate' else SharedHeads(base,task_info)
    if checkpoint is not None:model.load_state_dict(load_file(str(checkpoint/'model.safetensors')),strict=True)
    return model.to(device),cfg


def loss_values(model,kind,items,tokenizer,device,task_info):
    batch=collate(items,tokenizer.pad_token_id,device);task=items[0]['task']
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if kind=='candidate':
            logits=candidate_forward(model,batch,items,device)
            loss=nn.functional.cross_entropy(logits,torch.tensor([i['label'] for i in items],device=device),reduction='none')
        else:
            logits,scalar=model(**batch,task=task);primitive=task_info[task]['primitive']
            if primitive=='multilabel':
                targets=torch.tensor([i['labels'] for i in items],device=device)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none').mean(-1)
            else:
                loss=nn.functional.cross_entropy(logits,torch.tensor([i['label'] for i in items],device=device),reduction='none')
                if primitive=='score':
                    spec=task_info[task];target=torch.tensor([(i['value']-spec['mean'])/spec['std'] for i in items],device=device)
                    loss=.5*loss+.5*(scalar-target).square()
    return loss*torch.tensor([i['weight'] for i in items],device=device)


def evaluate(model,kind,entities,tokenizer,device,task_info,batch_size):
    model.eval();records=[];started=time.monotonic()
    with torch.no_grad():
        for task in data.TASKS:
            items=[i for e in entities if e['task']==task for i in e['items']]
            for start in range(0,len(items),batch_size):
                chunk=items[start:start+batch_size];batch=collate(chunk,tokenizer.pad_token_id,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if kind=='candidate':logits=candidate_forward(model,batch,chunk,device);scalar=None
                    else:logits,scalar=model(**batch,task=task)
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite eval logits')
                for j,item in enumerate(chunk):
                    common={'id':item['id'],'entity_id':item['entity_id'],'task':task}
                    if kind=='shared_heads' and task_info[task]['primitive']=='multilabel':
                        probs=torch.sigmoid(logits[j]).cpu().tolist()
                        for li,prob in enumerate(probs):
                            records.append({**common,'id':item['id']+':'+str(li),'label_index':li,
                                            'label':int(item['labels'][li]),'probs':[1-prob,prob]})
                    else:
                        k=len(item['choices']);p=torch.softmax(logits[j,:k].float(),-1).cpu().tolist()
                        out={**common,'label':int(item['label']),'probs':p}
                        if 'label_index' in item:out['label_index']=item['label_index']
                        if 'value' in item:
                            out['value']=item['value'];out['score_prediction']=float(np.dot(p,item['anchors']))
                            if scalar is not None:
                                spec=task_info[task];out['scalar_prediction']=float(scalar[j].cpu())*spec['std']+spec['mean']
                        records.append(out)
    torch.cuda.synchronize();return records,time.monotonic()-started


def metrics(records,task_info):
    by_task=defaultdict(list)
    for r in records:by_task[r['task']].append(r)
    output={}
    for task,rows in by_task.items():
        spec=task_info[task];primitive=spec['primitive']
        if primitive=='multilabel':
            entities=defaultdict(dict)
            for r in rows:entities[r['entity_id']][r['label_index']]=r
            assert all(len(v)==len(data.LOCATIONS) for v in entities.values())
            y=np.array([[v[i]['label'] for i in range(len(data.LOCATIONS))] for v in entities.values()])
            p=np.array([[v[i]['probs'][1] for i in range(len(data.LOCATIONS))] for v in entities.values()])
            valid=[i for i in range(y.shape[1]) if 0<y[:,i].sum()<len(y)]
            output[task]={'n_entities':len(y),'n_labels':y.shape[1],
                          'micro_auprc':float(average_precision_score(y.ravel(),p.ravel())),
                          'macro_auprc_observed_labels':float(np.mean([average_precision_score(y[:,i],p[:,i]) for i in valid])) if valid else None,
                          'macro_auprc_label_count':len(valid),'positive_support':y.sum(0).tolist(),
                          'micro_f1_at_0_5':float(f1_score(y,p>=.5,average='micro',zero_division=0)),
                          'macro_f1_at_0_5':float(f1_score(y,p>=.5,average='macro',zero_division=0)),
                          'brier_mean_per_label':float(np.mean((y-p)**2))}
            continue
        y=np.array([r['label'] for r in rows]);p=np.array([r['probs'] for r in rows]);pred=p.argmax(1)
        out={'n_entities':len(rows),'accuracy':float(accuracy_score(y,pred)),
             'macro_f1':float(f1_score(y,pred,labels=list(range(spec['n_outputs'])),average='macro',zero_division=0)),
             'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())}
        if primitive=='noul':
            out.update(mcc=float(matthews_corrcoef(y,pred)),auroc=float(roc_auc_score(y,p[:,1])) if len(set(y))==2 else None,
                       auprc=float(average_precision_score(y,p[:,1])) if y.sum() else None)
        if primitive=='score':
            true=np.array([r['value'] for r in rows]); anchors=spec['anchors']
            for field in ['score_prediction','scalar_prediction']:
                if field not in rows[0]:continue
                value=np.array([r[field] for r in rows]);rho=spearmanr(true,value).statistic
                out[field]={'mae':float(mean_absolute_error(true,value)), 'rmse':float(np.sqrt(mean_squared_error(true,value))),
                            'spearman':float(rho) if np.isfinite(rho) else None}
            out['oracle_quantization_mae']=float(mean_absolute_error(true,np.array(anchors)[y]))
            out['outside_training_range_fraction']=float(np.mean((true<spec['train_min'])|(true>spec['train_max'])))
        output[task]=out
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['candidate','shared_heads'],required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--laya-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=100);p.add_argument('--effective-batch',type=int,default=32)
    p.add_argument('--micro-batch',type=int,default=4);p.add_argument('--eval-batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=20260924);p.add_argument('--lr',type=float,default=2e-5)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(8);core.set_seed(a.seed);device=torch.device('cuda')
    manifest=json.loads((a.data_dir/'manifest.json').read_text());rows={}
    for split in ['train','dev']:
        path=a.data_dir/manifest['outputs'][split]['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['outputs'][split]['sha256']
        rows[split]=[json.loads(line) for line in path.open()]
    task_info={}
    for task in data.TASKS:
        values=[r for r in rows['train'] if r['task']==task];r=values[0]
        info={'primitive':r['primitive'],'n_outputs':len(data.LOCATIONS) if r['primitive']=='multilabel' else len(r['choices'])}
        if r['primitive']=='score':
            arr=np.array([x['value'] for x in values]);info.update(manifest['score_spec'],mean=float(arr.mean()),std=max(1e-6,float(arr.std())))
        task_info[task]=info
    _,_,builder=core.import_laya(str(a.laya_repo));tok=AutoTokenizer.from_pretrained(a.model_dir/'tokenizer')
    entities={s:make_items(v,tok,builder,a.kind,a.seed,s=='train',manifest['max_length']) for s,v in rows.items()}
    a.output.mkdir(parents=True)
    config={'kind':a.kind,'seed':a.seed,'updates':a.updates,'effective_entity_batch':a.effective_batch,
            'micro_batch':a.micro_batch,'lr':a.lr,'task_info':task_info,'data_manifest_sha256':hashlib.sha256((a.data_dir/'manifest.json').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gpu':torch.cuda.get_device_name(),'gpu_total_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
            'test_access':False,'status':'engineering_pilot_not_confirmatory','initialization':'original_Laya_not_legacy_task_finetune',
            'score_control_loss':'half ordinal CE plus half standardized scalar MSE' if a.kind=='shared_heads' else None}
    (a.output/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    model,cfg=build(a.kind,a.model_dir,a.laya_repo,task_info,device)
    type_before=model.type_emb.weight.detach().cpu().clone()
    print(json.dumps({'event':'loaded','kind':a.kind,'task_info':task_info,'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}),flush=True)
    before,before_seconds=evaluate(model,a.kind,entities['dev'],tok,device,task_info,a.eval_batch)
    (a.output/'dev_before.json').write_text(json.dumps(metrics(before,task_info),indent=2)+'\n')
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    rng=random.Random(a.seed);by_task={t:[e for e in entities['train'] if e['task']==t] for t in data.TASKS}
    for pool in by_task.values():rng.shuffle(pool)
    cursors=Counter();consumed=Counter();propositions=Counter();trace=[];cycle=[];started=time.monotonic()
    torch.cuda.reset_peak_memory_stats();model.train()
    for step in range(1,a.updates+1):
        if not cycle:cycle=data.TASKS.copy();rng.shuffle(cycle)
        task=cycle.pop();pool=by_task[task];chunk=[]
        while len(chunk)<a.effective_batch:
            if cursors[task]>=len(pool):rng.shuffle(pool);cursors[task]=0
            chunk.append(pool[cursors[task]]);cursors[task]+=1
        items=[item for e in chunk for item in e['items']]
        optimizer.zero_grad(set_to_none=True);step_loss=0.
        warmup=max(1,math.ceil(.05*a.updates))
        factor=step/warmup if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,a.updates-warmup)))
        for group in optimizer.param_groups:group['lr']=a.lr*factor
        for offset in range(0,len(items),a.micro_batch):
            losses=loss_values(model,a.kind,items[offset:offset+a.micro_batch],tok,device,task_info)
            loss=losses.sum()/len(chunk)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss step {step}')
            loss.backward();step_loss+=float(loss.detach())
        norm=float(nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step()
        consumed[task]+=len(chunk);propositions[task]+=len(items)
        rec={'step':step,'task':task,'loss':step_loss,'gradient_norm':norm,'seconds':time.monotonic()-started}
        trace.append(rec)
        with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        if step==1 or step%5==0 or step==a.updates:print(json.dumps(rec),flush=True)
    torch.cuda.synchronize();training_seconds=time.monotonic()-started
    peak=torch.cuda.max_memory_allocated()/2**30
    after,after_seconds=evaluate(model,a.kind,entities['dev'],tok,device,task_info,a.eval_batch)
    type_delta=(model.type_emb.weight.detach().cpu()-type_before).norm(dim=1).tolist()
    checkpoint=a.output/'checkpoint';checkpoint.mkdir()
    save_file({k:v.detach().contiguous().cpu() for k,v in model.state_dict().items()},str(checkpoint/'model.safetensors'))
    model.encoder.config.save_pretrained(checkpoint/'encoder');tok.save_pretrained(checkpoint/'tokenizer')
    (checkpoint/'rl_agent_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (checkpoint/'task_info.json').write_text(json.dumps(task_info,indent=2)+'\n')
    del optimizer,model;gc.collect();torch.cuda.empty_cache()
    reloaded,_=build(a.kind,a.model_dir,a.laya_repo,task_info,device,checkpoint)
    reload_records,_=evaluate(reloaded,a.kind,entities['dev'],tok,device,task_info,a.eval_batch)
    max_error=max(float(np.max(np.abs(np.array(x['probs'])-np.array(y['probs'])))) for x,y in zip(after,reload_records))
    assert len(after)==len(reload_records) and all(x['id']==y['id'] for x,y in zip(after,reload_records)) and max_error<1e-6
    for tag,records in [('before',before),('after',after)]:
        (a.output/f'dev_{tag}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary={**config,'complete':True,'training_seconds':training_seconds,'peak_allocated_gib':peak,
             'entity_presentations':dict(consumed),'proposition_presentations':dict(propositions),
             'type_embedding_update_norms':dict(zip(['choice','score','noul'],type_delta)),
             'before':metrics(before,task_info),'after':metrics(after,task_info),'dev_eval_seconds':after_seconds,
             'reload_max_probability_error':max_error,'checkpoint_reload_passed':True,
             'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
             'limitations':manifest['limitations']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
