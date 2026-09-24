#!/usr/bin/env python3
"""Audited biological MLM adaptation and paired conventional classification."""
from __future__ import annotations
import argparse
from collections import Counter
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
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM
from laya_biocpt_data import read_lines, write_lines, write_json, sha

SEED = 20260925


def seed(value=SEED):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def expand(model, entries, size):
    old = model.get_input_embeddings().weight.detach().clone()
    model.resize_token_embeddings(size, mean_resizing=False)
    with torch.no_grad():
        for e in entries:
            model.get_input_embeddings().weight[e['expanded_id']].copy_(old[e['base_ids']].mean(0))
    assert torch.equal(old, model.get_input_embeddings().weight[:len(old)])
    model.tie_weights()
    assert model.decoder.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    return old


def build(model_dir, data_dir, checkpoint=None):
    seed()
    config = ModernBertConfig.from_pretrained(model_dir / 'encoder')
    config._attn_implementation = 'sdpa'
    config.sparse_prediction = True
    config.reference_compile = False
    model = ModernBertForMaskedLM(config)
    original = load_file(str(model_dir / 'model.safetensors'))
    model.model.load_state_dict({k.removeprefix('encoder.'): v for k,v in original.items() if k.startswith('encoder.')}, strict=True)
    del original
    entries = json.loads((data_dir / 'representation/new_tokens.json').read_text())
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    expand(model, entries, manifest['new_vocab_size'])
    if checkpoint:
        missing, extra = model.load_state_dict(load_file(str(checkpoint)), strict=False)
        assert missing == ['decoder.weight'] and not extra, (missing, extra)
        model.tie_weights()
    model.to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, entries, manifest


def pack(rows, pad, device='cuda'):
    width = max(len(r['ids']) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for i,r in enumerate(rows):
        ids[i,:len(r['ids'])] = torch.tensor(r['ids'])
        attention[i,:len(r['ids'])] = 1
    return ids.to(device), attention.to(device)


class Masker:
    def __init__(self, tokenizer, entries):
        self.pad = tokenizer.pad_token_id
        self.mask = tokenizer.mask_token_id
        self.special = set(tokenizer.all_special_ids)
        self.old_size = len(tokenizer)
        self.pool = {m: [e['expanded_id'] for e in entries if e['modality'] == m] for m in ['dna', 'protein']}
        self.pool['text'] = [i for i in range(len(tokenizer)) if i not in self.special]

    def __call__(self, rows, rng, device='cuda', fixed=False):
        ids, attention = pack(rows, self.pad, 'cpu')
        labels = torch.full_like(ids, -100)
        inputs, targets = Counter(), Counter()
        for j,r in enumerate(rows):
            local = random.Random(int(hashlib.sha256(('fixed-mask:' + r['id']).encode()).hexdigest()[:16], 16)) if fixed else rng
            eligible = [i for i in range(r['body_start'], len(r['ids'])-1) if r['ids'][i] not in self.special]
            assert eligible
            chosen = [i for i in eligible if local.random() < .15]
            if not chosen:
                chosen = [local.choice(eligible)]
            inputs.update(i for i in r['ids'] if i >= self.old_size)
            for i in chosen:
                token = int(ids[j,i])
                labels[j,i] = token
                if token >= self.old_size:
                    targets[token] += 1
                action = local.random()
                if action < .8:
                    ids[j,i] = self.mask
                elif action < .9:
                    ids[j,i] = local.choice(self.pool[r['modality']])
        return ids.to(device), attention.to(device), labels.to(device), inputs, targets


class Stream:
    def __init__(self, rows):
        self.rows = {m: [r for r in rows if r['modality'] == m] for m in ['dna','protein','text']}
        self.rng = random.Random(SEED + 31)
        self.order, self.position = {}, {}
        for m in self.rows:
            self.order[m] = list(range(len(self.rows[m])))
            self.rng.shuffle(self.order[m])
            self.position[m] = 0

    def batch(self):
        out = []
        for m,n in [('dna',14),('protein',14),('text',4)]:
            for _ in range(n):
                if self.position[m] == len(self.order[m]):
                    self.rng.shuffle(self.order[m])
                    self.position[m] = 0
                out.append(self.rows[m][self.order[m][self.position[m]]])
                self.position[m] += 1
        self.rng.shuffle(out)
        return out

    def state(self):
        return {'order': self.order, 'position': self.position, 'rng': self.rng.getstate()}


def warmup_mode(model, old_size):
    model.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    if model.decoder.bias is not None:
        model.decoder.bias.requires_grad_(True)
    embedding = model.get_input_embeddings().weight
    embedding.requires_grad_(True)
    def mask_old(grad):
        grad[:old_size].zero_()
        return grad
    return embedding.register_hook(mask_old)


def optimizer(model, phase):
    groups = {}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        embed = name.endswith('embeddings.tok_embeddings.weight')
        head = name.startswith(('head.', 'decoder.', 'classifier.'))
        lr = (1e-3 if embed else 1e-4) if phase == 'warmup' else (1e-4 if head else 2e-5)
        decay = 0. if embed or p.ndim < 2 or 'norm' in name else .01
        key = lr, decay
        groups.setdefault(key, []).append(p)
    return torch.optim.AdamW([{'params': ps, 'lr': lr, 'peak_lr': lr, 'weight_decay': wd} for (lr,wd),ps in groups.items()])


def schedule(opt, step, total, warmup, constant=False):
    scale = step / warmup if step <= warmup else (1. if constant else .1 + .9 * .5 * (1 + math.cos(math.pi * (step-warmup) / max(1,total-warmup))))
    for g in opt.param_groups:
        g['lr'] = g['peak_lr'] * scale


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def save_mlm(model, path):
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if k != 'decoder.weight'}, str(path))


@torch.no_grad()
def eval_mlm(model, rows, masker, micro):
    model.eval()
    result = {}
    for modality in ['dna','protein','text']:
        selected = [r for r in rows if r['modality'] == modality]
        loss, correct, count = 0.,0,0
        for start in range(0,len(selected),micro):
            x,att,y,_,_ = masker(selected[start:start+micro], None, fixed=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=att, labels=y).logits
            target = y[y != -100]
            loss += float(F.cross_entropy(logits.float(), target, reduction='sum'))
            correct += int((logits.argmax(-1) == target).sum())
            count += len(target)
        result[modality] = {'nll':loss/count, 'masked_accuracy':correct/count, 'targets':count}
    return result


def mlm_update(model, batch, masker, rng, opt, micro):
    model.train()
    opt.zero_grad(set_to_none=True)
    x,att,y,inputs,targets = masker(batch,rng)
    denominator = int((y != -100).sum())
    total = 0.
    for start in range(0,len(batch),micro):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=x[start:start+micro], attention_mask=att[start:start+micro], labels=y[start:start+micro]).logits
            target = y[start:start+micro]
            loss = F.cross_entropy(logits.float(),target[target != -100],reduction='sum') / denominator
        loss.backward()
        total += float(loss.detach())
    embed = model.get_input_embeddings().weight
    new_norm = float(embed.grad[masker.old_size:].norm())
    head_norm = float(model.head.dense.weight.grad.norm())
    encoder_norm = float(model.model.layers[0].mlp.Wi.weight.grad.norm()) if model.model.layers[0].mlp.Wi.weight.grad is not None else 0.
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    opt.step()
    return {'loss':total,'masked_targets':denominator,'gradient_norm':norm,'new_embedding_gradient_norm':new_norm,'mlm_head_gradient_norm':head_norm,'encoder_gradient_norm':encoder_norm},inputs,targets


def run_cpt(a, smoke=False):
    model,entries,manifest = build(a.model_dir,a.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(a.data_dir / 'representation/base_tokenizer')
    masker = Masker(tokenizer,entries)
    stream = Stream(read_lines(a.data_dir / 'cpt_train.jsonl'))
    validation = read_lines(a.data_dir / 'cpt_validation.jsonl') if not smoke else []
    rng = random.Random(SEED+99)
    old_size = manifest['old_vocab_size']
    embedding = model.get_input_embeddings().weight
    old = embedding[:old_size].detach().clone()
    initial_new = embedding[old_size:].detach().cpu().clone()
    counts, targets = Counter(),Counter()
    warmup_steps,full_steps = (2,2) if smoke else (128,1024)
    config = {'warmup_steps':warmup_steps,'full_steps':full_steps,'micro_batch':a.micro,'effective_batch':32,'mixture':{'dna':14,'protein':14,'text':4},'seed':SEED,'data_manifest_sha256':sha(a.data_dir/'manifest.json'),'base_model_sha256':sha(a.model_dir/'model.safetensors'),'test_access':False,'fresh_MLM_head':True,'optimizer_reset_after_warmup':True,'smoke_only':smoke}
    write_json(a.output/'run_config.json',config)
    def evaluate(step,phase):
        if smoke:
            return
        result = {'step':step,'phase':phase,'validation':eval_mlm(model,validation,masker,a.micro)}
        append(a.output/'learning_curve.jsonl',result)
        print(json.dumps(result),flush=True)
    evaluate(0,'initial')
    hook = warmup_mode(model,old_size)
    step = 0
    timings = {}
    for phase,total in [('warmup',warmup_steps),('full',full_steps)]:
        if phase == 'full':
            assert torch.equal(old, embedding[:old_size])
            hook.remove()
            model.requires_grad_(True)
        opt = optimizer(model,phase)
        t0 = time.monotonic()
        for local in range(1,total+1):
            step += 1
            schedule(opt,local,total,min(total,8 if phase == 'warmup' else 32),constant=phase=='warmup')
            start = time.monotonic()
            stats,seen,masked = mlm_update(model,stream.batch(),masker,rng,opt,a.micro)
            assert stats['new_embedding_gradient_norm'] > 0 and stats['mlm_head_gradient_norm'] > 0
            if phase == 'full':
                assert stats['encoder_gradient_norm'] > 0
            if phase == 'warmup':
                assert torch.equal(old,embedding[:old_size])
            counts.update(seen)
            targets.update(masked)
            stats.update(step=step,phase=phase,phase_step=local,seconds=time.monotonic()-start)
            append(a.output/'training_trace.jsonl',stats)
            if local == 1 or local % 16 == 0 or local == total:
                write_json(a.output/'status.json',{'status':'running',**stats})
                print(json.dumps(stats),flush=True)
            if local == total or (phase=='full' and local%256==0):
                evaluate(step,phase)
        timings[phase] = (time.monotonic()-t0)/total
        write_json(a.output/(phase+'_embedding_check.json'),{'old_rows_equal':torch.equal(old,embedding[:old_size]),'old_row_max_change':float((embedding[:old_size].detach()-old).abs().max()),'new_row_mean_l2_change':float((embedding[old_size:].detach().cpu()-initial_new).norm(dim=1).mean()),'seconds_per_step_including_evaluation':timings[phase]})
    model.eval()
    probe = masker(stream.rows['dna'][:2],None,fixed=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected = model(input_ids=probe[0],attention_mask=probe[1],labels=probe[2]).logits.detach().clone()
    path = a.output/'model.safetensors'
    save_mlm(model,path)
    missing,extra=model.load_state_dict(load_file(str(path)),strict=False)
    assert missing == ['decoder.weight'] and not extra
    model.tie_weights()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        actual = model(input_ids=probe[0],attention_mask=probe[1],labels=probe[2]).logits
    assert torch.equal(expected,actual)
    if not smoke:
        torch.save({'optimizer':opt.state_dict(),'phase':'full','step':step,'stream':stream.state(),'mask_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'input_counts':dict(counts),'target_counts':dict(targets),'schedule_full_steps':full_steps},a.output/'resume.pt')
    deltas=(embedding[old_size:].detach().cpu()-initial_new).norm(dim=1).tolist()
    write_json(a.output/'token_training_audit.json',[{**e,'observed_input_count':counts[e['expanded_id']],'masked_target_count':targets[e['expanded_id']],'weight_delta_l2':deltas[e['expanded_id']-old_size]} for e in entries])
    result={'status':'complete','updates':step,'timing':timings,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'checkpoint_reload_exact':True,'checkpoint_sha256':sha(path),'natural_input_token_coverage':sum(counts[e['expanded_id']]>0 for e in entries),'masked_target_token_coverage':sum(targets[e['expanded_id']]>0 for e in entries),'added_token_count':len(entries)}
    write_json(a.output/'status.json',result)
    print(json.dumps(result),flush=True)
    if smoke:
        path.unlink()
        # Exercise the conventional readout on training records only.
        del opt, old, expected, actual, probe
        classifier = Classifier(model.model).to('cuda')
        sample = read_lines(a.data_dir/'sft_train_4096.jsonl')[:64]
        sft_opt = optimizer(classifier,'sft')
        classifier.train()
        sft_opt.zero_grad(set_to_none=True)
        started = time.monotonic()
        for start in range(0,len(sample),a.micro):
            rr = sample[start:start+a.micro]
            x,att = pack(rr,tokenizer.pad_token_id)
            labels = torch.tensor([r['label'] for r in rr],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = F.cross_entropy(classifier(x,att).float(),labels,reduction='sum')/len(sample)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(),1.,error_if_nonfinite=True)
        sft_opt.step()
        result['sft_seconds_per_update'] = time.monotonic()-started
        result['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        write_json(a.output/'status.json',result)


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED+7)
            self.classifier = nn.Sequential(nn.LayerNorm(encoder.config.hidden_size),nn.Dropout(.1),nn.Linear(encoder.config.hidden_size,2))

    def forward(self,x,attention):
        hidden=self.encoder(input_ids=x,attention_mask=attention).last_hidden_state
        pooled=(hidden*attention.unsqueeze(-1)).sum(1)/attention.sum(1,keepdim=True)
        return self.classifier(pooled)


def metrics(records):
    matrix=np.zeros((2,2),dtype=int)
    nll=0.
    for r in records:
        matrix[r['label'],r['prediction']]+=1
        nll-=math.log(max(1e-30,r['probs'][r['label']]))
    f1=[]
    recalls=[]
    for i in range(2):
        f1.append(2*matrix[i,i]/max(1,matrix[i,:].sum()+matrix[:,i].sum()))
        recalls.append(matrix[i,i]/max(1,matrix[i,:].sum()))
    return {'accuracy':float(np.trace(matrix)/matrix.sum()),'macro_f1':float(np.mean(f1)),'per_class_recall':recalls,'confusion':matrix.tolist(),'nll':nll/len(records),'rows':len(records)}


@torch.no_grad()
def eval_sft(model,rows,pad,micro):
    model.eval()
    records=[]
    for start in range(0,len(rows),micro):
        rr=rows[start:start+micro]
        x,att=pack(rr,pad)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            probs=model(x,att).float().softmax(-1).cpu().tolist()
        records.extend({'id':r['id'],'group_id':r['group_id'],'label':r['label'],'probs':p,'prediction':int(np.argmax(p))} for r,p in zip(rr,probs))
    return records


def run_sft(a):
    mlm,_,manifest=build(a.model_dir,a.data_dir,a.checkpoint)
    model=Classifier(mlm.model).to('cuda')
    del mlm
    rows=read_lines(a.data_dir / ('sft_train_'+a.samples+'.jsonl'))
    dev=read_lines(a.data_dir / 'sft_dev.jsonl')
    pad=model.encoder.config.pad_token_id
    order_rng=random.Random(SEED+201)
    batches=[]
    for epoch in range(3):
        order=list(range(len(rows)))
        order_rng.shuffle(order)
        batches.append([order[i:i+64] for i in range(0,len(order),64)])
    total=sum(map(len,batches))
    digest=hashlib.sha256(json.dumps(batches).encode()).hexdigest()
    head_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.classifier.parameters())).hexdigest()
    config={'arm':'CPT' if a.checkpoint else 'no_CPT','samples':len(rows),'epochs':3,'updates':total,'effective_batch':64,'micro_batch':a.micro,'batch_order_sha256':digest,'initial_classifier_sha256':head_hash,'seed':SEED,'data_manifest_sha256':sha(a.data_dir/'manifest.json'),'encoder_initial_checkpoint_sha256':sha(a.checkpoint if a.checkpoint else a.model_dir/'model.safetensors'),'test_access':False,'final_step_primary':True,'precision':'FP32 parameters, BF16 autocast'}
    write_json(a.output/'run_config.json',config)
    opt=optimizer(model,'sft')
    seed(SEED+501)
    step=0
    def evaluate(epoch,training=False):
        for split,rr in [('dev',dev)]+([('train',rows)] if training else []):
            records=eval_sft(model,rr,pad,a.micro)
            write_lines(a.output/f'{split}_epoch{epoch}_predictions.jsonl',records)
            result={'epoch':epoch,'step':step,'split':split,**metrics(records)}
            append(a.output/'learning_curve.jsonl',result)
            print(json.dumps(result),flush=True)
    evaluate(0,True)
    for epoch,epoch_batches in enumerate(batches,1):
        for indices in epoch_batches:
            step+=1
            start=time.monotonic()
            schedule(opt,step,total,max(1,math.ceil(.05*total)))
            model.train()
            opt.zero_grad(set_to_none=True)
            rr=[rows[i] for i in indices]
            loss_total=0.
            for pos in range(0,len(rr),a.micro):
                sub=rr[pos:pos+a.micro]
                x,att=pack(sub,pad)
                target=torch.tensor([r['label'] for r in sub],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=F.cross_entropy(model(x,att).float(),target,reduction='sum')/len(rr)
                loss.backward()
                loss_total+=float(loss.detach())
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            opt.step()
            result={'step':step,'epoch':epoch,'loss':loss_total,'gradient_norm':norm,'seconds':time.monotonic()-start}
            append(a.output/'training_trace.jsonl',result)
            if step==1 or step%16==0:
                write_json(a.output/'status.json',{'status':'running',**result})
                print(json.dumps(result),flush=True)
        evaluate(epoch,epoch==3)
    model.eval()
    x,att=pack(dev[:2],pad)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=model(x,att).clone()
    path=a.output/'model.safetensors'
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path))
    model.load_state_dict(load_file(str(path)),strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        actual=model(x,att)
    assert torch.equal(expected,actual)
    result={'status':'complete','updates':step,'checkpoint_reload_exact':True,'checkpoint_sha256':sha(path),'checkpoint_retained':a.samples=='full','peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    write_json(a.output/'status.json',result)
    if a.samples!='full':
        path.unlink()
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['smoke','cpt','sft'])
    for arg in ['model-dir','data-dir','output']:
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--samples',choices=['4096','full'],default='4096')
    parser.add_argument('--micro',type=int,default=8)
    a=parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    seed()
    try:
        if a.mode=='sft':
            run_sft(a)
        else:
            run_cpt(a,a.mode=='smoke')
    except Exception as error:
        write_json(a.output/'status.json',{'status':'failed','error':repr(error)})
        raise


if __name__=='__main__':
    main()
