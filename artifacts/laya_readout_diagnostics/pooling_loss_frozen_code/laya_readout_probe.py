"""Training-subset feature and readout-gradient probes for GFP diagnostics."""
import numpy as np
import torch
import laya_multitask_experiment as exp


def feature_stats(values):
    x=values.float();centered=x-x.mean(0,keepdim=True)
    norms=x.norm(dim=-1);normalized=x/norms[:,None].clamp_min(1e-12)
    cosine=normalized@normalized.T;off=~torch.eye(len(x),dtype=torch.bool)
    sv=torch.linalg.svdvals(centered);energy=sv.square();p=energy/energy.sum().clamp_min(1e-30)
    return {'mean_feature_std':float(x.std(0,unbiased=False).mean()),
            'mean_vector_norm':float(norms.mean()),'mean_pairwise_cosine':float(cosine[off].mean()),
            'centered_frobenius_norm':float(centered.norm()),
            'effective_rank':float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum()))}


def probe(model,entities,tokenizer,device,task_info):
    model.eval();items=[i for e in entities for i in e['items']];task=items[0]['task']
    features={f'{stage}_{pool}':[] for stage in ['encoder','head','readout'] for pool in ['cls','mean']}
    selected=[]
    with torch.no_grad():
        for start in range(0,len(items),8):
            batch=exp.collate(items[start:start+8],tokenizer.pad_token_id,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h=model.encoder(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).last_hidden_state
                for pool in ['cls','mean']:features['encoder_'+pool].append(exp.pool_hidden(h,batch['attention_mask'],pool).float().cpu())
                h=h+model.type_emb(batch['qtype'])[:,None,:]
                if model.head is not None:
                    for layer in model.head.layers:h=layer(h,src_key_padding_mask=~batch['attention_mask'].bool())
                for pool in ['cls','mean']:
                    pooled=exp.pool_hidden(h,batch['attention_mask'],pool)
                    features['head_'+pool].append(pooled.float().cpu())
                    features['readout_'+pool].append(model.readout(pooled).float().cpu())
                    if pool==model.pooling:selected.append(pooled.detach())
    with torch.autocast('cuda',dtype=torch.bfloat16):
        h=model.readout(torch.cat(selected));logits=model.classifiers[task](h).float()
        scalar=model.regressors[task](h).squeeze(-1).float()
        labels=torch.tensor([i['label'] for i in items],device=device)
        spec=task_info[task];target=torch.tensor([(i['value']-spec['mean'])/spec['std'] for i in items],device=device)
        ce=torch.nn.functional.cross_entropy(logits,labels);mse=(scalar-target).square().mean()
    parameters=list(model.readout.parameters())
    gc=torch.cat([g.flatten() for g in torch.autograd.grad(ce,parameters,retain_graph=True)])
    gm=torch.cat([g.flatten() for g in torch.autograd.grad(mse,parameters)])
    denominator=gc.norm()*gm.norm()
    return {'features':{k:feature_stats(torch.cat(v)) for k,v in features.items()},
            'eval_ce':float(ce.detach()),'eval_standardized_mse':float(mse.detach()),
            'readout_gradient_ce_norm':float(gc.norm()),'readout_gradient_mse_norm':float(gm.norm()),
            'readout_gradient_cosine':float(torch.dot(gc,gm)/denominator) if denominator>0 else None,
            'gradient_scope':'readout parameters only, eval mode, all 32 fitted examples; not encoder gradient conflict'}
