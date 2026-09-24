"""Matched raw-input / full biological BPE supervision, without CPT or test access."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil

import torch

import laya_formal_experiment as core
from laya_direct_bpe import DirectBPE, DEFAULT, digest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition', choices=['raw', 'full_bpe'], default='raw')
    p.add_argument('--representation-dir', type=Path, default=DEFAULT)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--eligible-ids', default='artifacts/laya_formal_data/eligible_ids.json')
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--updates', type=int, default=3005)
    p.add_argument('--micro-batch', type=int, default=8)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--eval-batch', type=int, default=16)
    p.add_argument('--device', default='cuda')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--smoke', action='store_true', help='4 train and 2 eval rows/class; never a formal result')
    return p.parse_args()


def load_rows(a):
    eligible = core.load_eligible_ids(a.eligible_ids)
    rows = {s: [r for r in core.load_split(core.ROOT/'artifacts/laya_formal_data', 'both', s)
                if r['id'] in eligible] for s in ('train','selection_dev','calibration')}
    if a.smoke:
        for split, values in rows.items():
            groups = defaultdict(list)
            for r in values:
                groups[(r['task'], r['label'])].append(r)
            n = 4 if split == 'train' else 2
            rows[split] = [r for group in groups.values() for r in group[:n]]
    return rows


def make_items(rows, rep, condition, builder, seed, shuffle=False):
    items = []
    for row in rows:
        order = list(range(len(row['choices'])))
        if shuffle:
            h = hashlib.sha256(f"{seed}:{row['id']}".encode()).digest()
            random.Random(int.from_bytes(h[:8], 'big')).shuffle(order)
        ids, markers = rep.build(row, condition, builder, order)
        items.append({'ids': ids, 'markers': markers, 'qtype': 0,
                      'label': order.index(row['label']), 'task': row['task'], 'id': row['id'],
                      'n_tokens': len(ids), 'full_tokens': len(ids), 'truncated': False})
    return items


def stats(items):
    values = sorted(x['n_tokens'] for x in items)
    return {'n': len(values), 'length_p50': values[len(values)//2],
            'length_p95': values[max(0, int(len(values)*.95)-1)],
            'length_max': max(values), 'truncated_count': 0,
            'membership_sha256': hashlib.sha256('\n'.join(x['id'] for x in items).encode()).hexdigest()}


def audit(a, rep, rows, builder):
    report = {'no_cpt': True, 'test_access': False, 'seed': a.seed,
              'eligibility_policy': 'same frozen common-complete sample IDs as first 64-token pair',
              'representation_sha256': digest(rep.root/'metadata.json'), 'conditions': {}}
    all_sets = {}
    for condition in ('raw','full_bpe'):
        all_sets[condition] = {s: make_items(values, rep, condition, builder, a.seed, s == 'train')
                               for s, values in rows.items()}
        report['conditions'][condition] = {s: {task: stats([x for x in items if x['task']==task])
                                               for task in sorted({x['task'] for x in items})}
                                            for s,items in all_sets[condition].items()}
        print(f'Audited {condition}: '+str({s: stats(v) for s,v in all_sets[condition].items()}), flush=True)
    for split in rows:
        for raw, expanded in zip(all_sets['raw'][split],all_sets['full_bpe'][split]):
            assert (raw['id'],raw['label'],raw['markers']) == (expanded['id'],expanded['label'],expanded['markers'])
            end = raw['markers'][-1]
            assert raw['ids'][:end+1] == expanded['ids'][:end+1]
    old_size = len(rep.base)
    counts = Counter(i for x in all_sets['full_bpe']['train'] for i in x['ids'] if i >= old_size)
    report['train_token_coverage'] = {k: {'added': len(rep.maps[k]),
        'observed': sum(i in counts for i in rep.maps[k].values()),
        'occurrences': sum(counts[i] for i in rep.maps[k].values())} for k in rep.maps}
    report['same_samples_labels_candidate_markers'] = True
    report['full_sequence_reconstruction_checked'] = True
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'input_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return all_sets


def initialize(model, rep):
    old_size = len(rep.base)
    old = model.encoder.get_input_embeddings().weight.detach().clone()
    model.encoder.resize_token_embeddings(len(rep.expanded), mean_resizing=False)
    emb = model.encoder.get_input_embeddings().weight
    maximum_error = 0.0
    with torch.no_grad():
        for start in range(0,len(rep.entries),512):
            entries = rep.entries[start:start+512]
            width = max(len(e['base_ids']) for e in entries)
            indexes = torch.zeros((len(entries),width),dtype=torch.long,device=emb.device)
            mask = torch.zeros((len(entries),width),dtype=emb.dtype,device=emb.device)
            for j,e in enumerate(entries):
                indexes[j,:len(e['base_ids'])] = torch.tensor(e['base_ids'],device=emb.device)
                mask[j,:len(e['base_ids'])] = 1
            means = (old[indexes]*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
            dst = torch.tensor([e['expanded_id'] for e in entries],device=emb.device)
            emb[dst] = means
            maximum_error = max(maximum_error,float((emb[dst]-means).abs().max().cpu()))
        old_preserved = torch.equal(emb[:old_size],old)
    if not old_preserved or maximum_error != 0 or not torch.isfinite(emb).all():
        raise ValueError('Embedding initialization verification failed')
    return {'old_vocab_size': old_size, 'new_vocab_size': len(rep.expanded),
            'added_by_modality': rep.meta['added_by_modality'],
            'raw_fragment_mean_initialization': True, 'old_rows_unchanged_at_initialization': old_preserved,
            'maximum_initialization_error': maximum_error, 'fallback_initializations': 0}


def gradient_probe(model, rep, items, device):
    out = {}
    old_size = len(rep.base)
    model.eval()
    for task in sorted({x['task'] for x in items}):
        model.zero_grad(set_to_none=True)
        chosen = [x for x in items if x['task']==task][:2]
        batch = core.move(core.collate(chosen, rep.expanded.pad_token_id),device)
        with core.autocast_context(device,torch.bfloat16):
            logits,_ = model(batch['input_ids'],batch['attention_mask'],batch['marker_pos'],batch['marker_mask'],batch['qtype'])
            loss = torch.nn.functional.cross_entropy(logits.masked_fill(~batch['marker_mask'],-1e4),batch['label'])
        loss.backward()
        grad = model.encoder.get_input_embeddings().weight.grad
        active = sorted({i for x in chosen for i in x['ids'] if i >= old_size})
        norm = float(grad[active].norm().detach().cpu())
        if not torch.isfinite(loss) or not torch.isfinite(grad).all() or not norm > 0:
            raise ValueError('New embedding receives no finite task gradient')
        out[task] = {'active_new_ids': len(active),'gradient_norm': norm}
    model.zero_grad(set_to_none=True)
    return out


def main():
    a = parse_args()
    if (a.output_dir/'summary.json').exists():
        raise FileExistsError('Completed run exists; choose a new output directory')
    core.set_seed(a.seed)
    rep = DirectBPE(a.representation_dir)
    rows = load_rows(a)
    _, builder_model, builder_sequence = core.import_laya('vendor/laya')
    if a.audit_only:
        audit(a,rep,rows,builder_sequence)
        return
    items = {s: make_items(values, rep, a.condition, builder_sequence, a.seed, s=='train')
             for s,values in rows.items()}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    item_stats = {s:stats(v) for s,v in items.items()}
    print(json.dumps({'condition':a.condition,'smoke':a.smoke,'item_stats':item_stats},indent=2),flush=True)
    model_dir = core.ROOT/'artifacts/laya_model'
    cfg = json.loads((model_dir/'rl_agent_config.json').read_text())
    cfg.update(max_len=1024,head_max_len=256,gradient_checkpointing=True)
    device = torch.device(a.device)
    model = core.load_model(model_dir,cfg,builder_model,device,trainable=True)
    expansion = initialize(model,rep) if a.condition=='full_bpe' else None
    token = rep.expanded if a.condition=='full_bpe' else rep.base
    probes = gradient_probe(model,rep,items['train'],device) if expansion else None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    core.set_seed(a.seed)  # Equal dropout RNG start after representation-specific initialization/probes.
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    updates = 2 if a.smoke else a.updates
    training = core.train_model(model,items['train'],token,device,updates,a.micro_batch,a.grad_accum,
                                2e-5,.01,a.seed,25,3,.05)
    if device.type=='cuda':
        training['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        training['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
    predictions = {s:core.evaluate(model,items[s],token,device,a.eval_batch) for s in ('selection_dev','calibration')}
    groups = defaultdict(list)
    for r in predictions['calibration']:
        groups[r['task']].append(r)
    temps = {t: core.fit_metric_temperature([r['logits'] for r in values],[r['label'] for r in values],
                                          n_classes=values[0]['n_classes']) for t,values in groups.items()}
    temperature_values = {t:v['temperature'] for t,v in temps.items()}
    metrics = {s:{'raw':core.metric_from_records(records),
                  'calibrated':core.metric_from_records(records,temperature_values)} for s,records in predictions.items()}
    if any(not torch.isfinite(torch.tensor(r['logits'])).all() for records in predictions.values() for r in records):
        raise FloatingPointError('Non-finite evaluation logits')
    checkpoint = a.output_dir/'checkpoint'
    core.save_checkpoint(model,token,cfg,checkpoint)
    shutil.copytree(rep.root,checkpoint/'representation')
    disk_rep = DirectBPE(checkpoint/'representation')
    disk_items = make_items(rows['selection_dev'],disk_rep,a.condition,builder_sequence,a.seed)
    if disk_items != items['selection_dev']:
        raise ValueError('Saved representation changed evaluation IDs or candidate markers')
    del model
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    reloaded = core.fresh_reload(checkpoint,cfg,builder_model,device)
    again = core.evaluate(reloaded,disk_items,token,device,a.eval_batch)
    match = len(again)==len(predictions['selection_dev']) and all(
        x['id']==y['id'] and len(x['logits'])==len(y['logits']) and
        max(abs(l-r) for l,r in zip(x['logits'],y['logits'])) < 1e-5
        for x,y in zip(again,predictions['selection_dev']))
    if not match:raise ValueError('Checkpoint reload changed logits')
    for split,records in predictions.items():
        (a.output_dir/f'{split}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary = {'formal':not a.smoke,'smoke_only':a.smoke,'condition':a.condition,'seed':a.seed,
        'no_cpt':True,'test_access':False,'evaluated_splits':['selection_dev','calibration'],
        'representation_sha256':digest(rep.root/'metadata.json'),'eligible_id_filter':a.eligible_ids,
        'n_rows':{s:len(v) for s,v in rows.items()},'item_stats':item_stats,'training':training,
        'trainable_parameters':trainable,'embedding_trainable':True,'evaluation':metrics,
        'calibration_temperature':temps,'calibration_temperature_fit_on':'calibration',
        'expansion':expansion,'new_embedding_gradient_probe':probes,
        'checkpoint':str(checkpoint),'checkpoint_reload_input_ids_match':True,
        'checkpoint_reload_logits_match':match,'selection_policy':'final fixed-budget checkpoint'}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
