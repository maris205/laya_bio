#!/usr/bin/env python3
"""Fixed-budget GFP train/dev validation; never opens a test split."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
import laya_multitask_experiment as exp
import laya_multitask_data as data


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def lr_at(step, updates=512, warmup=16):
    if step <= warmup:
        return 1e-4 * step / warmup
    return 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * (step - warmup) / (updates - warmup)))


def batches_for_epoch(rows, rng, batch_size=32):
    ordered = rows.copy()
    rng.shuffle(ordered)
    if len(ordered) % batch_size:
        raise ValueError('Protocol requires complete equal-sized batches')
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir', 'model-dir', 'laya-repo', 'output', 'provenance']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if shutil.disk_usage(a.output.parent).free < 3.4 * 2**30:
        raise RuntimeError('Insufficient room for atomic checkpoint replacement')
    a.output.mkdir()
    started = time.monotonic()
    state = {'status': 'preparing', 'created_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'test_access': False}
    def status(**kwargs):
        state.update(kwargs, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(a.output / 'status.json', state)
    status()
    provenance = json.loads(a.provenance.read_text())
    assert platform.python_version() == provenance['python']
    for name, version in provenance['packages'].items():
        assert importlib.metadata.version(name) == version, name
    for name, expected in provenance['initial_model_and_tokenizer_sha256'].items():
        assert sha_file(Path(name)) == expected, name
    shutil.copy2(a.provenance, a.output / 'initialization_provenance.json')
    torch.set_num_threads(8)
    seed = 20260924
    exp.core.set_seed(seed)
    device = torch.device('cuda')
    manifest_path = a.data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rows = {}
    for split in ['train', 'dev']:
        path = a.data_dir / manifest['outputs'][split]['file']
        assert sha_file(path) == manifest['outputs'][split]['sha256']
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
    info = {}
    for task in data.TASKS:
        rr = [r for r in rows['train'] if r['task'] == task]
        row = rr[0]
        spec = {'primitive': row['primitive'], 'n_outputs': len(data.LOCATIONS) if row['primitive'] == 'multilabel' else len(row['choices'])}
        if row['primitive'] == 'score':
            values = np.array([r['value'] for r in rr])
            spec.update(manifest['score_spec'], mean=float(values.mean()), std=max(1e-6, float(values.std())))
        info[task] = spec
    rows = {s: [r for r in rr if r['task'] == 'fluorescence'] for s, rr in rows.items()}
    assert len(rows['train']) == 1024 and len(rows['dev']) == 128
    overlap = {}
    for key in ['id', 'group', 'inputs']:
        sets = {s: {json.dumps(r[key], sort_keys=True) for r in rr} for s, rr in rows.items()}
        overlap[key] = len(sets['train'] & sets['dev'])
        assert overlap[key] == 0, key
    assert all(r['split'] == s for s, rr in rows.items() for r in rr)
    assert {r['source_split'] for r in rows['train']} == {'train'}
    assert {r['source_split'] for r in rows['dev']} == {'valid'}
    config = {'seed': seed, 'updates': 512, 'epochs': 16, 'effective_batch': 32, 'micro_batch': 4,
              'lr_peak': 1e-4, 'lr_final': 1e-5, 'warmup_updates': 16, 'schedule': 'linear warmup then cosine decay',
              'optimizer': 'AdamW', 'weight_decay': .01, 'clip_norm': 1., 'precision': 'BF16 autocast, FP32 weights',
              'kind': 'shared_heads', 'pooling': 'mean', 'native_readout': True, 'bypass_shared_head': False,
              'score_loss': 'joint', 'initialization': 'original pretrained model, never tiny-fit weights',
              'evaluation_steps': list(range(0, 513, 64)), 'selection': 'minimum dev scalar RMSE, including step 0; earlier step wins ties',
              'checkpoint_policy': 'retain only best-dev weights; retain all evaluation predictions including final',
              'early_stopping': False, 'dev_access': True, 'test_access': False,
              'manifest_sha256': sha_file(manifest_path), 'source_outputs': manifest['outputs'], 'task_info': info,
              'train_dev_overlaps': overlap, 'split_counts': {s: len(rr) for s, rr in rows.items()},
              'label_counts': {s: dict(Counter(r['label'] for r in rr)) for s, rr in rows.items()},
              'selected_ids': {s: [r['id'] for r in rr] for s, rr in rows.items()},
              'gpu': torch.cuda.get_device_name(),
              'limitations': ['One seed; reused development set; no untouched test estimate.',
                              'GFP variants share a parent; sequence/group separation is not family separation.',
                              'Data scale and optimizer schedule both change versus tiny-fit; not a causal architecture comparison.']}
    write_json(a.output / 'run_config.json', config)
    _, _, builder = exp.core.import_laya(str(a.laya_repo))
    tok = AutoTokenizer.from_pretrained(a.model_dir / 'tokenizer')
    entities = {s: exp.make_items(rr, tok, builder, 'shared_heads', seed, False, manifest['max_length']) for s, rr in rows.items()}
    model, cfg = exp.build('shared_heads', a.model_dir, a.laya_repo, info, device, pooling='mean', native_readout=True)
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(a.output / 'run_config.json', config)
    checkpoint = a.output / 'checkpoint'
    checkpoint.mkdir()
    write_json(checkpoint / 'rl_agent_config.json', cfg)
    write_json(checkpoint / 'task_info.json', info)
    history = []
    best = None
    def evaluate_step(step):
        nonlocal best
        entry = {'step': step, 'epochs': step / 32, 'seconds': time.monotonic() - started}
        for split in ['train', 'dev']:
            records, seconds = exp.evaluate(model, 'shared_heads', entities[split], tok, device, info, 8)
            entry[split] = exp.metrics(records, info)['fluorescence']
            entry[split]['scalar_prediction']['prediction_std'] = float(np.std([r['scalar_prediction'] for r in records]))
            entry[split + '_evaluation_seconds'] = seconds
            with (a.output / f'{split}_step{step:04d}_predictions.jsonl').open('w') as f:
                for record in records:
                    f.write(json.dumps(record, allow_nan=False) + '\n')
        if best is None or entry['dev']['scalar_prediction']['rmse'] < best['dev']['scalar_prediction']['rmse']:
            tmp = checkpoint / 'model.tmp.safetensors'
            save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, str(tmp))
            tmp.replace(checkpoint / 'model.safetensors')
            best = entry
            write_json(a.output / 'best_checkpoint.json', {'step': step, 'dev': entry['dev']})
        history.append(entry)
        with (a.output / 'learning_curve.jsonl').open('a') as f:
            f.write(json.dumps(entry, allow_nan=False) + '\n')
        print(json.dumps({'event': 'evaluation', **entry}), flush=True)
        status(status='running', step=step, best_step=best['step'], latest=entry)
    evaluate_step(0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    rng = random.Random(seed)
    seen = Counter()
    torch.cuda.reset_peak_memory_stats()
    step = 0
    for epoch in range(1, 17):
        for chunk in batches_for_epoch(entities['train'], rng):
            step += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step)
            items = [i for e in chunk for i in e['items']]
            assert len(items) == 32
            seen.update(e['id'] for e in chunk)
            loss_value = 0.
            for offset in range(0, 32, 4):
                loss = exp.loss_values(model, 'shared_heads', items[offset:offset + 4], tok, device, info).sum() / 32
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                loss_value += float(loss.detach())
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            optimizer.step()
            trace = {'step': step, 'epoch': epoch, 'loss': loss_value, 'lr': lr_at(step), 'gradient_norm': norm,
                     'batch_ids': [e['id'] for e in chunk], 'seconds': time.monotonic() - started}
            with (a.output / 'training.jsonl').open('a') as f:
                f.write(json.dumps(trace, allow_nan=False) + '\n')
            if step % 64 == 0:
                evaluate_step(step)
    assert step == 512 and len(seen) == 1024 and set(seen.values()) == {16}
    seconds = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, optimizer, params, loss
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = exp.build('shared_heads', a.model_dir, a.laya_repo, info, device, checkpoint)
    reloaded, _ = exp.evaluate(model, 'shared_heads', entities['dev'], tok, device, info, 8)
    expected = [json.loads(l) for l in (a.output / f'dev_step{best["step"]:04d}_predictions.jsonl').read_text().splitlines()]
    assert [r['id'] for r in expected] == [r['id'] for r in reloaded]
    prob_error = max(abs(x - y) for r, s in zip(reloaded, expected) for x, y in zip(r['probs'], s['probs']))
    scalar_error = max(abs(r['scalar_prediction'] - s['scalar_prediction']) for r, s in zip(reloaded, expected))
    assert prob_error < 1e-6 and scalar_error < 1e-6
    summary = {**config, 'complete': True, 'actual_updates': step, 'presentations': sum(seen.values()),
               'per_example_presentations': 16, 'initial': history[0], 'best': best, 'final': history[-1],
               'training_and_evaluation_seconds': seconds, 'peak_allocated_gib': peak,
               'reload_max_probability_error': prob_error, 'reload_max_scalar_error': scalar_error,
               'checkpoint_sha256': sha_file(checkpoint / 'model.safetensors')}
    write_json(a.output / 'summary.json', summary)
    status(status='complete', best_step=best['step'], total_wall_seconds=time.monotonic() - started)
    print(json.dumps({'event': 'complete', 'best_step': best['step']}), flush=True)


if __name__ == '__main__':
    main()
