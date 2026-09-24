#!/usr/bin/env python3
"""Training-only 2x2 sample-scale/LR probe, with a fixed low-LR extension."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
import laya_multitask_experiment as exp
import laya_multitask_data as data
from laya_gfp_development import sha_file, write_json, batches_for_epoch, lr_at


def nested_subset(rows, count):
    pool = [r for r in rows if r['task'] == 'fluorescence']
    if count == 1024:
        assert len(pool) == count
        return pool
    if count != 32:
        raise ValueError('Only predeclared sample sizes supported')
    chosen = []
    classes = sorted({r['label'] for r in pool})
    for index, label in enumerate(classes):
        candidates = sorted([r for r in pool if r['label'] == label], key=lambda r: data.sha('microfit-v1:' + r['id']))
        chosen.extend(candidates[:count // len(classes) + int(index < count % len(classes))])
    chosen.sort(key=lambda r: data.sha('microfit-order-v1:' + r['id']))
    assert len(chosen) == count
    return chosen


def probe_lr(step, scale):
    # Exact original 512-step schedule, then hold its final rate for extension.
    return lr_at(min(step, 512)) * scale


def metric(records, info):
    result = exp.metrics(records, info)['fluorescence']
    result['scalar_prediction']['prediction_std'] = float(np.std([r['scalar_prediction'] for r in records]))
    result['scalar_prediction']['normalized_rmse'] = result['scalar_prediction']['rmse'] / info['fluorescence']['std']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['data-dir', 'model-dir', 'laya-repo', 'output', 'reference']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--samples', type=int, choices=[32, 1024], required=True)
    p.add_argument('--lr-scale', type=float, choices=[1., .2], required=True)
    p.add_argument('--updates', type=int, choices=[512, 1024], required=True)
    a = p.parse_args()
    assert (a.samples, a.lr_scale, a.updates) in {(32, 1., 512), (32, .2, 512), (1024, .2, 1024)}
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir()
    started = time.monotonic()
    torch.set_num_threads(8)
    exp.core.set_seed(20260924)
    device = torch.device('cuda')
    manifest_path = a.data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    train_path = a.data_dir / manifest['outputs']['train']['file']
    assert sha_file(train_path) == manifest['outputs']['train']['sha256']
    all_rows = [json.loads(l) for l in train_path.read_text().splitlines()]
    assert all(r['split'] == 'train' for r in all_rows)
    info = {}
    for task in data.TASKS:
        rr = [r for r in all_rows if r['task'] == task]
        row = rr[0]
        spec = {'primitive': row['primitive'], 'n_outputs': len(data.LOCATIONS) if row['primitive'] == 'multilabel' else len(row['choices'])}
        if row['primitive'] == 'score':
            y = np.array([r['value'] for r in rr])
            spec.update(manifest['score_spec'], mean=float(y.mean()), std=max(1e-6, float(y.std())))
        info[task] = spec
    selected = nested_subset(all_rows, a.samples)
    anchor_rows = nested_subset(all_rows, 32)
    anchor_ids = {r['id'] for r in anchor_rows}
    schedule = sorted({0, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512} | ({768, 1024} if a.updates == 1024 else set()))
    config = {'samples': a.samples, 'lr_scale': a.lr_scale, 'updates': a.updates, 'seed': 20260924,
              'effective_batch': 32, 'micro_batch': 4, 'lr_peak': 1e-4 * a.lr_scale, 'lr_at_512': 1e-5 * a.lr_scale,
              'warmup_updates': 16, 'schedule': 'original 512-update cosine schedule; hold final LR after step 512',
              'pooling': 'mean', 'native_readout': True, 'bypass_shared_head': False, 'score_loss': 'joint',
              'weight_decay': .01, 'clip_norm': 1., 'initialization': 'original pretrained weights',
              'precision': 'BF16 autocast, FP32 weights', 'dev_access': False, 'test_access': False,
              'early_stopping': False, 'checkpoint_selection': 'final only', 'evaluation_steps': schedule,
              'selected_ids': [r['id'] for r in selected], 'anchor_ids': [r['id'] for r in anchor_rows],
              'label_counts': dict(Counter(r['label'] for r in selected)), 'task_info': info,
              'train_sha256': sha_file(train_path), 'manifest_sha256': sha_file(manifest_path),
              'reference_directory': str(a.reference), 'gpu': torch.cuda.get_device_name(),
              'comparisons': '2x2 at step 512 with historical N1024/high-LR training predictions; within-trajectory extension at N1024/low LR',
              'limitations': ['One seed; subset composition changes with N.', 'Fixed updates and fixed per-example exposure are different comparisons.',
                              'At equal exposure across N, update count and LR history differ; this is not an isolated causal sample-size effect.',
                              'A failed extension only rejects adequacy of this budget/continuation; it does not prove training cannot converge.']}
    write_json(a.output / 'run_config.json', config)
    _, _, builder = exp.core.import_laya(str(a.laya_repo))
    tok = AutoTokenizer.from_pretrained(a.model_dir / 'tokenizer')
    entities = exp.make_items(selected, tok, builder, 'shared_heads', 20260924, False, manifest['max_length'])
    model, cfg = exp.build('shared_heads', a.model_dir, a.laya_repo, info, device, pooling='mean', native_readout=True)
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(a.output / 'run_config.json', config)
    curve = []
    def evaluate_step(step):
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        records, seconds = exp.evaluate(model, 'shared_heads', entities, tok, device, info, 8)
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        if step == 0:
            expected = {r['id']: r for l in (a.reference / 'train_step0000_predictions.jsonl').read_text().splitlines() if (r := json.loads(l))['id'] in set(config['selected_ids'])}
            # Different panel padding may change BF16 output slightly for N32.
            max_prob = max(abs(x - y) for r in records for x, y in zip(r['probs'], expected[r['id']]['probs']))
            max_scalar = max(abs(r['scalar_prediction'] - expected[r['id']]['scalar_prediction']) for r in records)
            check = {'max_probability_difference': max_prob, 'max_scalar_difference': max_scalar, 'same_panel_order': a.samples == 1024}
            if a.samples == 1024:
                assert max_prob == 0. and max_scalar == 0.
            write_json(a.output / 'initial_prediction_check.json', check)
        entry = {'step': step, 'per_example_exposures': step * 32 / a.samples, 'training': metric(records, info),
                 'anchor32': metric([r for r in records if r['id'] in anchor_ids], info), 'evaluation_seconds': seconds}
        with (a.output / f'train_step{step:04d}_predictions.jsonl').open('w') as f:
            for r in records:
                f.write(json.dumps(r, allow_nan=False) + '\n')
        with (a.output / 'learning_curve.jsonl').open('a') as f:
            f.write(json.dumps(entry, allow_nan=False) + '\n')
        curve.append(entry)
        write_json(a.output / 'status.json', {'status': 'running', 'step': step, 'latest': entry})
        print(json.dumps({'event': 'evaluation', **entry}), flush=True)
        return records
    evaluate_step(0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=probe_lr(1, a.lr_scale), weight_decay=.01)
    rng = random.Random(20260924)
    seen = Counter()
    torch.cuda.reset_peak_memory_stats()
    step = 0
    for epoch in range(1, a.updates * 32 // a.samples + 1):
        for chunk in batches_for_epoch(entities, rng):
            step += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = probe_lr(step, a.lr_scale)
            items = [i for e in chunk for i in e['items']]
            seen.update(e['id'] for e in chunk)
            loss_value = 0.
            for offset in range(0, 32, 4):
                loss = exp.loss_values(model, 'shared_heads', items[offset:offset + 4], tok, device, info).sum() / 32
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                loss_value += float(loss.detach())
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            optimizer.step()
            trace = {'step': step, 'epoch': epoch, 'loss': loss_value, 'lr': probe_lr(step, a.lr_scale), 'gradient_norm': norm,
                     'batch_ids': [e['id'] for e in chunk], 'seconds': time.monotonic() - started}
            with (a.output / 'training.jsonl').open('a') as f:
                f.write(json.dumps(trace, allow_nan=False) + '\n')
            if step in schedule:
                final_records = evaluate_step(step)
    assert step == a.updates and len(seen) == a.samples and set(seen.values()) == {a.updates * 32 // a.samples}
    peak = torch.cuda.max_memory_allocated() / 2**30
    checkpoint = a.output / 'checkpoint'
    checkpoint.mkdir()
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, str(checkpoint / 'model.safetensors'))
    write_json(checkpoint / 'rl_agent_config.json', cfg)
    write_json(checkpoint / 'task_info.json', info)
    del model, optimizer, params, loss
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = exp.build('shared_heads', a.model_dir, a.laya_repo, info, device, checkpoint)
    reloaded, _ = exp.evaluate(model, 'shared_heads', entities, tok, device, info, 8)
    assert [r['id'] for r in reloaded] == [r['id'] for r in final_records]
    prob_error = max(abs(x - y) for r, s in zip(reloaded, final_records) for x, y in zip(r['probs'], s['probs']))
    scalar_error = max(abs(r['scalar_prediction'] - s['scalar_prediction']) for r, s in zip(reloaded, final_records))
    assert prob_error == 0 and scalar_error == 0
    summary = {**config, 'complete': True, 'initial': curve[0], 'final': curve[-1], 'actual_updates': step,
               'total_presentations': sum(seen.values()), 'per_example_presentations': a.updates * 32 // a.samples,
               'total_seconds': time.monotonic() - started, 'peak_allocated_gib': peak,
               'reload_max_probability_error': prob_error, 'reload_max_scalar_error': scalar_error,
               'checkpoint_sha256': sha_file(checkpoint / 'model.safetensors')}
    write_json(a.output / 'summary.json', summary)
    write_json(a.output / 'status.json', {'status': 'complete', 'step': step})
    print(json.dumps({'event': 'complete', 'step': step}), flush=True)


if __name__ == '__main__':
    main()
