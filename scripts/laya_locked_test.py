"""Lock and evaluate twelve existing checkpoints; no training or temperature fitting."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
import laya_formal_experiment as core
import laya_direct_experiment as direct
import laya_control_experiment as control
from laya_direct_bpe import DirectBPE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/laya_locked_test'
DATA = ROOT/'artifacts/laya_formal_data'
SEEDS = (20260922, 20260923, 20260924)
CONDITIONS = ('raw', 'full_bpe', 'b1', 'text_only')
TASKS = ('promoter_detection', 'fold_class')
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'mcc', 'nll', 'brier', 'ece15')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(path.read_text())
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def status(state, **kwargs):
    value = {'state':state, 'updated_utc':now(), **kwargs}
    write(OUT/'status.json', value)
    print(json.dumps(value), flush=True)


def source(condition, seed):
    parent = 'laya_direct_legacy' if condition in ('raw', 'full_bpe') else 'laya_controls'
    return ROOT/'artifacts'/parent/f'{condition}_seed{seed}'


def lock():
    if (OUT/'manifest.json').exists(): raise FileExistsError('Test protocol already locked')
    paths = set()
    # Preserve the previously frozen training/data provenance chain.
    inherited = ROOT/'artifacts/laya_controls/run_manifest.json'
    for name, expected in read(inherited)['hashes'].items():
        if sha(ROOT/name) != expected: raise RuntimeError(f'Original frozen file changed: {name}')
        paths.add(ROOT/name)
    paths.add(inherited)
    paths.update(DATA.glob('*.jsonl'))
    paths.add(DATA/'eligible_ids.json')
    paths.add(Path(__file__).resolve())
    paths.add(ROOT/'research/laya_locked_test_protocol.md')
    paths.update((ROOT/'vendor/laya').rglob('*.py'))
    runs = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            folder = source(condition, seed)
            s = read(folder/'summary.json')
            assert s['formal'] and not s['smoke_only'] and not s['test_access'] and s['no_cpt']
            assert s.get('kind', s['condition']) == condition and s['seed'] == seed
            assert s['checkpoint_reload_logits_match'] and s['checkpoint_reload_input_ids_match']
            assert s['training']['finite'] and s['training']['updates'] == 3005
            assert s['training']['examples_consumed'] == 96160
            assert s['calibration_temperature_fit_on'] == 'calibration'
            temps = {t:s['calibration_temperature'][t]['temperature'] for t in TASKS}
            assert all(math.isfinite(x) and x > 0 for x in temps.values())
            paths.add(folder/'summary.json')
            paths.add(folder/'selection_dev_predictions.jsonl')
            paths.update(p for p in (folder/'checkpoint').rglob('*') if p.is_file())
            runs.append({'condition':condition, 'seed':seed, 'source':str(folder.relative_to(ROOT)),
                         'temperatures':temps})
    manifest = {'locked_utc':now(), 'runs':runs, 'expected_counts':dict(zip(TASKS, (2145, 1952))),
                'primary_model':'full_bpe', 'primary_comparisons':['b1', 'text_only'],
                'representation_reference':'raw', 'metrics':METRICS, 'batch_size':16,
                'policy':'All seeds; fixed final checkpoints; no training, selection, or temperature refitting',
                'hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}}
    write(OUT/'manifest.json', manifest)
    status('locked', n_models=len(runs), test_predictions_started=False)


def verify():
    manifest = read(OUT/'manifest.json')
    for name, expected in manifest['hashes'].items():
        if sha(ROOT/name) != expected: raise RuntimeError(f'Locked file changed: {name}')
    return manifest


def items(rows, rep, condition, builder, seed):
    if condition in ('b1', 'text_only'):
        return control.make_items(rows, rep, condition, builder, seed, training=False)
    return direct.make_items(rows, rep, condition, builder, seed, shuffle=False)


def validate(records, rows):
    if len(records) != len(rows) or len({r['id'] for r in records}) != len(rows):
        raise ValueError('Missing or duplicate predictions')
    for r, truth in zip(records, rows):
        if any(r[k] != truth[k] for k in ('id', 'task', 'label')):
            raise ValueError('Prediction membership/order/label mismatch')
        if r['n_classes'] != len(truth['choices']) or len(r['logits']) != r['n_classes']:
            raise ValueError('Wrong class count')
        if not all(math.isfinite(x) for x in r['logits']): raise ValueError('Non-finite logits')


def moments(values):
    return {'mean':statistics.mean(values), 'std':statistics.stdev(values), 'n':len(values)}


def report(manifest, rows):
    summaries = {}
    for run in manifest['runs']:
        condition, seed = run['condition'], run['seed']
        folder = OUT/f'{condition}_seed{seed}'
        s = read(folder/'summary.json')
        records = [json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
        validate(records, rows)
        metrics = {'raw':core.metric_from_records(records),
                   'calibrated':core.metric_from_records(records, run['temperatures'])}
        assert s['metrics'] == metrics and s['dev_logits_reproduced']
        assert s['manifest_sha256'] == sha(OUT/'manifest.json')
        summaries[(condition, seed)] = s
    aggregate = {c:{mode:{t:{k:moments([summaries[c, seed]['metrics'][mode][t][k] for seed in SEEDS])
                            for k in METRICS} for t in TASKS} for mode in ('raw', 'calibrated')}
                 for c in CONDITIONS}
    paired = {}
    for reference in ('raw', 'b1', 'text_only'):
        paired[reference] = {}
        for t in TASKS:
            paired[reference][t] = {}
            for k in METRICS:
                diffs = [summaries['full_bpe', seed]['metrics']['raw'][t][k] -
                         summaries[reference, seed]['metrics']['raw'][t][k] for seed in SEEDS]
                base = aggregate[reference]['raw'][t][k]['mean']
                paired[reference][t][k] = {**moments(diffs), 'per_seed':dict(zip(SEEDS, diffs)),
                                          'relative_change_percent':100*statistics.mean(diffs)/base if base else None}
    result = {'generated_utc':now(), 'manifest_sha256':sha(OUT/'manifest.json'),
              'split':'test', 'no_cpt':True, 'n_models':12, 'counts':manifest['expected_counts'],
              'aggregate':aggregate, 'paired_full_bpe_minus_reference':paired,
              'runs':[summaries[c, seed] for seed in SEEDS for c in CONDITIONS]}
    write(OUT/'results.json', result)
    def fmt(c, t, k, mode='raw', scale=100):
        x = aggregate[c][mode][t][k]
        return f"{x['mean']*scale:.2f} ± {x['std']*scale:.2f}"
    lines = ['# Laya fixed-checkpoint test results', '',
             'All 12 locked checkpoints evaluated once on the common eligible test subset. '
             'Values are mean ± sample standard deviation across three seeds, not confidence intervals. '
             'No model selection, training, or temperature refitting used test results.', '',
             'DNA n=2,145; protein fold n=1,952. All inputs are complete (no truncation).', '',
             '| Model | DNA accuracy (%) | DNA macro-F1 (%) | Fold accuracy (%) | Fold macro-F1 (%) |',
             '|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        lines.append('| '+c+' | '+' | '.join(fmt(c,t,k) for t in TASKS for k in ('accuracy','macro_f1'))+' |')
    lines += ['', '## Calibration (temperatures fitted previously on calibration split)', '',
              '| Model | DNA raw NLL | DNA calibrated NLL | Fold raw NLL | Fold calibrated NLL |',
              '|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        lines.append('| '+c+' | '+' | '.join(fmt(c,t,'nll',mode,1) for t in TASKS for mode in ('raw','calibrated'))+' |')
    lines += ['', '## All seeds', '', '| Model | Seed | DNA accuracy (%) | Fold accuracy (%) | Fold macro-F1 (%) |',
              '|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        for c in CONDITIONS:
            m = summaries[c,seed]['metrics']['raw']
            lines.append(f"| {c} | {seed} | {100*m[TASKS[0]]['accuracy']:.2f} | {100*m[TASKS[1]]['accuracy']:.2f} | {100*m[TASKS[1]]['macro_f1']:.2f} |")
    lines += ['', '## Paired differences: full BPE minus reference', '',
              '| Reference | Task | Accuracy delta (pp) | Relative accuracy change (%) | Macro-F1 delta (pp) |',
              '|---|---|---:|---:|---:|']
    for ref in paired:
        for t in TASKS:
            a, f = paired[ref][t]['accuracy'], paired[ref][t]['macro_f1']
            lines.append(f"| {ref} | {t} | {100*a['mean']:+.2f} ± {100*a['std']:.2f} | {a['relative_change_percent']:+.2f} | {100*f['mean']:+.2f} ± {100*f['std']:.2f} |")
    lines += ['', '## Scope', '',
              'The fixed-class B1 comparison concerns the implemented CLS readout with two task-specific output matrices. '
              'This is not evidence against every possible fixed-class architecture. '
              'The BPE vocabulary was trained on historical external corpora whose overlap with this benchmark is not fully established. '
              'The eligible subset excludes examples that failed earlier shared complete-input checks. '
              'No claim of homology-independent generalization or statistical significance is made. '
              'Previously measured candidate-order sensitivity remains a dev diagnostic; no additional test perturbations were selected.', '',
              'Machine-readable source: `artifacts/laya_locked_test/results.json`; '
              'per-sample logits and labels: `artifacts/laya_locked_test/*/predictions.jsonl`.', '']
    (ROOT/'research/laya_locked_test_results.md').write_text('\n'.join(lines))


def run():
    manifest = verify()  # Fails before loading test labels if no lock or hashes differ.
    eligible = core.load_eligible_ids(str(DATA/'eligible_ids.json'))
    rows = [r for r in core.load_split(DATA, 'both', 'test') if r['id'] in eligible]
    dev = [r for r in core.load_split(DATA, 'both', 'selection_dev') if r['id'] in eligible]
    assert dict(Counter(r['task'] for r in rows)) == manifest['expected_counts']
    assert len({r['id'] for r in rows}) == len(rows)
    _, build_model, builder = core.import_laya('vendor/laya')
    device = torch.device('cuda')
    for index, spec in enumerate(manifest['runs']):
        condition, seed = spec['condition'], spec['seed']
        output = OUT/f'{condition}_seed{seed}'
        if output.exists(): raise FileExistsError(f'Refusing to overwrite/re-evaluate {output}')
        status('running', completed=index, total=12, condition=condition, seed=seed)
        folder = ROOT/spec['source']; checkpoint = folder/'checkpoint'
        rep = DirectBPE(checkpoint/'representation')
        cfg = read(checkpoint/'rl_agent_config.json')
        core.set_seed(seed)
        model = (control.fresh_reload(checkpoint, cfg, build_model, device, condition)
                 if condition in ('b1','text_only') else core.fresh_reload(checkpoint, cfg, build_model, device))
        tokenizer = rep.base if condition == 'raw' else rep.expanded
        actual = core.evaluate(model, items(dev, rep, condition, builder, seed), tokenizer, device, 16)
        previous = [json.loads(x) for x in (folder/'selection_dev_predictions.jsonl').read_text().splitlines()]
        validate(actual, dev)
        assert len(actual) == len(previous)
        max_error = max(abs(a-b) for x,y in zip(actual, previous) for a,b in zip(x['logits'],y['logits']))
        assert all(x['id'] == y['id'] and x['n_classes'] == y['n_classes'] for x,y in zip(actual,previous))
        if max_error >= 1e-5: raise ValueError(f'Dev logits changed: {max_error}')
        test_items = items(rows, rep, condition, builder, seed)
        assert all(not x['truncated'] and x['n_tokens'] <= 1024 for x in test_items)
        started = time.time()
        predictions = core.evaluate(model, test_items, tokenizer, device, 16)
        validate(predictions, rows)
        metrics = {'raw':core.metric_from_records(predictions),
                   'calibrated':core.metric_from_records(predictions, spec['temperatures'])}
        s = {'condition':condition, 'seed':seed, 'split':'test', 'test_access':True, 'no_cpt':True,
             'completed_utc':now(), 'manifest_sha256':sha(OUT/'manifest.json'),
             'source':spec['source'], 'n_rows':len(rows), 'item_stats':direct.stats(test_items),
             'dev_logits_reproduced':True, 'dev_max_logit_error':max_error,
             'temperatures':spec['temperatures'], 'temperature_fit_on':'previous calibration split',
             'evaluation_seconds':time.time()-started, 'metrics':metrics}
        output.mkdir()
        (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        write(output/'summary.json', s)
        del model
        gc.collect(); torch.cuda.empty_cache()
        status('model_complete', completed=index+1, total=12, condition=condition, seed=seed)
    verify()
    report(manifest, rows)
    status('complete', completed=12, total=12, no_training=True, no_temperature_refit=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lock', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'worker.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: lock() if args.lock else run()
        except Exception as exc:
            status('failed', error=repr(exc))
            raise
