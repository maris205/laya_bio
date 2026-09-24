#!/usr/bin/env python3
"""Continue the first fixed-budget pair after the already running M1.

Checks completed results and frozen input/code hashes before launching another
GPU process. Does not evaluate test or choose hyperparameters from calibration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'artifacts/laya_formal'


def write_status(value):
    value['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    path = RUNS / 'first_pair_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(value), flush=True)


def verify_frozen():
    manifest = json.loads((RUNS / 'frozen_first_pair/run_manifest.json').read_text())
    for name, expected in manifest['hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen first-pair input/code changed: {name}')
    for task in json.loads((ROOT / 'artifacts/laya_formal_data/manifest.json').read_text())['tasks'].values():
        for split in ('train', 'selection_dev', 'calibration'):
            item = task['files'][split]
            if hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError(f'Data hash mismatch: {item["path"]}')


def check_summary(condition):
    result = json.loads((RUNS / f'{condition}_seed20260922/summary.json').read_text())
    if result['test_access'] or not result['no_cpt'] or not result['checkpoint_reload_logits_match']:
        raise RuntimeError(f'{condition}: test/supervision/reload invariant failed')
    if not result['training']['finite']:
        raise RuntimeError(f'{condition}: finite training invariant failed')
    if condition != 'b0' and (result['training']['updates'] != 3005 or result['training']['examples_consumed'] != 96160):
        raise RuntimeError(f'{condition}: training budget mismatch')
    if result['n_rows'] != {'train':32050, 'selection_dev':1978, 'calibration':1986}:
        raise RuntimeError(f'{condition}: dataset membership counts changed')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait-m1-pid', type=int, required=True)
    a = p.parse_args()
    verify_frozen()
    while Path(f'/proc/{a.wait_m1_pid}').exists():
        write_status({'state':'waiting_for_m1','m1_pid':a.wait_m1_pid,
                      'queue':['m2','b0'],'test_access':False})
        time.sleep(30)
    check_summary('m1')
    for condition in ('m2','b0'):
        verify_frozen()
        output = RUNS / f'{condition}_seed20260922'
        if (output / 'summary.json').exists():
            check_summary(condition)
            continue
        command = [sys.executable, '-u', str(ROOT / 'scripts/laya_formal_experiment.py'),
                   '--condition',condition,'--task','both','--device','cuda',
                   '--epochs','3','--updates','0','--micro-batch','8','--grad-accum','4',
                   '--eval-batch','16','--seed','20260922',
                   '--eligible-ids','artifacts/laya_formal_data/eligible_ids.json',
                   '--output-dir',str(output)]
        with (RUNS / 'logs' / f'{condition}_seed20260922.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                write_status({'state':'running','condition':condition,'pid':process.pid,
                              'command':command,'test_access':False})
                time.sleep(30)
            if process.returncode != 0:
                raise RuntimeError(f'{condition} failed with exit code {process.returncode}; see log')
        check_summary(condition)
    write_status({'state':'complete','completed':['m1','m2','b0'],'test_access':False,
                  'remaining_paper_work':['fixed-head control','additional seeds','locked test','sequence dependence diagnostics']})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_status({'state':'failed','error':str(exc),'test_access':False})
        raise
