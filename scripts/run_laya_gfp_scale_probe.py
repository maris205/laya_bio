#!/usr/bin/env python3
"""Freeze and run the three predeclared training-only scale/LR arms."""
from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from laya_gfp_development import sha_file, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir', 'model-dir', 'laya-repo', 'output', 'reference', 'provenance']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(used.splitlines()[0]) >= 500 or shutil.disk_usage(a.output.parent).free < 7 * 2**30:
        raise RuntimeError('GPU busy or insufficient room for three retained checkpoints')
    provenance = json.loads(a.provenance.read_text())
    assert platform.python_version() == provenance['python']
    for name, version in provenance['packages'].items():
        assert importlib.metadata.version(name) == version
    for name, expected in provenance['initial_model_and_tokenizer_sha256'].items():
        assert sha_file(Path(name)) == expected, name
    a.output.mkdir()
    frozen = a.output / 'frozen_code'
    frozen.mkdir()
    source_hashes = {}
    for name in ['laya_gfp_scale_probe.py', 'run_laya_gfp_scale_probe.py', 'laya_gfp_development.py', 'laya_multitask_experiment.py', 'laya_multitask_data.py', 'laya_formal_experiment.py', 'laya_metrics.py']:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, frozen / name)
        source_hashes[name] = sha_file(source)
    reference = a.output / 'reference_training_only'
    reference.mkdir()
    reference_hashes = {}
    for source in [a.reference / 'training.jsonl', *sorted(a.reference.glob('train_step*_predictions.jsonl'))]:
        shutil.copy2(source, reference / source.name)
        reference_hashes[source.name] = sha_file(source)
    shutil.copy2(a.provenance, a.output / 'initialization_provenance.json')
    arms = [('n32_high', 32, 1., 512), ('n32_low', 32, .2, 512), ('n1024_low', 1024, .2, 1024)]
    state = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'pid': os.getpid(),
             'dev_access': False, 'test_access': False, 'source_sha256': source_hashes, 'reference_sha256': reference_hashes,
             'protocol': {'arms': arms, 'reference': 'completed N1024/high LR, training outputs only', 'primary': 'full selected-training scalar normalized RMSE',
                          'secondary': ['training Spearman', 'training prediction std', 'five-bin accuracy/NLL', 'shared anchor32 metrics'],
                          'comparison_budget': '2x2 at step512; N1024/low continuation at768,1024',
                          'equal_exposure_readouts': '32 samples at step16 versus 1024 at512; atstep32 versus1024at1024; LR history/update count differs',
                          'fit_reference': 'normalized scalar RMSE <=0.15; categorical accuracy>=31/32 and NLL<=0.15 on N32 only',
                          'run_policy': 'complete all arms, no adaptive early stopping or extra arms',
                          'checkpoint': 'retain final model-only weights after exact reload check'}, 'runs': []}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(a.output / 'status.json', state)
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    start = time.monotonic()
    for name, count, scale, updates in arms:
        cmd = [sys.executable, '-u', str(frozen / 'laya_gfp_scale_probe.py'), '--data-dir', str(a.data_dir), '--model-dir', str(a.model_dir),
               '--laya-repo', str(a.laya_repo), '--output', str(a.output / name), '--reference', str(reference),
               '--samples', str(count), '--lr-scale', str(scale), '--updates', str(updates)]
        run = {'name': name, 'command': cmd, 'status': 'running'}
        state['runs'].append(run)
        save()
        with (a.output / (name + '.log')).open('w') as f:
            worker = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            run['pid'] = worker.pid
            save()
            try:
                code = worker.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
                code = 124
        run['exit_code'] = code
        run['status'] = 'complete' if code == 0 else 'failed'
        if code:
            state['status'] = 'failed'
            save()
            return code
        result = json.loads((a.output / name / 'summary.json').read_text())
        assert result['complete'] and result['actual_updates'] == updates
        save()
    state.update(status='complete', total_wall_seconds=time.monotonic() - start)
    save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
