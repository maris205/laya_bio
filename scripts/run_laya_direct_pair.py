"""Frozen sequential raw/full-BPE matched CE pair; stops on invariant failure."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'artifacts/laya_direct_legacy'
REP=ROOT/'artifacts/laya_direct_bpe_legacy_representation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(**value):
    value.update(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_access=False,no_cpt=True)
    temp=RUNS/'pair_status.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(RUNS/'pair_status.json')
    print(json.dumps(value),flush=True)


def check_summary(path,smoke=False):
    s=json.loads(path.read_text())
    assert s['smoke_only']==smoke and s['formal']!=smoke
    assert not s['test_access'] and s['no_cpt']
    assert s['checkpoint_reload_logits_match'] and s['checkpoint_reload_input_ids_match']
    assert s['training']['finite'] and s['embedding_trainable']
    if not smoke:
        assert s['training']['updates']==3005 and s['training']['examples_consumed']==96160
        assert s['n_rows']=={'train':32050,'selection_dev':1978,'calibration':1986}
    assert all(x['truncated_count']==0 for x in s['item_stats'].values())
    if s['condition']=='full_bpe':
        assert s['expansion']['new_vocab_size']==78367
        assert s['expansion']['maximum_initialization_error']==0
        assert len(s['new_embedding_gradient_probe'])==2
        assert all(x['gradient_norm']>0 for x in s['new_embedding_gradient_probe'].values())
    return s


def main():
    for condition in ('raw','full_bpe'):
        check_summary(RUNS/f'smoke_{condition}/summary.json',smoke=True)
    audit=json.loads((RUNS/'audit/input_audit.json').read_text())
    assert audit['same_samples_labels_candidate_markers'] and audit['full_sequence_reconstruction_checked']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(gpu.strip().splitlines()[0])>=500:raise RuntimeError('GPU is occupied')
    paths=[ROOT/'scripts'/name for name in ('laya_direct_bpe.py','laya_direct_experiment.py',
           'laya_formal_experiment.py','laya_metrics.py','run_laya_direct_pair.py')]
    paths+=list(p for p in REP.rglob('*') if p.is_file())
    paths+=[ROOT/'artifacts/laya_formal_data/eligible_ids.json',RUNS/'audit/input_audit.json']
    paths+=sorted((ROOT/'vendor/laya/laya').glob('*.py'))
    for task in ('promoter_detection','fold_class'):
        paths += [ROOT/f'artifacts/laya_formal_data/{task}_{s}.jsonl' for s in ('train','selection_dev','calibration')]
    snapshot=RUNS/'frozen_pair'
    snapshot.mkdir(exist_ok=True)
    manifest={'seed':20260922,'conditions':['raw','full_bpe'],'updates':3005,'examples_consumed':96160,
              'hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},'no_cpt':True,'test_access':False}
    manifest_path=snapshot/'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest:raise RuntimeError('Frozen manifest changed')
    else:manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    (RUNS/'logs').mkdir(exist_ok=True)
    completed=[]
    for condition in ('raw','full_bpe'):
        for name,expected in manifest['hashes'].items():
            if sha(ROOT/name)!=expected:raise RuntimeError(f'Frozen input changed: {name}')
        output=RUNS/f'{condition}_seed20260922'
        if (output/'summary.json').exists():
            check_summary(output/'summary.json')
            completed.append(condition)
            continue
        command=[sys.executable,'-u',str(ROOT/'scripts/laya_direct_experiment.py'),
                 '--condition',condition,'--output-dir',str(output)]
        with (RUNS/'logs'/f'{condition}_seed20260922.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                status(state='running',condition=condition,pid=process.pid,completed=completed,command=command)
                time.sleep(30)
            if process.returncode!=0:raise RuntimeError(f'{condition} exited {process.returncode}; see log')
        check_summary(output/'summary.json')
        completed.append(condition)
    left=check_summary(RUNS/'raw_seed20260922/summary.json')
    right=check_summary(RUNS/'full_bpe_seed20260922/summary.json')
    assert all(left['item_stats'][s]['membership_sha256']==right['item_stats'][s]['membership_sha256'] for s in left['item_stats'])
    status(state='complete',completed=completed)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(state='failed',error=str(exc))
        raise
