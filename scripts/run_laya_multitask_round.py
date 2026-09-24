#!/usr/bin/env python3
"""Run one bounded, sequential six-task development round on a single GPU."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--laya-repo',type=Path,required=True);p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--max-hours',type=float,default=2);p.add_argument('--seed',type=int,default=20260924)
    a=p.parse_args()
    for kind in ['candidate','shared_heads']:
        path=a.pilot_root/f'pilot_{kind}_seed{a.seed}'/'summary.json';d=json.loads(path.read_text())
        assert d['complete'] and d['checkpoint_reload_passed'] and not d['test_access']
        assert all(v>0 for v in d['type_embedding_update_norms'].values())
    manifest=json.loads((a.data_dir/'manifest.json').read_text());n=manifest['outputs']['train']['entities']
    counts=list(manifest['outputs']['train']['by_task'].values());assert len(set(counts))==1
    updates=a.epochs*n//32;assert updates*32==a.epochs*n and updates%len(counts)==0
    if a.output.exists():raise FileExistsError(a.output)
    if shutil.disk_usage(a.output.parent).free<6*2**30:raise RuntimeError('Less than 6 GiB free for new checkpoints')
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(used.splitlines()[0])>=500:raise RuntimeError('GPU 0 is occupied; refusing to overlap jobs')
    a.output.mkdir(parents=True);started=time.monotonic()
    state={'stage':'six_task_development_round_not_confirmatory','status':'running','pid':os.getpid(),
           'created_utc':datetime.now(timezone.utc).isoformat(),'epochs_equivalent_per_task':a.epochs,
           'updates_per_model':updates,'effective_entity_batch':32,'seed':a.seed,
           'max_hours':a.max_hours,'test_access':False,'jobs':[]}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();temp=a.output/'status.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'status.json')
    save();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    script=Path(__file__).with_name('laya_multitask_experiment.py')
    for kind in ['candidate','shared_heads']:
        dest=a.output/f'{kind}_seed{a.seed}';log=a.output/f'{kind}.log'
        cmd=[sys.executable,'-u',str(script),'--kind',kind,'--data-dir',str(a.data_dir),
             '--model-dir',str(a.model_dir),'--laya-repo',str(a.laya_repo),'--output',str(dest),
             '--updates',str(updates),'--seed',str(a.seed),'--micro-batch','4','--eval-batch','8']
        job={'kind':kind,'command':cmd,'output':str(dest),'log':str(log),'status':'running'}
        state['jobs'].append(job)
        with log.open('w') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
            job['pid']=proc.pid;save();print(json.dumps(job),flush=True)
            try:code=proc.wait(timeout=max(1,a.max_hours*3600-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                job['status']='time_budget_exceeded';state['status']='stopped_at_time_budget';save();return 1
        job['exit_code']=code;job['status']='complete' if code==0 else 'failed'
        if code:
            state['status']='failed';save();return code
        summary=json.loads((dest/'summary.json').read_text())
        if not summary.get('complete') or not summary.get('checkpoint_reload_passed'):
            state['status']='failed_validation';save();return 1
        job['training_seconds']=summary['training_seconds'];save()
    comparison=a.output/'comparison.json'
    cmd=[sys.executable,str(script.with_name('summarize_laya_multitask.py')),'--data-dir',str(a.data_dir),
         '--candidate',str(a.output/f'candidate_seed{a.seed}'),
         '--shared-heads',str(a.output/f'shared_heads_seed{a.seed}'),'--output',str(comparison)]
    code=subprocess.call(cmd,env=env)
    state['status']='complete' if code==0 else 'failed_summary'
    state['comparison']=str(comparison);state['total_wall_seconds']=time.monotonic()-started;save();return code

if __name__=='__main__':sys.exit(main())
