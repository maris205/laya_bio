#!/usr/bin/env python3
"""Bounded four-run single-task controls with exact joint sampler replay."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import laya_multitask_experiment as exp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data-dir','model-dir','laya-repo','joint-dir','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-hours',type=float,default=1);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if shutil.disk_usage(a.output.parent).free<10*2**30:raise RuntimeError('Need 10 GiB free')
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(used.splitlines()[0])>=500:raise RuntimeError('GPU occupied')
    for kind in ['candidate','shared_heads']:
        d=json.loads((a.joint_dir/f'{kind}_seed20260924'/'summary.json').read_text())
        assert d['complete'] and d['checkpoint_reload_passed'] and d['updates']==576 and not d['test_access']
    a.output.mkdir(parents=True);started=time.monotonic()
    frozen=a.output/'frozen_code';frozen.mkdir()
    script=Path(__file__).with_name('laya_multitask_experiment.py')
    sources={}
    for name in ['laya_multitask_experiment.py','laya_multitask_data.py','laya_formal_experiment.py','laya_metrics.py','run_laya_single_task_diagnostics.py']:
        src=script.with_name(name);shutil.copy2(src,frozen/name);sources[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    state={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
           'test_access':False,'seed':20260924,'reference_joint_updates':576,'updates_per_single_task':96,
           'source_sha256':sources,'max_hours':a.max_hours,'jobs':[]}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();tmp=a.output/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.output/'status.json')
    save();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    # Build expected batches using the same original data ordering, independent of tokenization.
    rows=[json.loads(l) for l in (a.data_dir/'train.jsonl').read_text().splitlines()]
    expected=list(exp.training_schedule(rows,20260924,576,32))
    for task in ['splice','fluorescence']:
        for kind in ['candidate','shared_heads']:
            name=task+'_'+kind;dest=a.output/name;log=a.output/(name+'.log')
            cmd=[sys.executable,'-u',str(frozen/script.name),'--kind',kind,'--data-dir',str(a.data_dir),
                 '--model-dir',str(a.model_dir),'--laya-repo',str(a.laya_repo),'--output',str(dest),
                 '--updates','576','--only-task',task,'--train-diagnostics-cap','128','--seed','20260924']
            job={'task':task,'kind':kind,'output':str(dest),'log':str(log),'command':cmd,'status':'running'}
            state['jobs'].append(job)
            with log.open('w') as f:
                proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);job['pid']=proc.pid;save()
                try:code=proc.wait(timeout=max(1,a.max_hours*3600-(time.monotonic()-started)))
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    job['status']='timeout';state['status']='timeout';save();return 1
            job['exit_code']=code
            if code:job['status']='failed';state['status']='failed';save();return code
            summary=json.loads((dest/'summary.json').read_text())
            assert summary['complete'] and summary['checkpoint_reload_passed'] and summary['entity_presentations']=={task:3072}
            trace=[json.loads(l) for l in (dest/'training.jsonl').read_text().splitlines()]
            ref=[(s,t,b) for s,t,b in expected if t==task];assert len(trace)==len(ref)==96
            for rec,(step,_,batch) in zip(trace,ref):
                factor=step/29 if step<=29 else .5*(1+math.cos(math.pi*(step-29)/(576-29)))
                assert rec['step']==step and rec['entity_batch_sha256']==hashlib.sha256('\n'.join(r['id'] for r in batch).encode()).hexdigest()
                assert abs(rec['lr']-2e-5*factor)<1e-12
            # Same initial predictions verify identical model/head initialization.
            joint=[json.loads(l) for l in (a.joint_dir/f'{kind}_seed20260924'/'dev_before_predictions.jsonl').read_text().splitlines()]
            joint={r['id']:r for r in joint if r['task']==task}
            single=[json.loads(l) for l in (dest/'dev_before_predictions.jsonl').read_text().splitlines()]
            error=max(abs(x-y) for r in single for x,y in zip(r['probs'],joint[r['id']]['probs']))
            assert error<1e-6
            job.update(status='complete',batch_and_lr_replay_passed=True,initial_prediction_max_error=error,training_seconds=summary['training_seconds']);save()
    state['status']='complete';state['total_wall_seconds']=time.monotonic()-started;save();return 0

if __name__=='__main__':sys.exit(main())
