#!/usr/bin/env python3
"""Run four fixed microfits and at most two predeclared LR-only rescue controls."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data-dir','model-dir','laya-repo','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-hours',type=float,default=1);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if shutil.disk_usage(a.output.parent).free<13*2**30:raise RuntimeError('Need 13 GiB for up to six checkpoints and margin')
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(used.splitlines()[0])>=500:raise RuntimeError('GPU occupied')
    a.output.mkdir(parents=True);frozen=a.output/'frozen_code';frozen.mkdir();sources={}
    for name in ['laya_microfit.py','laya_multitask_experiment.py','laya_multitask_data.py','laya_formal_experiment.py','laya_metrics.py','run_laya_microfit.py']:
        path=Path(__file__).with_name(name);shutil.copy2(path,frozen/name);sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    state={'status':'running','stage':'32-example_training_memorization_not_generalization','created_utc':datetime.now(timezone.utc).isoformat(),
           'pid':os.getpid(),'dev_access':False,'test_access':False,'max_hours':a.max_hours,'source_sha256':sources,
           'protocol':{'initial_jobs':'splice/fluorescence x candidate/shared_heads, 128 updates, LR 2e-5',
                       'rescue_rule':'For each candidate run failing training-panel accuracy >=31/32 and NLL <=0.15, one fresh-init run with LR 1e-4 only changed; same 128 updates',
                       'maximum_training_jobs':6,'early_stopping':False},'jobs':[]}
    started=time.monotonic()
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();temp=a.output/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'status.json')
    save();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    def run(task,kind,lr,phase):
        name=f'{task}_{kind}_{phase}';dest=a.output/name;log=a.output/(name+'.log')
        cmd=[sys.executable,'-u',str(frozen/'laya_microfit.py'),'--data-dir',str(a.data_dir),'--model-dir',str(a.model_dir),
             '--laya-repo',str(a.laya_repo),'--output',str(dest),'--task',task,'--kind',kind,'--lr',str(lr),'--updates','128']
        job={'name':name,'task':task,'kind':kind,'lr':lr,'phase':phase,'status':'running','output':str(dest),'command':cmd}
        state['jobs'].append(job);save()
        with log.open('w') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);job['pid']=proc.pid;save()
            try:code=proc.wait(timeout=max(1,a.max_hours*3600-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                code=124
        job['exit_code']=code
        if code:job['status']='failed';state['status']='failed';save();raise RuntimeError('Microfit child failed: '+name)
        result=json.loads((dest/'summary.json').read_text())
        assert result['complete'] and result['actual_updates']==128 and not result['dev_access'] and not result['test_access']
        job.update(status='complete',training_panel_fit_passed=result['training_panel_fit_passed'],canonical_panel_fit_passed=result['canonical_panel_fit_passed'],
                   seconds=result['training_and_periodic_eval_seconds'],subset_sha256=result['subset_sha256']);save()
        return result
    initial={}
    for task in ['splice','fluorescence']:
        for kind in ['candidate','shared_heads']:initial[(task,kind)]=run(task,kind,2e-5,'base')
    for task in ['splice','fluorescence']:
        if not initial[(task,'candidate')]['training_panel_fit_passed']:
            rescue=run(task,'candidate',1e-4,'lr_rescue')
            assert rescue['subset_sha256']==initial[(task,'candidate')]['subset_sha256']
    state['status']='complete';state['total_wall_seconds']=time.monotonic()-started;save();return 0

if __name__=='__main__':sys.exit(main())
