#!/usr/bin/env python3
"""Freeze and execute a single bounded Laya JEV-style decision round."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from laya_biocpt_data import sha,read_lines,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','model-dir','cpt-root']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();smoke=json.loads((a.root/'smoke/status.json').read_text())
    assert smoke['status']=='complete' and smoke['checkpoint_reload_exact'] and smoke['updates']==3
    traces=read_lines(a.root/'smoke/training_trace.jsonl')
    assert {r['task'] for r in traces}=={'promoter','structural_class','fluorescence'}
    assert all(r['encoder_gradient_norm']>0 and r['scorer_gradient_norm']>0 for r in traces)
    assert json.loads((a.root/'data/cpt_admission.json').read_text())['status']=='pass'
    assert shutil.disk_usage(a.root).free>9*2**30
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert int(gpu.strip())<500
    scripts=Path(__file__).resolve().parent
    checks=subprocess.run([sys.executable,str(scripts/'test_laya_jev_multitask.py')],capture_output=True,text=True)
    (a.root/'contract_tests.log').write_text(checks.stdout+checks.stderr)
    assert checks.returncode==0,checks.stderr
    output=a.root/'round';output.mkdir();frozen=output/'frozen_code';frozen.mkdir()
    for name in ['laya_jev_multitask_train.py','laya_jev_multitask_data.py','laya_biocpt_train.py','laya_biocpt_data.py','test_laya_jev_multitask.py','run_laya_jev_multitask.py']:
        shutil.copy2(scripts/name,frozen/name)
    shutil.copy2(scripts.parent/'refine-logs/EXPERIMENT_PLAN.md',output/'EXPERIMENT_PLAN.md')
    jobs=[('cpt_joint','cpt','joint'),('no_cpt_joint','no_cpt','joint'),('cpt_promoter','cpt','promoter'),
          ('cpt_structural_class','cpt','structural_class'),('cpt_fluorescence','cpt','fluorescence')]
    start=datetime.now(timezone.utc);timer=time.monotonic()
    config={'created_utc':start.isoformat(),'code_sha256':{f.name:sha(f) for f in frozen.iterdir()},'data_manifest_sha256':sha(a.root/'data/manifest.json'),
       'initial_model_sha256':sha(a.model_dir/'model.safetensors'),'cpt_checkpoint_sha256':sha(a.cpt_root/'round/cpt/model.safetensors'),
       'run_order':[x[0] for x in jobs],'smoke':smoke,'production_updates':3456,'test_inference':False,
       'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','safetensors','numpy','scipy','scikit-learn']},
       'estimated_minutes_before_dev':[115,150],'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()}
    write_json(output/'manifest.json',config)
    for name,arm,task in jobs:
        command=[sys.executable,'-u',str(frozen/'laya_jev_multitask_train.py'),'--model-dir',str(a.model_dir),'--cpt-root',str(a.cpt_root),
          '--data-dir',str(a.root/'data'),'--output',str(output/name),'--arm',arm,'--task',task]
        if task=='joint':command.append('--retain-model')
        write_json(output/'status.json',{'status':'running','current':name,'command':command,'created_utc':start.isoformat()})
        with (output/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(output/'status.json',{'status':'failed','current':name,'returncode':result.returncode});raise RuntimeError(name+' failed; inspect preserved log')
        state=json.loads((output/name/'status.json').read_text());assert state['status']=='complete' and state['checkpoint_reload_exact']
    write_json(output/'status.json',{'status':'complete','runs':[j[0] for j in jobs],'created_utc':start.isoformat(),
        'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-timer})

if __name__=='__main__':main()
