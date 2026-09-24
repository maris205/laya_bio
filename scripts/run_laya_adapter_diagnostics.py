#!/usr/bin/env python3
"""Conditional GFP adapter ablations after the pooling/loss checks fail."""
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data-dir','model-dir','laya-repo','output','prerequisite']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir();frozen=a.output/'frozen_code';frozen.mkdir()
    state={'status':'waiting','created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'dev_access':False,'test_access':False,'source_sha256':{},'jobs':[],
        'protocol':{'condition':'All six original pooling/loss arms fail their supervised fitting thresholds',
                    'design':'mean pooling / joint loss fixed; bypass typed transformer head only, replace pretrained readout only, or both',
                    'native_readout':'fresh LayerNorm(1024) followed by identically initialized task output layers; retains original type embedding',
                    'additional_condition':'If both-adapters-removed joint arm still fails scalar fit, run one native-readout/bypassed-head MSE-only arm',
                    'budget':'3 runs plus at most 1 conditional MSE run; 128 updates, LR 1e-4, same 32 training examples',
                    'interpretation':'Post-hoc diagnostics, not generalization or matched-parameter architecture ranking',
                    'checkpoint_policy':'verified temporary FP32 save/reload and hash, then remove new weights only'}}
    for name in ['laya_microfit.py','laya_readout_probe.py','laya_multitask_experiment.py','laya_multitask_data.py','laya_formal_experiment.py','laya_metrics.py','run_laya_adapter_diagnostics.py']:
        source=Path(__file__).with_name(name);shutil.copy2(source,frozen/name);state['source_sha256'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();temp=a.output/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'status.json')
    save();start=time.monotonic()
    while True:
        prerequisite=json.loads((a.prerequisite/'status.json').read_text())
        if prerequisite['status']=='complete':break
        if prerequisite['status']!='running' or time.monotonic()-start>3600:state['status']='prerequisite_failed_or_timeout';save();return 1
        time.sleep(10)
    summaries=[json.loads((a.prerequisite/j['name']/'summary.json').read_text()) for j in prerequisite['jobs']]
    if any(s['training_panel_fit_passed'] or s['scalar_fit_passed'] for s in summaries):state['status']='skipped_condition_not_met';save();return 0
    state['status']='running';save();reference=summaries[0]
    jobs=[('bypass_head',True,False,'joint'),('native_readout',False,True,'joint'),('native_both',True,True,'joint')]
    for name,bypass,native,objective in jobs:
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(used.splitlines()[0])>=500 or shutil.disk_usage(a.output).free<3*2**30:state['status']='resource_check_failed';save();return 1
        dest=a.output/name
        cmd=[sys.executable,'-u',str(frozen/'laya_microfit.py'),'--data-dir',str(a.data_dir),'--model-dir',str(a.model_dir),'--laya-repo',str(a.laya_repo),
             '--output',str(dest),'--task','fluorescence','--kind','shared_heads','--lr','1e-4','--updates','128','--pooling','mean','--score-loss',objective,
             '--probe-readout','--discard-checkpoint-after-verification']
        if bypass:cmd.append('--bypass-shared-head')
        if native:cmd.append('--native-readout')
        job={'name':name,'status':'running','command':cmd};state['jobs'].append(job);save()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
        with (a.output/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);job['pid']=proc.pid;save()
            try:code=proc.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                code=124
        job['exit_code']=code
        if code:job['status']='failed';state['status']='failed';save();return code
        s=json.loads((dest/'summary.json').read_text())
        assert s['subset_sha256']==reference['subset_sha256'] and s['selected_ids']==reference['selected_ids'] and s['actual_updates']==128
        job.update(status='complete',class_fit=s['training_panel_fit_passed'],scalar_fit=s['scalar_fit_passed'],seconds=s['training_and_periodic_eval_seconds'])
        if name=='native_both' and not s['scalar_fit_passed']:jobs.append(('native_both_mse',True,True,'mse'))
        save()
    state['status']='complete';state['total_wall_seconds_including_wait']=time.monotonic()-start;save();return 0

if __name__=='__main__':sys.exit(main())
