#!/usr/bin/env python3
"""Fixed 2 x 3 GFP readout/loss diagnostic; training rows only."""
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','model-dir','laya-repo','output','reference']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(used.splitlines()[0])>=500 or shutil.disk_usage(a.output.parent).free<3*2**30:raise RuntimeError('GPU busy or insufficient temporary checkpoint space')
    a.output.mkdir();frozen=a.output/'frozen_code';frozen.mkdir()
    state={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
           'dev_access':False,'test_access':False,'source_sha256':{},'jobs':[],
           'protocol':{'design':'pooling cls/mean x Score objective joint/ce/mse',
                       'sample_count':32,'updates':128,'lr':1e-4,'seed':20260924,
                       'mean_pool':'all attention-valid tokens, including prompt and special tokens; after typed head, before pretrained scorer readout',
                       'joint_loss':'0.5 CE + 0.5 standardized scalar MSE','ce_loss':'unscaled CE','mse_loss':'unscaled standardized scalar MSE',
                       'class_fit':'accuracy >=31/32 and NLL <=0.15; only for CE-supervised arms',
                       'scalar_fit':'standardized RMSE <=0.15; only for MSE-supervised arms',
                       'selection':'no early stopping, no dev/test, final checkpoint metrics',
                       'checkpoint_policy':'save full FP32 state, reload and compare, hash, then remove this run temporary weights to bound disk use; retain configs and predictions',
                       'limits':'post-hoc training diagnostics; single-loss arms also change coefficient from 0.5 to 1; no pure loss-interference attribution'}}
    for name in ['laya_microfit.py','laya_readout_probe.py','laya_multitask_experiment.py','laya_multitask_data.py','laya_formal_experiment.py','laya_metrics.py','run_laya_readout_diagnostics.py']:
        source=Path(__file__).with_name(name);shutil.copy2(source,frozen/name)
        state['source_sha256'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();temp=a.output/'status.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'status.json')
    save();reference=json.loads(a.reference.read_text());start=time.monotonic()
    for pooling,objective in [('cls','joint'),('mean','joint'),('cls','ce'),('mean','ce'),('cls','mse'),('mean','mse')]:
        name=pooling+'_'+objective;dest=a.output/name
        cmd=[sys.executable,'-u',str(frozen/'laya_microfit.py'),'--data-dir',str(a.data_dir),'--model-dir',str(a.model_dir),
             '--laya-repo',str(a.laya_repo),'--output',str(dest),'--task','fluorescence','--kind','shared_heads',
             '--lr','1e-4','--updates','128','--pooling',pooling,'--score-loss',objective,'--probe-readout','--discard-checkpoint-after-verification']
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
        result=json.loads((dest/'summary.json').read_text())
        assert result['subset_sha256']==reference['subset_sha256'] and result['actual_updates']==128
        assert result['selected_ids']==reference['selected_ids'] and result['task_info']==reference['task_info']
        job.update(status='complete',class_fit=result['training_panel_fit_passed'],scalar_fit=result['scalar_fit_passed'],seconds=result['training_and_periodic_eval_seconds'])
        if pooling=='cls':assert result['initial']['canonical_panel']==reference['initial']['canonical_panel']
        if name=='cls_joint':job['historical_final_metrics_identical']=result['final']['canonical_panel']==reference['final']['canonical_panel']
        save()
    state['status']='complete';state['total_wall_seconds']=time.monotonic()-start;save();return 0

if __name__=='__main__':sys.exit(main())
