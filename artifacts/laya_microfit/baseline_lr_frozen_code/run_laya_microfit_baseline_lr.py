#!/usr/bin/env python3
"""Post-hoc matched LR check for the failed GFP shared-head microfit."""
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path


def main():
    root=Path('/root/autodl-tmp/jev_gene/artifacts/laya_multitask_v2');out=root/'microfit_baseline_lr_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir();state={'status':'waiting','created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'dev_access':False,'test_access':False,
        'rationale':'Post-hoc same-LR comparator after candidate GFP succeeded at LR 1e-4 and the baseline failed at 2e-5; no dev selection','jobs':[]}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'status.json')
    save();started=time.monotonic()
    while True:
        status=json.loads((root/'microfit_order_v1/status.json').read_text())['status']
        if status in ['complete','skipped_condition_not_met']:break
        if status not in ['running','waiting'] or time.monotonic()-started>3600:state['status']='prerequisite_failed_or_timeout';save();return 1
        time.sleep(10)
    base=json.loads((root/'microfit_v1/fluorescence_shared_heads_base/summary.json').read_text())
    if base['training_panel_fit_passed']:state['status']='skipped_already_fit';save();return 0
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(used.splitlines()[0])>=500 or shutil.disk_usage(out).free<3*2**30:state['status']='resource_check_failed';save();return 1
    frozen=out/'frozen_code';frozen.mkdir();state['source_sha256']={}
    for name in ['laya_microfit.py','laya_multitask_experiment.py','laya_multitask_data.py','laya_formal_experiment.py','laya_metrics.py','run_laya_microfit_baseline_lr.py']:
        path=Path(__file__).with_name(name);shutil.copy2(path,frozen/name);state['source_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    dest=out/'fluorescence_shared_heads_lr_control'
    cmd=[sys.executable,'-u',str(frozen/'laya_microfit.py'),'--data-dir',str(root/'pilot_data_v2_splice_names'),
        '--model-dir','/root/autodl-tmp/jev_gene/artifacts/laya_model','--laya-repo','/root/autodl-tmp/laya_bio/vendor/laya',
        '--output',str(dest),'--task','fluorescence','--kind','shared_heads','--lr','1e-4','--updates','128']
    state['status']='running';job={'name':dest.name,'status':'running','command':cmd};state['jobs'].append(job);save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    with (out/'training.log').open('w') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);job['pid']=proc.pid;save()
        try:code=proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            code=124
    job['exit_code']=code
    if code:job['status']='failed';state['status']='failed';save();return code
    result=json.loads((dest/'summary.json').read_text())
    assert result['subset_sha256']==base['subset_sha256'] and result['initial']==base['initial'] and result['actual_updates']==128
    job['status']='complete';state['status']='complete';state['identical_initial_metrics_and_subset']=True;save();return 0

if __name__=='__main__':sys.exit(main())
