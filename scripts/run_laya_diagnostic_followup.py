#!/usr/bin/env python3
"""Wait for the bounded single-task queue, then run the text-only correction arm."""
from datetime import datetime,timezone
import json,os,subprocess,sys,time,hashlib,shutil
from pathlib import Path


def main():
    base=Path('/root/autodl-tmp/jev_gene/artifacts/laya_multitask_v2');queue=base/'single_task_diagnostics_v2'
    output=base/'diagnostic_followup_v1'
    if output.exists():raise FileExistsError(output)
    output.mkdir();started=time.monotonic();state={'status':'waiting_for_single_task_queue','test_access':False,'jobs':[],
        'created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();p=output/'status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(output/'status.json')
    save()
    while True:
        status=json.loads((queue/'status.json').read_text())['status']
        if status=='complete':break
        if status!='running' or time.monotonic()-started>3600:state['status']='prerequisite_failed_or_timeout';save();return 1
        time.sleep(10)
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(used.splitlines()[0])>=500:state['status']='gpu_occupied';save();return 1
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    model='/root/autodl-tmp/jev_gene/artifacts/laya_model';repo='/root/autodl-tmp/laya_bio/vendor/laya'
    evaluator=Path(__file__).with_name('evaluate_laya_joint_train_diagnostics.py')
    snapshot=queue/'frozen_code'/evaluator.name;shutil.copy2(evaluator,snapshot)
    state['evaluator_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    commands=[('corrected_splice_candidate',[sys.executable,'-u',str(queue/'frozen_code/laya_multitask_experiment.py'),
        '--kind','candidate','--data-dir',str(base/'pilot_data_v2_splice_names'),'--model-dir',model,'--laya-repo',repo,
        '--output',str(output/'corrected_splice_candidate'),'--updates','576','--only-task','splice','--train-diagnostics-cap','128','--seed','20260924']),
        ('joint_train_evaluation',[sys.executable,'-u',str(snapshot),'--data-dir',str(base/'pilot_data_v1'),'--model-dir',model,
        '--laya-repo',repo,'--joint-dir',str(base/'development_round1'),'--output',str(output/'joint_train_metrics.json')])]
    for name,cmd in commands:
        job={'name':name,'status':'running','command':cmd};state['jobs'].append(job);state['status']='running';save()
        with (output/(name+'.log')).open('w') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);job['pid']=proc.pid;save()
            try:code=proc.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                code=124
        job['exit_code']=code;job['status']='complete' if code==0 else 'failed';save()
        if code:state['status']='failed';save();return code
    state['status']='complete';state['total_wall_seconds_including_wait']=time.monotonic()-started;save();return 0

if __name__=='__main__':sys.exit(main())
