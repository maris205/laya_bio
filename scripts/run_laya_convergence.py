#!/usr/bin/env python3
"""Extend the GFP native-readout microfit to 512 updates with exact prefix replay."""
from datetime import datetime,timezone
import argparse,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data-dir','model-dir','laya-repo','output','reference-run','provenance']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(used.splitlines()[0])>=500 or shutil.disk_usage(a.output.parent).free<3*2**30:raise RuntimeError('GPU busy or insufficient checkpoint space')
    a.output.mkdir();frozen=a.output/'frozen_code';frozen.mkdir();reference=a.output/'reference_run';reference.mkdir()
    state={'status':'preparing','created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'dev_access':False,'test_access':False,
           'source_sha256':{},'reference_sha256':{},'reference_original':str(a.reference_run),
           'protocol':{'training_change':'Only update budget increases from 128 to 512; optimizer trajectory must replay exactly through step 128',
                       'pooling':'mean over all attended tokens','readout':'fresh LayerNorm; retain typed transformer head and type embedding',
                       'loss':'0.5 ordinal CE + 0.5 standardized scalar MSE','lr':1e-4,'warmup_updates':8,'lr_after_warmup':'constant',
                       'examples':32,'seed':20260924,'effective_batch':32,'micro_batch':4,'updates':512,'early_stopping':False,
                       'evaluation':'historical early steps and every 64 updates; same training examples only',
                       'failure_policy':'Abort at step 128 if configuration, optimization trace, metrics or predictions fail exact replay',
                       'checkpoint_policy':'Retain final FP32 weights locally after verified save/reload; no weight upload',
                       'limitations':'Configuration selected after earlier training diagnostics; one seed; no dev/test generalization or equal-budget candidate comparison'}}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();tmp=a.output/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.output/'status.json')
    save()
    provenance=json.loads(a.provenance.read_text())
    assert platform.python_version()==provenance['python']
    for name,version in provenance['packages'].items():assert importlib.metadata.version(name)==version,(name,version)
    for name,expected in provenance['initial_model_and_tokenizer_sha256'].items():assert sha_file(Path(name))==expected,name
    state['initialization_and_environment_hashes_verified']=True
    shutil.copy2(a.provenance,a.output/'provenance.json')
    for name in ['summary.json','training.jsonl','canonical_panel_predictions.jsonl','learning_curve.jsonl']:
        source=a.reference_run/name;shutil.copy2(source,reference/name);state['reference_sha256'][name]=sha_file(source)
    for name in ['laya_microfit.py','laya_readout_probe.py','laya_multitask_experiment.py','laya_multitask_data.py','laya_formal_experiment.py','laya_metrics.py','run_laya_convergence.py']:
        source=Path(__file__).with_name(name);shutil.copy2(source,frozen/name);state['source_sha256'][name]=sha_file(source)
    dest=a.output/'native_readout_512'
    cmd=[sys.executable,'-u',str(frozen/'laya_microfit.py'),'--data-dir',str(a.data_dir),'--model-dir',str(a.model_dir),'--laya-repo',str(a.laya_repo),
         '--output',str(dest),'--task','fluorescence','--kind','shared_heads','--lr','1e-4','--updates','512','--pooling','mean','--score-loss','joint',
         '--native-readout','--probe-readout','--eval-every','64','--reference-run',str(reference)]
    state['status']='running';state['command']=cmd;save();start=time.monotonic()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    with (a.output/'training.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);state['worker_pid']=proc.pid;save()
        try:code=proc.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            code=124
    state['exit_code']=code
    if code:state['status']='failed';save();return code
    result=json.loads((dest/'summary.json').read_text());prefix=json.loads((dest/'prefix_check.json').read_text())
    assert result['actual_updates']==512 and result['checkpoint_retained'] and prefix['exact_optimization_trace_match'] and prefix['exact_prediction_match']
    assert sha_file(dest/'checkpoint/model.safetensors')==result['checkpoint_sha256']
    state.update(status='complete',total_worker_wall_seconds=time.monotonic()-start,class_fit=result['training_panel_fit_passed'],scalar_fit=result['scalar_fit_passed'],prefix_check=prefix)
    save();return 0

if __name__=='__main__':sys.exit(main())
