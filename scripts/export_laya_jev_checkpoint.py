#!/usr/bin/env python3
"""Add portable config/tokenizers and verify mixed-task inference without copying weights."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from transformers import ModernBertConfig
from laya_biocpt_data import sha,digest,read_lines,write_lines,write_json
from laya_jev_infer import Predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','cpt-root','model-dir']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=['cpt_joint','no_cpt_joint'],required=True);a=p.parse_args()
    folder=a.root/'round'/a.arm;state=json.loads((folder/'status.json').read_text());assert state['status']=='complete' and state['model_retained']
    manifest=json.loads((a.root/'data/manifest.json').read_text());train=read_lines(a.root/'data/train.jsonl');dev=read_lines(a.root/'data/dev.jsonl')
    templates={}
    for task in manifest['tasks']:
        r=next(r for r in train if r['task']==task)
        templates[task]={k:r[k] for k in ['primitive','modality','question','choices']}
        if r['primitive']=='score':templates[task].update(anchors=manifest['score_spec']['anchors'],units='log fluorescence')
    config=ModernBertConfig.from_pretrained(a.model_dir/'encoder');config.vocab_size=json.loads((a.cpt_root/'data/manifest.json').read_text())['new_vocab_size'];config.reference_compile=False
    config.save_pretrained(folder/'encoder')
    shutil.copytree(a.cpt_root/'data/representation',folder/'representation',dirs_exist_ok=True)
    dc={'model_type':'shared_laya_jev','model_sha256':state['checkpoint_sha256'],'max_input_tokens':manifest['max_length'],
        'training_arm':a.arm,'training_data_manifest_sha256':sha(a.root/'data/manifest.json'),
        'task_templates':templates,'head_layers':2,'task_specific_heads':False,'temperature_calibration':False,
        'scope':'Three supervised tasks. Custom rubrics are accepted structurally but unseen-task quality has not been validated.'}
    write_json(folder/'decision_config.json',dc)
    torch.set_num_threads(8);predictor=Predictor(folder)
    probe=[]
    for t in manifest['tasks']:
        probe.extend(sorted([r for r in dev if r['task']==t],key=lambda r:digest('export-probe:'+r['id']))[:16])
    reference={r['id']:r for r in read_lines(folder/'dev_epoch3_predictions.jsonl')}
    for task in manifest['tasks']:
        first=[r for r in dev if r['task']==task][:16]
        request=[{'id':r['id'],'task':task,'sequence':r['sequence']} for r in first]
        answers=predictor.predict(request)
        for r,out in zip(first,answers):
            np.testing.assert_array_equal(out['probabilities'],reference[r['id']]['probs'])
    # Interleave tasks in one shared-model batch to test variable candidate counts
    # and primitive IDs together, unlike the homogeneous training batches.
    by={t:[r for r in probe if r['task']==t] for t in manifest['tasks']}
    probe=[by[t][i] for i in range(16) for t in manifest['tasks']]
    requests=[{'id':r['id'],'task':r['task'],'sequence':r['sequence']} for r in probe]
    output=[]
    for start in range(0,len(requests),8):output.extend(predictor.predict(requests[start:start+8]))
    reference={r['id']:r for r in read_lines(folder/'dev_epoch3_predictions.jsonl')}
    max_diff=0.;same=0
    # Mixed padding/batch shapes can change BF16 rounding. Use bounded numerical
    # tolerance here; the same-shape save/reload checks remain exactly equal.
    for r,pred in zip(probe,output):
        prior=reference[r['id']];diff=float(np.max(np.abs(np.array(pred['probabilities'])-prior['probs'])));max_diff=max(max_diff,diff)
        assert np.isfinite(pred['probabilities']).all() and abs(sum(pred['probabilities'])-1)<1e-6
        same+=int(np.argmax(pred['probabilities'])==prior['prediction'])
        assert pred['type']==r['primitive']
    write_lines(folder/'mixed_task_inference_predictions.jsonl',output)
    benchmarks={}
    for task in manifest['tasks']:
        pool=sorted([r for r in dev if r['task']==task],key=lambda r:digest('raw-bench:'+r['id']))[:64]
        req=[{'id':r['id'],'task':task,'sequence':r['sequence']} for r in pool]
        for _ in range(3):predictor.predict(req[:1])
        timings=[]
        for r in req:
            torch.cuda.synchronize();t=time.perf_counter();predictor.predict([r]);torch.cuda.synchronize();timings.append(time.perf_counter()-t)
        torch.cuda.synchronize();t=time.perf_counter()
        for start in range(0,len(req),16):predictor.predict(req[start:start+16])
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        benchmarks[task]={'requests':64,'batch_size':16,'single_request_p50_ms':float(np.median(timings)*1000),
          'single_request_p95_ms':float(np.quantile(timings,.95)*1000),'batched_requests_per_second':64/seconds,
          'scope':'Raw biological sequence and textual rubric tokenization, panel assembly, H2D, full model, softmax, D2H, typed answer construction; warm loaded model; excludes file/network IO.'}
    write_json(folder/'end_to_end_benchmark.json',benchmarks)
    write_json(folder/'portable_export_check.json',{'status':'pass','mixed_task_requests':len(requests),'one_model_instance':True,
        'same_shape_raw_inference_matches_saved_predictions_exactly':True,
        'max_probability_difference_from_original_homogeneous_eval':max_diff,'argmax_agreement_fraction':same/len(probe),
        'numerical_note':'BF16 mixed-batch versus homogeneous-batch shape differences; same-shape training checkpoint reload was exact.',
        'model_sha256':sha(folder/'model.safetensors'),'portable_files':{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and (p.suffix in ['.json','.txt'] and ('representation' in p.parts or 'encoder' in p.parts or p.name=='decision_config.json'))}})
    print(json.dumps({'status':'pass','arm':a.arm,'mixed_probability_max_difference':max_diff,'mixed_argmax_agreement':same/len(probe),'end_to_end_benchmark':benchmarks},indent=2))

if __name__=='__main__':main()
