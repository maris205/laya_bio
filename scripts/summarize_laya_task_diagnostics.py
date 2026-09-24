#!/usr/bin/env python3
"""Collect completed diagnostics and verify exposure, data invariants and weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    queue=a.root/'single_task_diagnostics_v2';follow=a.root/'diagnostic_followup_v1'
    for root in [queue,follow]:assert json.loads((root/'status.json').read_text())['status']=='complete'
    report={'test_access':False,'seed':20260924,'comparison_unit':'Same task batches and learning rates; not equal total optimization or compute',
            'script_sha256':digest(Path(__file__)),'runs':{},'corrected_data_invariants':{}}
    for split in ['train','dev']:
        old=[json.loads(l) for l in (a.root/'pilot_data_v1'/(split+'.jsonl')).read_text().splitlines()]
        new=[json.loads(l) for l in (a.root/'pilot_data_v2_splice_names'/(split+'.jsonl')).read_text().splitlines()]
        assert len(old)==len(new);changed=0
        for x,y in zip(old,new):
            differing=[k for k in x if x[k]!=y[k]]
            if x['task']=='splice':assert differing==['choices'];changed+=1
            else:assert not differing
        report['corrected_data_invariants'][split]={'entities':len(old),'changed_entities':changed,'only_changed_field':'splice.choices','passed':True}
    folders=[queue/(t+'_'+k) for t in ['splice','fluorescence'] for k in ['candidate','shared_heads']]+[follow/'corrected_splice_candidate']
    for root in folders:
        summary=json.loads((root/'summary.json').read_text());assert summary['complete'] and summary['checkpoint_reload_passed'] and not summary['test_access']
        assert digest(root/'checkpoint/model.safetensors')==summary['checkpoint_sha256']
        trace=[json.loads(l) for l in (root/'training.jsonl').read_text().splitlines()]
        assert len(trace)==96 and all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in trace)
        assert list(summary['entity_presentations'].values())==[3072]
        pred=[json.loads(l) for l in (root/'dev_after_predictions.jsonl').read_text().splitlines()]
        extra={'predicted_class_counts':np.bincount(np.array([r['probs'] for r in pred]).argmax(1),minlength=len(pred[0]['probs'])).tolist()}
        for key in ['score_prediction','scalar_prediction']:
            if key in pred[0]:extra[key+'_std']=float(np.std([r[key] for r in pred]))
        report['runs'][root.name]={'after':summary['after'],'train_diagnostic':summary['train_diagnostic'],'prediction_distribution':extra,
            'training_seconds':summary['training_seconds'],'peak_allocated_gib':summary['peak_allocated_gib'],'checkpoint_sha256':summary['checkpoint_sha256'],
            'reload_max_probability_error':summary['reload_max_probability_error'],'finite_trace_records':len(trace),'checkpoint_hash_passed':True}
    # A text-only correction arm has identical batches and LR but different candidate names.
    original=[json.loads(l) for l in (queue/'splice_candidate/training.jsonl').read_text().splitlines()]
    corrected=[json.loads(l) for l in (follow/'corrected_splice_candidate/training.jsonl').read_text().splitlines()]
    assert all((x['step'],x['entity_batch_sha256'],x['lr'])==(y['step'],y['entity_batch_sha256'],y['lr']) for x,y in zip(original,corrected))
    report['corrected_name_arm_batch_and_lr_match']=True
    report['joint_train_diagnostics']=json.loads((follow/'joint_train_metrics.json').read_text())['models']
    report['source_status_files']=[str(queue/'status.json'),str(follow/'status.json')]
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
