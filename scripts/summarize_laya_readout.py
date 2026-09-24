#!/usr/bin/env python3
"""Summarize training-only 2 x 3 GFP readout/loss controls."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter-root',type=Path);a=p.parse_args()
    status=json.loads((a.root/'status.json').read_text());assert status['status']=='complete'
    for name,h in status['source_sha256'].items():assert hashlib.sha256((a.root/'frozen_code'/name).read_bytes()).hexdigest()==h
    jobs=[(a.root,j) for j in status['jobs']]
    if a.adapter_root:
        adapter_status=json.loads((a.adapter_root/'status.json').read_text());assert adapter_status['status'] in ['complete','skipped_condition_not_met']
        for name,h in adapter_status['source_sha256'].items():assert hashlib.sha256((a.adapter_root/'frozen_code'/name).read_bytes()).hexdigest()==h
        jobs.extend((a.adapter_root,j) for j in adapter_status['jobs'])
    results={'dev_access':False,'test_access':False,'runs':{},'scope':'same 32 training examples; not generalization'}
    fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    colors={'joint':'#0072B2','ce':'#D55E00','mse':'#009E73'};subset=None
    for parent,job in jobs:
        root=parent/job['name'];s=json.loads((root/'summary.json').read_text())
        assert s['complete'] and s['actual_updates']==128 and not s['dev_access'] and not s['test_access']
        assert s['reload_max_probability_error']<1e-6 and s['reload_max_scalar_error']<1e-6
        assert not s['checkpoint_retained'] and not (root/'checkpoint/model.safetensors').exists()
        subset=subset or s['subset_sha256'];assert subset==s['subset_sha256']
        trace=[json.loads(l) for l in (root/'training.jsonl').read_text().splitlines()]
        assert len(trace)==128 and all(np.isfinite(t['loss']) and np.isfinite(t['gradient_norm']) for t in trace)
        curve=[json.loads(l) for l in (root/'learning_curve.jsonl').read_text().splitlines()]
        m=s['final']['canonical_panel'];std=s['task_info']['fluorescence']['std']
        results['runs'][job['name']]={'pooling':s['pooling'],'loss':s['score_loss'],'bypass_head':s.get('bypass_shared_head',False),'native_readout':s.get('native_readout',False),'classification_supervised':s['classification_supervised'],
             'scalar_supervised':s['scalar_supervised'],'class_fit':s['training_panel_fit_passed'],'scalar_fit':s['scalar_fit_passed'],
             'final':m,'standardized_rmse':m['scalar_prediction']['rmse']/std if s['scalar_supervised'] else None,
             'readout_initial':s['initial']['readout_probe'],'readout_final':s['final']['readout_probe'],
             'rotated_sequences':s['rotated_sequences'],'seconds':s['training_and_periodic_eval_seconds'],'checkpoint_sha256':s['checkpoint_sha256']}
        steps=[x['step'] for x in curve];kw={'color':colors[s['score_loss']],'ls':'-' if s['pooling']=='cls' else '--','label':job['name'],'marker':'.'}
        if job['name'] in ['bypass_head','native_readout','native_both','native_both_mse']:
            kw.update(color={'bypass_head':'#CC79A7','native_readout':'#E69F00','native_both':'#000000','native_both_mse':'#56B4E9'}[job['name']],ls='-')
        if s['classification_supervised']:
            axes[0].plot(steps,[x['canonical_panel']['accuracy'] for x in curve],**kw)
            axes[1].plot(steps,[max(x['canonical_panel']['nll'],1e-8) for x in curve],**kw)
        if s['scalar_supervised']:axes[2].plot(steps,[x['canonical_panel']['scalar_prediction']['rmse']/std for x in curve],**kw)
    for ax in axes:ax.grid(alpha=.2);ax.set_xlabel('Optimizer updates');ax.legend(fontsize=8)
    axes[0].set(title='Supervised five-bin classification',ylabel='Accuracy',ylim=(-.03,1.03));axes[0].axhline(31/32,color='gray',ls=':')
    axes[1].set(title='Supervised five-bin classification',ylabel='NLL',yscale='log');axes[1].axhline(.15,color='gray',ls=':')
    axes[2].set(title='Supervised scalar regression',ylabel='Standardized RMSE',yscale='log');axes[2].axhline(.15,color='gray',ls=':')
    fig.suptitle('GFP readout/loss diagnostics: 32 fitted training examples, no dev/test')
    a.output.mkdir(parents=True,exist_ok=True);fig.savefig(a.output/'learning_curves.png',dpi=180);fig.savefig(a.output/'learning_curves.pdf');plt.close(fig)
    (a.output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:{f:v[f] for f in ['class_fit','scalar_fit','standardized_rmse','seconds']} for k,v in results['runs'].items()},indent=2))

if __name__=='__main__':main()
