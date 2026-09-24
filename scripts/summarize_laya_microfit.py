#!/usr/bin/env python3
"""Validate completed training-only microfits and plot their learning curves."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import laya_microfit as micro


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main_root=a.root/'microfit_v1';order_root=a.root/'microfit_order_v1'
    status=json.loads((main_root/'status.json').read_text());order=json.loads((order_root/'status.json').read_text())
    assert status['status']=='complete' and order['status'] in ['complete','skipped_condition_not_met']
    folders=[Path(j['output']) for j in status['jobs']]
    if order['status']=='complete':folders.append(order_root/'splice_candidate_resampled')
    extra_root=a.root/'microfit_baseline_lr_v1'
    if (extra_root/'status.json').exists():
        extra_status=json.loads((extra_root/'status.json').read_text())
        assert extra_status['status'] in ['complete','skipped_already_fit']
        if extra_status['status']=='complete':folders.append(extra_root/'fluorescence_shared_heads_lr_control')
    result={'dev_access':False,'test_access':False,'runs':{},'checkpoint_and_finite_trace_checks':True}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    styles={'candidate_base':('Candidate, fixed order','#0072B2'), 'shared_heads_base':('Shared heads','#D55E00'),
            'candidate_lr_rescue':('Candidate, LR 1e-4','#009E73'), 'candidate_resampled':('Candidate, resampled order','#CC79A7'), 'shared_heads_lr_control':('Shared heads, LR 1e-4','#E69F00')}
    subset={}
    for folder in folders:
        s=json.loads((folder/'summary.json').read_text());assert s['complete'] and s['actual_updates']==128 and not s['dev_access'] and not s['test_access']
        assert micro.sha_file(folder/'checkpoint/model.safetensors')==s['checkpoint_sha256']
        trace=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()]
        assert len(trace)==128 and all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in trace)
        assert s['reload_max_probability_error']<1e-6 and s['reload_max_scalar_error']<1e-6
        subset.setdefault(s['task'],s['subset_sha256']);assert subset[s['task']]==s['subset_sha256']
        curve=[json.loads(l) for l in (folder/'learning_curve.jsonl').read_text().splitlines()]
        passed=[r['step'] for r in curve if micro.fit_passed(r['training_panel'])]
        result['runs'][folder.name]={'task':s['task'],'kind':s['kind'],'lr':s['lr'],'final':s['final'],
             'first_observed_training_panel_pass_step':min(passed) if passed else None,
             'final_training_panel_fit_passed':s['training_panel_fit_passed'],'final_canonical_panel_fit_passed':s['canonical_panel_fit_passed'],
             'rotated_sequences':s['rotated_sequences'],'training_and_periodic_eval_seconds':s['training_and_periodic_eval_seconds'],
             'checkpoint_sha256':s['checkpoint_sha256'],'gradient_probes':[r for r in trace if r['gradient_norms_by_module']]}
        row=0 if s['task']=='splice' else 1;key=folder.name.removeprefix(s['task']+'_');label,color=styles[key]
        steps=[r['step'] for r in curve]
        axes[row,0].plot(steps,[r['training_panel']['accuracy'] for r in curve],marker='.',color=color,label=label)
        axes[row,1].plot(steps,[max(r['training_panel']['nll'],1e-8) for r in curve],marker='.',color=color,label=label)
        if s['task']=='splice' and s['kind']=='candidate':
            axes[row,0].plot(steps,[r['canonical_panel']['accuracy'] for r in curve],ls='--',color=color,alpha=.8,label=label+' (canonical)')
    for row,title in enumerate(['Splice / Choice','GFP / Score']):
        axes[row,0].set(title=title+' — training accuracy',ylim=(-.03,1.03),xlabel='Optimizer updates',ylabel='Accuracy on 32 fitted examples')
        axes[row,0].axhline(31/32,color='gray',ls=':',lw=1)
        axes[row,1].set(title=title+' — training NLL',xlabel='Optimizer updates',ylabel='Categorical NLL',yscale='log')
        axes[row,1].axhline(.15,color='gray',ls=':',lw=1)
        for ax in axes[row]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Training-only memorization checks; no dev/test evaluation',fontsize=12)
    a.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output/'learning_curves.png',dpi=180);fig.savefig(a.output/'learning_curves.pdf');plt.close(fig)
    orders=order_root/'all_order_evaluation.json'
    if orders.exists():result['all_candidate_orders']=json.loads(orders.read_text())
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
