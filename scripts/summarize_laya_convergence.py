#!/usr/bin/env python3
"""Verify and plot the 128-to-512 GFP training-only convergence extension."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    status=json.loads((a.root/'status.json').read_text());assert status['status']=='complete'
    for name,h in status['source_sha256'].items():assert hashlib.sha256((a.root/'frozen_code'/name).read_bytes()).hexdigest()==h
    for name,h in status['reference_sha256'].items():assert hashlib.sha256((a.root/'reference_run'/name).read_bytes()).hexdigest()==h
    folder=a.root/'native_readout_512';s=json.loads((folder/'summary.json').read_text());reference=json.loads((a.root/'reference_run/summary.json').read_text())
    assert s['complete'] and s['actual_updates']==512 and s['checkpoint_retained'] and not s['dev_access'] and not s['test_access']
    assert s['reload_max_probability_error']<1e-6 and s['reload_max_scalar_error']<1e-6
    check=json.loads((folder/'prefix_check.json').read_text())
    assert check['updates_compared']==128 and all(check[k] for k in ['configuration_match','exact_optimization_trace_match','exact_prediction_match','exact_final_metrics_match'])
    trace=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];old_trace=[json.loads(x) for x in (a.root/'reference_run/training.jsonl').read_text().splitlines()]
    assert len(trace)==512 and all(np.isfinite(t['loss']) and np.isfinite(t['gradient_norm']) for t in trace)
    for x,y in zip(trace[:128],old_trace):assert all(x[k]==y[k] for k in ['step','loss','lr','gradient_norm','gradient_norms_by_module'])
    curve=[json.loads(x) for x in (folder/'learning_curve.jsonl').read_text().splitlines()];old_curve=[json.loads(x) for x in (a.root/'reference_run/learning_curve.jsonl').read_text().splitlines()]
    std=s['task_info']['fluorescence']['std'];rows=[]
    for x in curve:
        m=x['canonical_panel'];nr=m['scalar_prediction']['rmse']/std
        rows.append({'step':x['step'],'accuracy':m['accuracy'],'nll':m['nll'],**m['scalar_prediction'],'standardized_rmse':nr,
                     'class_fit':m['accuracy']>=31/32 and m['nll']<=.15,'scalar_fit':nr<=.15})
    result={'dev_access':False,'test_access':False,'prefix_check':check,'reference_final':reference['final']['canonical_panel'],
            'final':s['final']['canonical_panel'],'class_fit':s['canonical_panel_fit_passed'],'scalar_fit':s['scalar_fit_passed'],
            'first_observed_class_pass':next((r['step'] for r in rows if r['class_fit']),None),
            'first_observed_scalar_pass':next((r['step'] for r in rows if r['scalar_fit']),None),
            'first_observed_joint_pass':next((r['step'] for r in rows if r['class_fit'] and r['scalar_fit']),None),
            'curve':rows,'rotated_sequences':s['rotated_sequences'],'checkpoint_sha256':s['checkpoint_sha256'],
            'training_and_periodic_eval_seconds':s['training_and_periodic_eval_seconds']}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    steps=[r['step'] for r in rows];old_steps=[r['step'] for r in old_curve]
    fields=[('accuracy','Five-bin accuracy',False),('nll','Categorical NLL',True),('standardized_rmse','Standardized scalar RMSE',True)]
    for ax,(field,title,log) in zip(axes.flat,fields):
        vals=[r[field] for r in rows]
        old_vals=[r['canonical_panel']['scalar_prediction']['rmse']/std if field=='standardized_rmse' else r['canonical_panel'][field] for r in old_curve]
        ax.plot(steps,vals,marker='.',color='#0072B2',label='512-update run')
        ax.scatter(old_steps,old_vals,facecolors='none',edgecolors='#CC79A7',s=45,label='Original 128-update run',zorder=3)
        ax.set(title=title,xlabel='Optimizer updates',ylabel=title)
        if log:ax.set_yscale('log')
        else:ax.set_ylim(-.03,1.03)
        ax.axhline(31/32 if field=='accuracy' else .15,color='gray',ls=':',label='Fit threshold')
        ax.legend(fontsize=8)
    ax=axes[1,1];ax.plot([t['step'] for t in trace],[t['loss'] for t in trace],lw=1,color='#0072B2',label='Joint training loss, dropout active')
    ax.set(title='Optimization trace',xlabel='Optimizer updates',ylabel='0.5 CE + 0.5 standardized MSE',yscale='log');ax.legend(fontsize=8)
    for ax in axes.flat:ax.axvline(128,color='gray',ls='--',lw=1);ax.grid(alpha=.2)
    fig.suptitle('GFP: same 32 fitted training examples; exact replay through update 128; no dev/test')
    a.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output/'learning_curves.png',dpi=180);fig.savefig(a.output/'learning_curves.pdf');plt.close(fig)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
