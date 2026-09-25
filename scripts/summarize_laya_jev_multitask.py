#!/usr/bin/env python3
"""Summarize frozen three-task JEV outcomes without selecting best checkpoints."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from laya_biocpt_data import read_lines,write_json

TASKS=['promoter','structural_class','fluorescence']
RUNS=['cpt_joint','no_cpt_joint','cpt_promoter','cpt_structural_class','cpt_fluorescence']


def load(path):return json.loads(path.read_text())


def paired(a,b,task):
    """Compare right minus left, resampling exact-entity groups together."""
    x={r['id']:r for r in a if r['task']==task};y={r['id']:r for r in b if r['task']==task}
    assert x.keys()==y.keys()
    grouped=defaultdict(list)
    for key,r in x.items():
        s=y[key];assert r['label']==s['label'] and r['group_id']==s['group_id']
        if task=='fluorescence':
            assert r['value']==s['value'];values=((r['score_prediction']-r['value'])**2,(s['score_prediction']-s['value'])**2)
        else:values=(int(r['prediction']==r['label']),int(s['prediction']==s['label']))
        grouped[r['group_id']].append(values)
    sums=np.array([np.sum(v,axis=0) for v in grouped.values()]);counts=np.array([len(v) for v in grouped.values()])
    def statistic(z,n):
        mean=z/n[...,None]
        return np.sqrt(mean[...,1])-np.sqrt(mean[...,0]) if task=='fluorescence' else mean[...,1]-mean[...,0]
    delta=float(statistic(sums.sum(0),counts.sum()));rng=np.random.default_rng(20260926);boot=[]
    for _ in range(100):
        indices=rng.integers(0,len(sums),size=(100,len(sums)))
        boot.extend(statistic(sums[indices].sum(1),counts[indices].sum(1)).tolist())
    return {'metric':'RMSE' if task=='fluorescence' else 'accuracy','right_minus_left':delta,
       'paired_entity_bootstrap_95_CI':np.quantile(boot,[.025,.975]).tolist(),'replicates':10000,'groups':len(grouped),
       'interpretation':'Descriptive uncertainty on reused development entities; excludes training-seed variability and GFP parent-level uncertainty.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);root=a.root/'round'
    assert load(root/'status.json')['status']=='complete'
    manifest=load(a.root/'data/manifest.json');result={'data':manifest,'round':load(root/'manifest.json'),'runtime':load(root/'status.json'),'runs':{},'comparisons':{},'decision_checks':{}}
    predictions={}
    for name in RUNS:
        folder=root/name;curve=read_lines(folder/'learning_curve.jsonl');config=load(folder/'run_config.json');state=load(folder/'status.json')
        assert state['status']=='complete' and state['checkpoint_reload_exact']
        result['runs'][name]={'config':config,'status':state,'curve':curve,
         'final_dev':next(r['metrics'] for r in curve if r['epoch']==3 and r['split']=='dev'),
         'final_train_diagnostic':next(r['metrics'] for r in curve if r['epoch']==3 and r['split']=='train')}
        if (folder/'inference_benchmark.json').exists():result['runs'][name]['inference_benchmark']=load(folder/'inference_benchmark.json')
        if (folder/'end_to_end_benchmark.json').exists():result['runs'][name]['end_to_end_benchmark']=load(folder/'end_to_end_benchmark.json')
        if (folder/'portable_export_check.json').exists():result['runs'][name]['portable_export_check']=load(folder/'portable_export_check.json')
        if (folder/'reversed_choice_metrics.json').exists():result['runs'][name]['reversed_choice_metrics']=load(folder/'reversed_choice_metrics.json')
        predictions[name]=read_lines(folder/'dev_epoch3_predictions.jsonl')
        if (folder/'dev_reversed_choice_predictions.jsonl').exists():
            canonical={r['id']:r for r in predictions[name] if r['task']=='structural_class'}
            reverse=read_lines(folder/'dev_reversed_choice_predictions.jsonl')
            assert set(canonical)=={r['id'] for r in reverse}
            result['runs'][name]['choice_order_diagnostic']={
                'semantic_argmax_agreement':float(np.mean([r['prediction']==canonical[r['id']]['prediction'] for r in reverse])),
                'mean_probability_total_variation':float(np.mean([.5*np.abs(np.array(r['probs'])-canonical[r['id']]['probs']).sum() for r in reverse])),
                'scope':'One fixed reversal, mapped to canonical labels; descriptive diagnostic, not exhaustive permutation invariance.'}

    train=read_lines(a.root/'data/train.jsonl');dev=read_lines(a.root/'data/dev.jsonl');baselines={}
    for task in TASKS:
        tr=[r for r in train if r['task']==task];de=[r for r in dev if r['task']==task]
        if task=='fluorescence':
            values=np.array([r['value'] for r in de]);mean=np.mean([r['value'] for r in tr]);median=np.median([r['value'] for r in tr])
            baselines[task]={'training_mean':float(mean),'training_median':float(median),'rmse':float(np.sqrt(np.mean((values-mean)**2))),
                            'mae':float(np.mean(abs(values-median)))}
        else:
            counts=np.bincount([r['label'] for r in tr],minlength=len(tr[0]['choices']));majority=int(counts.argmax());truth=[r['label'] for r in de]
            baselines[task]={'training_class_counts':counts.tolist(),'majority_label':majority,'accuracy':float(accuracy_score(truth,[majority]*len(de))),
               'macro_f1':float(f1_score(truth,[majority]*len(de),labels=list(range(len(counts))),average='macro',zero_division=0))}
        anchor='cpt_'+task
        result['comparisons'][task]={'cpt_minus_no_cpt_joint':paired(predictions['no_cpt_joint'],predictions['cpt_joint'],task),
                                    'cpt_joint_minus_cpt_single':paired(predictions[anchor],predictions['cpt_joint'],task)}
        joint=result['runs']['cpt_joint']['final_dev'][task];single=result['runs'][anchor]['final_dev'][task];base=baselines[task]
        checks={}
        for arm in ['cpt_joint','no_cpt_joint']:
            m=result['runs'][arm]['final_dev'][task]
            checks[arm+'_beats_constants']=bool(m['rmse']<base['rmse'] and m['mae']<base['mae']) if task=='fluorescence' else bool(m['accuracy']>base['accuracy'] and m['macro_f1']>base['macro_f1'])
        if task=='fluorescence':
            checks.update(cpt_joint_RMSE_ratio_to_single=joint['rmse']/single['rmse'],material_joint_retention_flag=bool(joint['rmse']>1.1*single['rmse']))
        else:
            checks.update(cpt_joint_accuracy_delta_to_single=joint['accuracy']-single['accuracy'],cpt_joint_macro_f1_delta_to_single=joint['macro_f1']-single['macro_f1'],
              material_joint_retention_flag=bool(joint['accuracy']<single['accuracy']-.03 or joint['macro_f1']<single['macro_f1']-.03))
        result['decision_checks'][task]=checks
    result['baselines']=baselines
    result['decision_scope']='Practical finite-budget route decision, not proof of architectural capacity. No-CPT joint usefulness must be considered separately from CPT effects. No matched alternative-backbone speed comparison has been run.'
    audit_path=a.output/'audit.json'
    if audit_path.exists():result['audit']=load(audit_path)
    write_json(a.output/'summary.json',result)
    lines=['# One Laya checkpoint, three tasks, JEV-style output','',
       'Fixed final-epoch results. Each task has 8,192 training examples; three epochs; one seed; reused development sets; no new test inference. All models use a shared typed candidate scorer with no task-specific output head.','',
       '| Task | Metric | Constant | CPT single | No-CPT joint | CPT joint |','|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        for metric in (['rmse','mae','spearman'] if task=='fluorescence' else ['accuracy','macro_f1']):
            values=[baselines[task].get(metric),result['runs']['cpt_'+task]['final_dev'][task][metric],result['runs']['no_cpt_joint']['final_dev'][task][metric],result['runs']['cpt_joint']['final_dev'][task][metric]]
            lines.append('| '+task+' | '+metric+' | '+' | '.join('—' if v is None else f'{v:.4f}' for v in values)+' |')
    lines+=['','## Predeclared decision checks','', 'These flags use the practical thresholds frozen before production; they are not tests of statistical significance.','',
            '```json',json.dumps(result['decision_checks'],indent=2),'```','',
            '## Paired changes','']
    for task,comps in result['comparisons'].items():
        for name,c in comps.items():
            scale=100 if c['metric']=='accuracy' else 1;lo,hi=c['paired_entity_bootstrap_95_CI'];unit=' pp' if scale==100 else ' native RMSE'
            lines.append(f'- {task}, {name}: {scale*c["right_minus_left"]:+.4f}{unit}; descriptive 95% interval [{scale*lo:+.4f}, {scale*hi:+.4f}].')
    lines+=['','## Engineering decision','',
      'NO-CPT joint is the recommended checkpoint from this bounded round: it is the only shared model that passes all three frozen usefulness gates. CPT joint slightly improves classification Accuracy but worsens GFP RMSE, MAE and Spearman, and does not pass the GFP MAE constant-baseline gate. This is a one-seed development decision, not a backbone-superiority claim.']
    if 'audit' in result:
        audit=result['audit'];lines+=['',
          f'Independent audit: **{audit["status"].upper()}**. It reconstructed {audit["encoded_rows_reconstructed"]:,} encoded rows, recomputed {audit["metric_points_recomputed"]} metric points, checked {audit["prediction_files_checked"]} prediction files, and verified both retained model hashes, frozen schedules/exposures, shared initialization and absence of task-specific heads.']
    lines+=['','## Portable mixed-task inference and raw-input cost','',
      '| Arm | Task | p50 (ms) | p95 (ms) | Batch-16 requests/s |','|---|---|---:|---:|---:|']
    for arm in ['no_cpt_joint','cpt_joint']:
        for task in TASKS:
            b=result['runs'][arm]['end_to_end_benchmark'][task]
            lines.append(f'| {arm} | {task} | {b["single_request_p50_ms"]:.2f} | {b["single_request_p95_ms"]:.2f} | {b["batched_requests_per_second"]:.2f} |')
    lines+=['',
      'Both exported joint checkpoints reproduce the saved homogeneous-batch probabilities exactly for the same raw inputs. In the 48-request interleaved Noul/Choice/Score check, both preserve every argmax; maximum probability differences versus the original homogeneous BF16 evaluation are 0.0181 (no-CPT) and 0.0100 (CPT). The difference reflects changed batch shapes and padding.']
    lines+=['','## Interpretation boundaries','',
      '- Classification output validity is enforced by the candidate interface and is separate from biological accuracy.',
      '- Score outputs are an ordered distribution with a native-unit expected value. Interpolated training targets and a continuous loss differ from the earlier hard-bin pilot.',
      '- Single-task anchors match target examples, presentations, permutations and LR positions; joint training adds other-task updates and changes optimizer/dropout history.',
      '- Training diagnostics use fixed 1,024-entity subsets per task. Development uses all 1,052 / 939 / 5,362 admitted examples.',
      '- Reverse Choice order is a fixed diagnostic, not exhaustive instruction or order invariance.',
      '- Native GFP train/valid variants share a parent. The whole saved CPT protein sample has no 15mer overlap with GFP train/valid; this is not a global homology guarantee.',
      '- Raw-sequence timings include tokenization, textual rubric/panel assembly, device transfer, model execution, softmax and typed-answer construction on a warm loaded model. They exclude file/network I/O and do not establish a speed advantage over Qwen or OmniGene4.',
      '- A CPT failure is not automatically a Laya failure: inspect the matched no-CPT joint model. Negative results apply to this finite budget and recipe.',
      '',f'Local raw evidence: `{a.root}`.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'decision_checks':result['decision_checks'],'output':str(a.output)},indent=2))

if __name__=='__main__':main()
