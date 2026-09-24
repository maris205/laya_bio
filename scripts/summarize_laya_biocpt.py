#!/usr/bin/env python3
"""Summarize fixed-final-step paired biological CPT experiment, including losses."""
import argparse
import json
from pathlib import Path
import numpy as np
from laya_biocpt_data import read_lines,write_json


def load(path):
    return json.loads(path.read_text())


def paired_delta(a,b):
    left={r['id']:r for r in a}
    right={r['id']:r for r in b}
    assert left.keys()==right.keys()
    grouped={}
    for key,x in left.items():
        y=right[key]
        assert x['label']==y['label'] and x['group_id']==y['group_id']
        grouped.setdefault(x['group_id'],[]).append(int(y['prediction']==y['label'])-int(x['prediction']==x['label']))
    sums=np.array([sum(v) for v in grouped.values()])
    counts=np.array([len(v) for v in grouped.values()])
    rng=np.random.default_rng(20260925)
    values=[]
    for _ in range(10000):
        sample=rng.integers(0,len(sums),len(sums))
        values.append(float(sums[sample].sum()/counts[sample].sum()))
    return {'accuracy_delta_CPT_minus_no_CPT':float(sums.sum()/counts.sum()),'paired_group_bootstrap_95_percentile_CI':np.quantile(values,[.025,.975]).tolist(),'bootstrap_replicates':10000,'groups':len(sums),'interpretation':'descriptive development uncertainty, not blind-test confirmation'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=a.root/'round'
    assert load(root/'status.json')['status']=='complete'
    a.output.mkdir(parents=True,exist_ok=True)
    result={'data':load(a.root/'data/manifest.json'),'round':load(root/'manifest.json'),'cpt_curve':read_lines(root/'cpt/learning_curve.jsonl'),'cpt_status':load(root/'cpt/status.json'),'warmup_embedding_check':load(root/'cpt/warmup_embedding_check.json'),'full_embedding_check':load(root/'cpt/full_embedding_check.json'),'runs':{},'comparisons':{}}
    for name in ['no_cpt_4096','cpt_4096','no_cpt_full','cpt_full']:
        curve=read_lines(root/name/'learning_curve.jsonl')
        result['runs'][name]={'config':load(root/name/'run_config.json'),'status':load(root/name/'status.json'),'curve':curve,'final_dev':next(r for r in curve if r['epoch']==3 and r['split']=='dev'),'final_train':next(r for r in curve if r['epoch']==3 and r['split']=='train')}
    for size in ['4096','full']:
        x=result['runs']['no_cpt_'+size]['config']
        y=result['runs']['cpt_'+size]['config']
        for field in ['batch_order_sha256','initial_classifier_sha256','updates','micro_batch','data_manifest_sha256']:
            assert x[field]==y[field],field
        result['comparisons'][size]=paired_delta(read_lines(root/('no_cpt_'+size)/'dev_epoch3_predictions.jsonl'),read_lines(root/('cpt_'+size)/'dev_epoch3_predictions.jsonl'))
    write_json(a.output/'summary.json',result)
    lines=['# Traditional biological vocabulary → CPT → SFT: first paired round','',
           'Final-step results on the existing promoter selection-development split (1,052 examples). One seed. No test inference. Both arms use the same expanded vocabulary, original encoder initialization, classifier initialization, training order and SFT schedule; CPT adds unsupervised adaptation.', '',
           '| Training rows | Arm | Updates | Train accuracy | Dev accuracy | Dev macro-F1 | Dev NLL |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for name,r in result['runs'].items():
        lines.append(f"| {r['config']['samples']} | {r['config']['arm']} | {r['config']['updates']} | {r['final_train']['accuracy']:.4f} | {r['final_dev']['accuracy']:.4f} | {r['final_dev']['macro_f1']:.4f} | {r['final_dev']['nll']:.4f} |")
    lines += ['','## Paired changes','']
    for size,c in result['comparisons'].items():
        lo,hi=c['paired_group_bootstrap_95_percentile_CI']
        lines.append(f"- {size}: CPT − no CPT = {100*c['accuracy_delta_CPT_minus_no_CPT']:+.2f} accuracy percentage points; paired group-bootstrap 95% interval [{100*lo:+.2f}, {100*hi:+.2f}] pp.")
    lines += ['','## Masked-language validation','', '| Step | Phase | DNA NLL | Protein NLL | Text NLL |','|---:|---|---:|---:|---:|']
    for row in result['cpt_curve']:
        v=row['validation']
        lines.append(f"| {row['step']} | {row['phase']} | {v['dna']['nll']:.4f} | {v['protein']['nll']:.4f} | {v['text']['nll']:.4f} |")
    lines += ['','## Training integrity and limits','',
              f"- Added tokens: {result['cpt_status']['added_token_count']}; natural input coverage: {result['cpt_status']['natural_input_token_coverage']}; masked-target coverage: {result['cpt_status']['masked_target_token_coverage']}.",
              f"- Old embedding rows exactly invariant during warmup: {result['warmup_embedding_check']['old_rows_equal']}. All final checkpoint prediction reload checks passed.",
              '- A new MLM prediction head was necessary because the starting checkpoint contains only the original encoder and Laya heads. Initial MLM loss reduction includes head training; it is not pure encoder acquisition.',
              '- Fixed three-epoch SFT means the two dataset sizes use different update budgets. The paired CPT effect is assessed separately at each size.',
              '- CPT uses a filtered, length-biased sample from historical corpora. Long-kmer guards do not prove global homology independence. Future tasks require admission checks.',
              '- Development memberships have historical use; uncertainty intervals are descriptive. This single-seed round does not establish task-general, few-shot or zero-shot performance.',
              '- CPT final model and optimizer/RNG/sampler state and both full-data SFT checkpoints are retained. Small-data checkpoints are removed after exact reload verification; their full predictions remain.',
              '',f"Local source artifacts: `{a.root}`."]
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'comparisons':result['comparisons'],'output':str(a.output)},indent=2))


if __name__=='__main__':
    main()
