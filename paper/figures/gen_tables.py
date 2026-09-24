"""Generate every reported table directly from completed experiment artifacts."""
from paper_plot_style import *
import hashlib
import statistics as st

TABLES=FIG.parent/'tables';TABLES.mkdir(exist_ok=True)
result=read('artifacts/laya_locked_test/results.json')
agg=result['aggregate']
conditions=('raw','full_bpe','b1','text_only')
names={'raw':'Raw candidate','full_bpe':'BioBPE candidate','b1':'BioBPE fixed head (B1)','text_only':'Text-only candidate'}


def table(name,header,rows,align):
    value='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(header)+' \\\\\n\\midrule\n'
    value+=''.join(' & '.join(map(str,row))+' \\\\\n' for row in rows)
    value+='\\bottomrule\n\\end{tabular}\n'
    (TABLES/f'{name}.tex').write_text(value)


def fmt(c,t,k,mode='raw',scale=100,digits=2):
    x=agg[c][mode][t][k]
    return f"${x['mean']*scale:.{digits}f} \\pm {x['std']*scale:.{digits}f}$"


table('main_results',['Model','DNA Acc.','DNA F1','Protein Acc.','Protein F1'],
      [[names[c]]+[fmt(c,t,k) for t in TASKS for k in ('accuracy','macro_f1')] for c in conditions],'lrrrr')
table('calibration',['Model','Task','NLL (raw)','NLL (cal.)','Brier (cal.)','ECE (cal.)'],
      [[names[c],short,fmt(c,t,'nll','raw',1,3),fmt(c,t,'nll','calibrated',1,3),
        fmt(c,t,'brier','calibrated',1,3),fmt(c,t,'ece15','calibrated',1,3)]
       for t,short in zip(TASKS,('DNA','Protein')) for c in conditions],'llrrrr')
eligible=read('artifacts/laya_formal_data/eligible_ids.json')['tasks']
table('data',['Task','Train','Dev','Calibration','Test'],
      [[short]+[f"{eligible[t]['splits'][s]['count']:,}" for s in ('train','selection_dev','calibration','test')]
       for t,short in zip(TASKS,('DNA promoter (2 classes)','Protein structure (7 classes)'))],'lrrrr')
table('seeds',['Seed','Model','DNA Acc.','DNA F1','Protein Acc.','Protein F1'],
      [[r['seed'],names[r['condition']]]+[f"{100*r['metrics']['raw'][t][k]:.2f}" for t in TASKS for k in ('accuracy','macro_f1')]
      for r in result['runs']],'llrrrr')
table('all_metrics',['Model','Task','Bal. Acc.','MCC','Brier (raw)','ECE (raw)'],
      [[names[c],short,fmt(c,t,'balanced_accuracy'),fmt(c,t,'mcc','raw',1,3),
        fmt(c,t,'brier','raw',1,3),fmt(c,t,'ece15','raw',1,3)]
       for t,short in zip(TASKS,('DNA','Protein')) for c in conditions],'llrrrr')
runtime=[]
for c in conditions:
    parent='laya_direct_legacy' if c in ('raw','full_bpe') else 'laya_controls'
    s=[read(f'artifacts/{parent}/{c}_seed{seed}/summary.json') for seed in SEEDS]
    seconds=[r['training']['seconds']/60 for r in s]
    memory=[r['training']['peak_allocated_gib'] for r in s]
    runtime.append([names[c],f"{s[0]['trainable_parameters']/1e6:.3f}",
                    f"${st.mean(seconds):.2f} \\pm {st.stdev(seconds):.2f}$",
                    f"${st.mean(memory):.2f} \\pm {st.stdev(memory):.2f}$"])
table('resources',['Model','Trainable (M)','Training (min)','Peak allocated (GiB)'],runtime,'lrrr')
diagnostics=[]
for c in ('raw','full_bpe'):
    for t,short in zip(TASKS,('DNA','Protein')):
        ss=[read(f'artifacts/laya_controls/sequence_diagnostics/{c}_seed{seed}/summary.json') for seed in SEEDS]
        measures=[[s['original']['raw'][t]['accuracy'] for s in ss],
                  [s['variants']['sequence_removed']['reference_label_metrics']['raw'][t]['accuracy'] for s in ss],
                  [st.mean(s['variants'][f'residue_shuffle_{i}']['reference_label_metrics']['raw'][t]['accuracy'] for i in range(3)) for s in ss],
                  [s['variants']['candidate_permutation']['comparison_to_original'][t]['prediction_agreement'] for s in ss]]
        diagnostics.append([names[c],short]+[f'${100*st.mean(v):.2f} \\pm {100*st.stdev(v):.2f}$' for v in measures])
table('diagnostics',['Model','Task','Original','Removed','Shuffled','Order agreement'],diagnostics,'llrrrr')
audit=read('artifacts/laya_locked_test/independent_metric_audit.json')
truth=audit['runs'][0]['tasks']
classes=read('artifacts/laya_controls/b1_seed20260922/summary.json')['class_names']
classrows=[]
for t,short in zip(TASKS,('DNA','Protein')):
    for i,label in enumerate(classes[t]):classrows.append([short,i,label,truth[t]['true_counts'][str(i)]])
table('class_counts',['Task','ID','Canonical class','Test n'],classrows,'lr lr'.replace(' ',''))
coverage=read('artifacts/laya_direct_legacy/audit/input_audit.json')
table('lengths',['Task','Representation','Train median','Train p95','Train max'],
      [[short,names[c]]+[coverage['conditions'][c]['train'][t][k] for k in ('length_p50','length_p95','length_max')]
       for t,short in zip(TASKS,('DNA','Protein')) for c in ('raw','full_bpe')],'llrrr')
provenance={'source':'artifacts/laya_locked_test/results.json',
            'source_sha256':hashlib.sha256((ROOT/'artifacts/laya_locked_test/results.json').read_bytes()).hexdigest(),
            'tables':[p.name for p in sorted(TABLES.glob('*.tex'))],
            'test_manifest_sha256':result['manifest_sha256'], 'no_new_evaluation':True}
(FIG.parent/'table_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance))
