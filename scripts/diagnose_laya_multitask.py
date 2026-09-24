#!/usr/bin/env python3
"""Train/dev-only diagnostics and fixed position-aware CPU controls; no tuning."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from Bio.Align import PairwiseAligner
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer
import laya_multitask_data as data
import laya_multitask_experiment as exp


def score_metrics(y,p):
    rho=spearmanr(y,p).statistic if np.std(p)>0 and np.std(y)>0 else float('nan')
    return {'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**.5),
            'spearman':float(rho) if np.isfinite(rho) else None,'prediction_std':float(np.std(p))}


def class_metrics(y,p,k):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=list(range(k)),average='macro',zero_division=0)),
            'confusion_true_rows_pred_columns':confusion_matrix(y,p,labels=list(range(k))).tolist(),
            'predicted_class_counts':np.bincount(p,minlength=k).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['data-dir','round-dir','workspace','model-dir','laya-repo','output']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.data_dir/'manifest.json').read_text())
    rows={}
    for split in ['train','dev']:
        path=a.data_dir/manifest['outputs'][split]['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['outputs'][split]['sha256']
        rows[split]=[json.loads(x) for x in path.read_text().splitlines()]
    tok=AutoTokenizer.from_pretrained(a.model_dir/'tokenizer');sys.path.insert(0,str(a.laya_repo))
    from laya.common import build_sequence
    source=a.workspace/'data/05_task_extension_sources'
    splice=pq.read_table(source/'dna_splice_site_prediction/data/train-00000-of-00001.parquet').to_pylist()
    gfp={s:json.loads((source/f'tape/fluorescence/fluorescence_{s}.json').read_text()) for s in ['train','valid']}
    report={'test_access':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_manifest_sha256':hashlib.sha256((a.data_dir/'manifest.json').read_bytes()).hexdigest(),
            'controls':'Position one-hot fit on train only; logistic C=1 max_iter=2000; ridge alpha=10 lsqr. Fixed before results, no dev selection.',
            'tasks':{}}
    for task in ['splice','fluorescence']:
        selected={s:[r for r in rr if r['task']==task] for s,rr in rows.items()};out={}
        for split,rr in selected.items():
            seq=[r['inputs'][0]['sequence'] for r in rr];source_errors=0
            for r in rr:
                if task=='splice':
                    raw=splice[int(r['id'].split(':')[-1])]
                    source_errors+=int(raw['sequence'].upper()!=r['inputs'][0]['sequence'] or int(raw['label'])!=r['label'])
                else:
                    _,native,index=r['id'].split(':');raw=gfp[native][int(index)]
                    source_errors+=int(raw['primary'].upper()!=r['inputs'][0]['sequence'] or float(raw['log_fluorescence'][0])!=r['value'])
            encoded=exp.make_items(rr,tok,build_sequence,'candidate',20260924,split=='train')
            items=[i for e in encoded for i in e['items']]
            unknown=sum(i['ids'].count(tok.unk_token_id) for i in items)
            label_errors=sum(i['order'][i['label']]!=r['label'] for i,r in zip(items,rr))
            out[split]={'entities':len(rr),'label_counts':dict(Counter(r['label'] for r in rr)),
                        'sequence_lengths':dict(Counter(map(len,seq))),
                        'full_input_token_min_max':[min(len(i['ids']) for i in items),max(len(i['ids']) for i in items)],
                        'unknown_tokens':unknown,'source_row_mismatches':source_errors,'candidate_label_order_errors':label_errors,
                        'marker_count_errors':sum(len(i['markers'])!=len(i['choices']) for i in items)}
            if task=='fluorescence':
                v=np.array([r['value'] for r in rr]);out[split]['target_quantiles']=np.quantile(v,[0,.1,.25,.5,.75,.9,1]).tolist()
                out[split]['target_std']=float(v.std())
            if source_errors or label_errors or unknown:raise ValueError((task,split,out[split]))
        tr=selected['train'];de=selected['dev'];seqs={s:[r['inputs'][0]['sequence'] for r in rr] for s,rr in selected.items()}
        if task=='fluorescence':
            modal_length=Counter(map(len,seqs['train'])).most_common(1)[0][0]
            consensus=''.join(Counter(col).most_common(1)[0][0] for col in zip(*[s for s in seqs['train'] if len(s)==modal_length]))
            aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-1,open_gap_score=-3,extend_gap_score=-.5)
            aligned={};insertions={}
            for split,ss in seqs.items():
                aligned[split]=[];insertions[split]=0
                for seq in ss:
                    al=aligner.align(consensus,seq)[0];ref,target=al[0],al[1]
                    ins=sum(r=='-' for r in ref);insertions[split]+=ins
                    aligned[split].append(''.join(t for r,t in zip(ref,target) if r!='-'))
            out['alignment']={'reference':'Training modal-length positional consensus','reference_sha256':hashlib.sha256(consensus.encode()).hexdigest(),
                              'match':2,'mismatch':-1,'gap_open':-3,'gap_extend':-.5,'inserted_residues_not_encoded':insertions}
            seqs=aligned
        assert len({len(s) for ss in seqs.values() for s in ss})==1
        encoder=OneHotEncoder(handle_unknown='ignore',dtype=np.float64)
        x=encoder.fit_transform([list(s) for s in seqs['train']]);xd=encoder.transform([list(s) for s in seqs['dev']])
        if task=='splice':
            y=np.array([r['label'] for r in tr]);yd=np.array([r['label'] for r in de])
            model=LogisticRegression(C=1,max_iter=2000,random_state=20260924).fit(x,y)
            out['position_logistic']={'train':class_metrics(y,model.predict(x),3),'dev':class_metrics(yd,model.predict(xd),3),'iterations':model.n_iter_.tolist()}
            # Report class-specific central motifs without changing labels or selecting features.
            out['train_central_20mer_top3_by_label']={str(k):Counter(s[190:210] for s,r in zip(seqs['train'],tr) if r['label']==k).most_common(3) for k in range(3)}
        else:
            y=np.array([r['value'] for r in tr]);yd=np.array([r['value'] for r in de])
            model=Ridge(alpha=10,solver='lsqr').fit(x,y)
            out['position_ridge']={'train':score_metrics(y,model.predict(x)),'dev':score_metrics(yd,model.predict(xd))}
            out['train_median_constant_dev']=score_metrics(yd,np.full(len(yd),np.median(y)))
            out['train_mean_constant_dev']=score_metrics(yd,np.full(len(yd),np.mean(y)))
            out['oracle_bin_anchor_dev']=score_metrics(yd,np.array(manifest['score_spec']['anchors'])[[r['label'] for r in de]])
            consensus=''.join(Counter(col).most_common(1)[0][0] for col in zip(*seqs['train']))
            out['hamming_to_training_consensus']={s:{'quantiles':np.quantile([sum(a!=b for a,b in zip(v,consensus)) for v in vv],[0,.5,.9,1]).tolist()} for s,vv in seqs.items()}
        out['joint_predictions']={}
        for kind in ['candidate','shared_heads']:
            out['joint_predictions'][kind]={}
            for tag in ['before','after']:
                pred=[json.loads(l) for l in (a.round_dir/f'{kind}_seed20260924'/f'dev_{tag}_predictions.jsonl').read_text().splitlines()]
                pred=[r for r in pred if r['task']==task];truth={r['id']:r for r in de}
                assert len(pred)==len(de) and all(r['label']==truth[r['id']]['label'] for r in pred)
                pp=np.array([r['probs'] for r in pred]);yy=np.array([r['label'] for r in pred])
                diag=class_metrics(yy,pp.argmax(1),pp.shape[1]);diag['mean_probability']=pp.mean(0).tolist()
                if task=='fluorescence':
                    for field in ['score_prediction','scalar_prediction']:
                        if field in pred[0]:diag[field]=score_metrics([r['value'] for r in pred],[r[field] for r in pred])
                out['joint_predictions'][kind][tag]=diag
        report['tasks'][task]=out
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
