#!/usr/bin/env python3
"""Audit fixed training-only microfit subsets and fit position-feature controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from Bio.Align import PairwiseAligner
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
from transformers import AutoTokenizer
import laya_microfit as micro
import laya_multitask_experiment as exp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','model-dir','laya-repo','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.data_dir/'manifest.json').read_text())
    path=a.data_dir/manifest['outputs']['train']['file'];assert micro.sha_file(path)==manifest['outputs']['train']['sha256']
    rows=[json.loads(l) for l in path.read_text().splitlines()];tok=AutoTokenizer.from_pretrained(a.model_dir/'tokenizer')
    _,_,builder=exp.core.import_laya(str(a.laya_repo))
    out={'dev_access':False,'test_access':False,'script_sha256':micro.sha_file(Path(__file__)),
         'control_hyperparameters':{'position_logistic_C':1e6,'max_iter':4000,'position_ridge_alpha':1e-6},'tasks':{}}
    for task in ['splice','fluorescence']:
        selected=micro.select_training_subset(rows,task);seqs=[r['inputs'][0]['sequence'] for r in selected]
        audit={'entities':len(selected),'unique_sequences':len(set(seqs)),'unique_groups':len({r['group'] for r in selected}),
               'selected_ids':[r['id'] for r in selected],'label_counts':dict(Counter(r['label'] for r in selected)),
               'subset_sha256':exp.data.sha('\n'.join(json.dumps(r,sort_keys=True) for r in selected)),'tokenization':{}}
        for kind in ['candidate','shared_heads']:
            ents=exp.make_items(selected,tok,builder,kind,20260924,False,manifest['max_length']);items=[i for e in ents for i in e['items']]
            audit['tokenization'][kind]={'unique_full_input_ids':len({tuple(i['ids']) for i in items}),
                'unknown_tokens':sum(i['ids'].count(tok.unk_token_id) for i in items),'min_tokens':min(len(i['ids']) for i in items),'max_tokens':max(len(i['ids']) for i in items)}
            assert audit['tokenization'][kind]['unique_full_input_ids']==32 and not audit['tokenization'][kind]['unknown_tokens']
        if task=='fluorescence':
            modal=Counter(map(len,seqs)).most_common(1)[0][0]
            reference=''.join(Counter(col).most_common(1)[0][0] for col in zip(*[s for s in seqs if len(s)==modal]))
            aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-1,open_gap_score=-3,extend_gap_score=-.5)
            aligned=[];insertions=0
            for seq in seqs:
                al=aligner.align(reference,seq)[0];ref,target=al[0],al[1];insertions+=ref.count('-')
                aligned.append(''.join(s for r,s in zip(ref,target) if r!='-'))
            audit['alignment_reference_sha256']=exp.data.sha(reference);audit['unencoded_insertions']=insertions;assert not insertions
            seqs=aligned
        onehot=OneHotEncoder(handle_unknown='ignore',dtype=np.float64);x=onehot.fit_transform([list(s) for s in seqs]);y=np.array([r['label'] for r in selected])
        clf=LogisticRegression(C=1e6,max_iter=4000,random_state=20260924).fit(x,y)
        audit['position_logistic_training_accuracy']=float(accuracy_score(y,clf.predict(x)))
        if task=='fluorescence':
            values=np.array([r['value'] for r in selected]);reg=Ridge(alpha=1e-6,solver='lsqr',tol=1e-10).fit(x,values);pred=reg.predict(x)
            audit['position_ridge_training']={'mae':float(mean_absolute_error(values,pred)),'rmse':float(mean_squared_error(values,pred)**.5)}
            audit['true_value_std']=float(values.std())
            audit['oracle_anchor_mae']=float(mean_absolute_error(values,[r['anchors'][r['label']] for r in selected]))
        out['tasks'][task]=audit
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
