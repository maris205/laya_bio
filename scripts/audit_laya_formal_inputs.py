#!/usr/bin/env python3
"""Independently recheck frozen full-input eligibility without a model forward.

Version 2 keeps the original full audit/eligibility manifests intact and checks
that the newly computed common-complete set matches the set used by training.
Test membership/lengths are checked, but test labels never decide eligibility.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/laya'))
from laya.common import build_sequence, render_options
from laya_representation import Representation

SPLITS = ('train','selection_dev','calibration','test')
TASKS = ('promoter_detection','fold_class')
QUESTION = 'Choose the correct biological label for the sequence.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x = sorted(values)
    return {'n':len(x),'min':x[0], 'p50':x[int((len(x)-1)*.5)],
            'p95':x[int((len(x)-1)*.95)], 'p99':x[int((len(x)-1)*.99)], 'max':x[-1]}


def uncapped_head(tok, q):
    """Head/candidate text before all internal builder caps, including 48/option."""
    head = tok('choice question: ' + q['ins'], add_special_tokens=False)['input_ids']
    parts = [[tok.mask_token_id]+tok(' '+x, add_special_tokens=False)['input_ids'] for x in render_options(q)]
    ids = [tok.cls_token_id] + head + [tok.sep_token_id]
    for part in parts:
        ids.extend(part)
    return ids+[tok.sep_token_id]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'artifacts/laya_formal_data')
    parser.add_argument('--representation-dir',type=Path,default=ROOT/'artifacts/laya_formal_representation')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/laya_formal_input_audit_v2.json')
    a=parser.parse_args()
    old_ids=json.loads((a.data_dir/'eligible_ids.json').read_text())
    reps=[Representation.load(a.representation_dir,expanded=x) for x in (False,True)]
    report={'audit_version':2,'model_forward':False,'max_len':1024,'head_max_len':256,
            'representation_metadata_sha256':sha(a.representation_dir/'metadata.json'),
            'reference_eligibility_sha256':sha(a.data_dir/'eligible_ids.json'), 'tasks':{}}
    for task in TASKS:
        result={}
        for split in SPLITS:
            path=a.data_dir/f'{task}_{split}.jsonl'
            rows=[json.loads(x) for x in path.read_text().splitlines() if x]
            lengths=[[],[]];heads=[0,0];overflow=[0,0];exceptions=[];kept=[];states_differ=0
            prompt_changed=0; seen_prompt={}; class_before=Counter();class_after=Counter()
            for row in rows:
                q={'t':'choice','ins':QUESTION,'crit':dict.fromkeys(row['choices'])}
                states=[rep.state(row) for rep in reps]
                if states[0]!=states[1]:
                    states_differ+=1
                prompts=(row['context'],*row['choices'],QUESTION)
                if prompts not in seen_prompt:
                    before=reps[0].tokenizer(list(prompts),add_special_tokens=False)['input_ids']
                    after=reps[1].tokenizer(list(prompts),add_special_tokens=False)['input_ids']
                    seen_prompt[prompts]=before!=after
                prompt_changed+=int(seen_prompt[prompts])
                reasons=[]
                for i,rep in enumerate(reps):
                    full,markers=build_sequence(rep.tokenizer,states[i],q,max_len=1000000,head_max_len=256)
                    actual,_=build_sequence(rep.tokenizer,states[i],q,max_len=1024,head_max_len=256)
                    blank,_=build_sequence(rep.tokenizer,'',q,max_len=1000000,head_max_len=256)
                    head_ok=blank[:-1]==uncapped_head(rep.tokenizer,q)
                    assert actual==full[:1023]+[rep.tokenizer.sep_token_id] if len(full)>1024 else actual==full
                    assert len(markers)==len(row['choices'])
                    lengths[i].append(len(full)); heads[i]+=int(not head_ok);overflow[i]+=int(len(full)>1024)
                    if not head_ok or len(full)>1024:
                        reasons.append({'condition':['m1','m2'][i], 'full_tokens':len(full),'head_truncated':not head_ok})
                    _,pieces=rep.source_pieces(row)
                    if ''.join(pieces)!=row['sequence']:
                        raise AssertionError('Source BPE does not preserve sequence: '+row['id'])
                if not reasons and states[0]==states[1]:
                    kept.append(row['id'])
                else:
                    exceptions.append({'id':row['id'],'reasons':reasons})
                if split!='test':
                    class_before[str(row['label'])]+=1
                    if not reasons: class_after[str(row['label'])]+=1
            expected=old_ids['tasks'][task]['splits'][split]['ids']
            assert kept==expected, f'Eligibility drift: {task}/{split}'
            assert states_differ==0 and prompt_changed==0
            result[split]={'n':len(rows),'eligible':len(kept),'eligible_rate':len(kept)/len(rows),
                           'm1_full_length':stats(lengths[0]),'m2_full_length':stats(lengths[1]),
                           'overflow_m1_m2':overflow,'head_truncation_m1_m2':heads,
                           'same_states':True,'prompt_changed':prompt_changed,
                           'reference_ids_identical':True,'input_sha256':sha(path),'exceptions':exceptions}
            if split!='test':
                assert set(class_before)==set(class_after)
                result[split]['class_coverage']={k:{'before':v,'retained':class_after[k],
                                                 'rate':class_after[k]/v} for k,v in sorted(class_before.items())}
            print(task,split,len(rows),'eligible',len(kept),'overflow',overflow,flush=True)
        report['tasks'][task]=result
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Verified all frozen eligible ID lists; wrote',a.output,flush=True)

if __name__=='__main__':
    main()
