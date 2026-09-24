#!/usr/bin/env python3
"""Evaluate fixed training subsets for existing joint checkpoints, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import gc
import torch
from transformers import AutoTokenizer
import laya_multitask_experiment as exp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data-dir','model-dir','laya-repo','joint-dir','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8);manifest=json.loads((a.data_dir/'manifest.json').read_text())
    path=a.data_dir/'train.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['outputs']['train']['sha256']
    rows=[json.loads(l) for l in path.read_text().splitlines()];subset=[]
    for task in ['splice','fluorescence']:subset.extend([r for r in rows if r['task']==task][:128])
    tok=AutoTokenizer.from_pretrained(a.model_dir/'tokenizer');_,_,builder=exp.core.import_laya(str(a.laya_repo))
    result={'test_access':False,'train_subset_selection':'First 128 existing hash-ordered training rows per task, same as single-task diagnostics','models':{}}
    for kind in ['candidate','shared_heads']:
        source=a.joint_dir/f'{kind}_seed20260924';summary=json.loads((source/'summary.json').read_text());task_info=summary['task_info']
        model,_=exp.build(kind,a.model_dir,a.laya_repo,task_info,torch.device('cuda'),source/'checkpoint')
        entities=exp.make_items(subset,tok,builder,kind,20260924,False,manifest['max_length'])
        records,_=exp.evaluate(model,kind,entities,tok,torch.device('cuda'),task_info,8)
        result['models'][kind]=exp.metrics(records,task_info);del model;gc.collect();torch.cuda.empty_cache()
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
