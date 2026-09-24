#!/usr/bin/env python3
"""Evaluate every global three-class order on the same 32 training examples."""
import argparse,gc,itertools,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
import laya_microfit as micro
import laya_multitask_experiment as exp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data-dir','model-dir','laya-repo','base-run','augmented-run','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8);tok=AutoTokenizer.from_pretrained(a.model_dir/'tokenizer')
    manifest=json.loads((a.data_dir/'manifest.json').read_text());path=a.data_dir/manifest['outputs']['train']['file']
    assert micro.sha_file(path)==manifest['outputs']['train']['sha256']
    rows=[json.loads(l) for l in path.read_text().splitlines()];selected=micro.select_training_subset(rows,'splice')
    _,_,builder=exp.core.import_laya(str(a.laya_repo));out={'dev_access':False,'test_access':False,'entities':32,'permutations':6,'script_sha256':micro.sha_file(Path(__file__)),'models':{}}
    for name,source in [('fixed_order_training',a.base_run),('resampled_order_training',a.augmented_run)]:
        summary=json.loads((source/'summary.json').read_text());info=summary['task_info']
        assert summary['selected_ids']==[r['id'] for r in selected]
        model,_=exp.build('candidate',a.model_dir,a.laya_repo,info,torch.device('cuda'),source/'checkpoint')
        by_order=[];all_predictions=[]
        for permutation in itertools.permutations(range(3)):
            entities=exp.make_items(selected,tok,builder,'candidate',20260924,False,manifest['max_length'])
            for entity in entities:
                for item in entity['items']:
                    ids,markers=builder(tok,item['state'],exp.data.rubric(item),max_len=1000000,head_max_len=256,option_order=list(permutation))
                    item.update(ids=ids,markers=markers,label=permutation.index(item['label']),order=list(permutation))
            records,_=exp.evaluate(model,'candidate',entities,tok,torch.device('cuda'),info,8)
            records=micro.canonicalize(records,entities,'candidate');metrics=micro.diagnostic_metrics(records,info)['splice']
            by_order.append({'order':list(permutation),'metrics':metrics});all_predictions.append(np.array([r['probs'] for r in records]).argmax(1))
        array=np.stack(all_predictions)
        out['models'][name]={'by_order':by_order,'min_accuracy':min(r['metrics']['accuracy'] for r in by_order),
            'max_accuracy':max(r['metrics']['accuracy'] for r in by_order),'mean_accuracy':float(np.mean([r['metrics']['accuracy'] for r in by_order])),
            'fraction_examples_same_prediction_all_orders':float(np.mean(np.all(array==array[0],axis=0)))}
        del model;gc.collect();torch.cuda.empty_cache()
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
