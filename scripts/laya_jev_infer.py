#!/usr/bin/env python3
"""Load one exported checkpoint and answer mixed biological JEV requests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from transformers import ModernBertConfig,ModernBertModel
from laya_biocpt_data import sha
from laya_jev_multitask_data import Representation
from laya_jev_multitask_train import SharedDecision,render,pack


class Predictor:
    def __init__(self,checkpoint_dir,verify_hash=True):
        self.directory=Path(checkpoint_dir)
        self.config=json.loads((self.directory/'decision_config.json').read_text())
        if verify_hash:assert sha(self.directory/'model.safetensors')==self.config['model_sha256']
        config=ModernBertConfig.from_pretrained(self.directory/'encoder')
        config._attn_implementation='sdpa';config.reference_compile=False
        self.model=SharedDecision(ModernBertModel(config))
        self.model.load_state_dict(load_file(str(self.directory/'model.safetensors')),strict=True)
        self.model.to('cuda').eval();self.rep=Representation(self.directory)
    def prepare(self,request,index=0):
        if 'task' in request:
            template=self.config['task_templates'][request['task']]
            # Task name chooses a textual rubric only, never a parameter head.
            request={**template,**request}
        primitive=request['primitive'];assert primitive in ['choice','noul','score']
        modality=request['modality'];assert modality in ['dna','protein']
        seq=request['sequence'].upper();assert seq and all('A'<=c<='Z' for c in seq)
        assert '<BIO_' not in request['question'] and self.rep.tokenizer.mask_token not in request['question']
        if primitive=='score':
            anchors=[float(v) for v in request['anchors']]
            assert len(anchors)>=2 and all(a<b for a,b in zip(anchors,anchors[1:]))
            choices=[f'log fluorescence = {v:.8g}' for v in anchors]
        elif primitive=='noul':choices=request.get('choices',['false: not a promoter','true: promoter']);assert len(choices)==2
        else:choices=request['choices'];assert len(choices)>=2 and len(set(choices))==len(choices)
        row={'id':str(request.get('id',index)),'group_id':'inference','task':request.get('task','custom'),'primitive':primitive,
             'modality':modality,'sequence':seq,'question':request['question'],'choices':choices,'label':0}
        if primitive=='score':row['anchors']=anchors
        row=self.rep.prepare(row)
        if row['length']>self.config['max_input_tokens']:raise ValueError('Full request exceeds admitted context; no truncation is performed.')
        return render(row)
    @torch.no_grad()
    def predict(self,requests):
        rows=[self.prepare(r,i) for i,r in enumerate(requests)]
        if not rows:return []
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.model(**pack(rows,self.rep.tokenizer.pad_token_id))
        answers=[]
        for row,logit in zip(rows,logits):
            probs=logit[:len(row['choices'])].float().softmax(-1).cpu().tolist();best=max(range(len(probs)),key=probs.__getitem__)
            answer={'id':row['id'],'type':row['primitive'],'probabilities':probs}
            if row['primitive']=='noul':answer['answer']=bool(best)
            elif row['primitive']=='choice':answer.update(index=best,answer=row['choices'][best])
            else:answer.update(level=best,value=sum(v*p for v,p in zip(row['anchors'],probs)),units='log fluorescence')
            answers.append(answer)
        return answers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();torch.set_num_threads(8);model=Predictor(a.checkpoint)
    requests=[json.loads(line) for line in a.requests.read_text().splitlines()];outputs=[]
    for start in range(0,len(requests),a.batch_size):outputs.extend(model.predict(requests[start:start+a.batch_size]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs))
    print(json.dumps({'checkpoint':str(a.checkpoint),'requests':len(outputs),'output':str(a.output)}))

if __name__=='__main__':main()
