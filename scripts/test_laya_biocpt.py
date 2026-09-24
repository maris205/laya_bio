"""Behavior tests for embedding adaptation and masked-loss normalization."""
import random
import tempfile
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import ModernBertConfig, ModernBertForMaskedLM
from laya_biocpt_train import expand, warmup_mode, optimizer, Masker, save_mlm


class Tokenizer:
    pad_token_id=0
    mask_token_id=3
    all_special_ids=[0,1,2,3]
    def __len__(self):
        return 32


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        config=ModernBertConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,pad_token_id=0,bos_token_id=1,eos_token_id=2,cls_token_id=1,sep_token_id=2,sparse_prediction=True,local_attention=16)
        config._attn_implementation='sdpa'
        config.reference_compile=False
        self.model=ModernBertForMaskedLM(config)
        self.entries=[{'expanded_id':32,'base_ids':[4,5],'modality':'dna'},{'expanded_id':33,'base_ids':[6],'modality':'protein'}]
        self.original=expand(self.model,self.entries,34)
        self.rows=[{'id':'a','modality':'dna','ids':[1,7,32,32,32,2],'body_start':2},{'id':'b','modality':'protein','ids':[1,7,33,33,2],'body_start':2}]
        self.masker=Masker(Tokenizer(),self.entries)

    def test_initialization_freeze_unfreeze_reload(self):
        model=self.model
        embedding=model.get_input_embeddings().weight
        self.assertTrue(torch.equal(embedding[32],self.original[[4,5]].mean(0)))
        self.assertEqual(model.decoder.weight.data_ptr(),embedding.data_ptr())
        hook=warmup_mode(model,32)
        opt=optimizer(model,'warmup')
        initial=embedding.detach().clone()
        x,att,y,_,_=self.masker(self.rows,random.Random(7),'cpu')
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            model(x,attention_mask=att,labels=y).loss.backward()
            opt.step()
        self.assertTrue(torch.equal(initial[:32],embedding[:32]))
        self.assertFalse(torch.equal(initial[32:],embedding[32:]))
        hook.remove()
        model.requires_grad_(True)
        opt=optimizer(model,'full')
        weight=model.model.layers[0].mlp.Wi.weight
        initial_encoder=weight.detach().clone()
        opt.zero_grad(set_to_none=True)
        model(x,attention_mask=att,labels=y).loss.backward()
        self.assertGreater(float(weight.grad.norm()),0)
        opt.step()
        self.assertFalse(torch.equal(initial_encoder,weight))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.safetensors'
            model.eval()
            expected=model(x,attention_mask=att,labels=y).logits.detach()
            save_mlm(model,path)
            missing,extra=model.load_state_dict(load_file(str(path)),strict=False)
            self.assertEqual(missing,['decoder.weight'])
            self.assertFalse(extra)
            model.tie_weights()
            self.assertTrue(torch.equal(expected,model(x,attention_mask=att,labels=y).logits))

    def test_mask_and_loss_normalization(self):
        x,att,y,inputs,targets=self.masker(self.rows,None,'cpu',fixed=True)
        self.assertTrue((y[:,:2]==-100).all())
        for i,row in enumerate(self.rows):
            self.assertTrue((y[i,len(row['ids'])-1:]==-100).all())
        self.assertEqual(dict(inputs),{32:3,33:2})
        self.assertGreater(sum(targets.values()),0)
        x2,_,y2,_,_=self.masker(self.rows,random.Random(222),'cpu',fixed=True)
        self.assertTrue(torch.equal(x,x2) and torch.equal(y,y2))
        model=self.model
        model.zero_grad(set_to_none=True)
        target=y[y!=-100]
        logits=model(x,attention_mask=att,labels=y).logits
        F.cross_entropy(logits,target).backward()
        full=model.get_input_embeddings().weight.grad.clone()
        model.zero_grad(set_to_none=True)
        for i in range(2):
            logits=model(x[i:i+1],attention_mask=att[i:i+1],labels=y[i:i+1]).logits
            F.cross_entropy(logits,y[i:i+1][y[i:i+1]!=-100],reduction='sum').div(len(target)).backward()
        torch.testing.assert_close(full,model.get_input_embeddings().weight.grad,rtol=2e-5,atol=2e-6)


if __name__=='__main__':
    unittest.main()
