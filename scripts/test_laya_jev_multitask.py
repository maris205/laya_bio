"""Scientific contract checks for shared JEV candidates and continuous Score."""
import copy
import unittest
import numpy as np
import torch
from transformers import ModernBertConfig,ModernBertModel
from laya_jev_multitask_data import interpolate
from laya_jev_multitask_train import SharedDecision,render,pack,losses,prediction_records


class ContractTest(unittest.TestCase):
    def test_continuous_target_expectation_and_gradient(self):
        anchors=[1.,1.75,2.5,3.25,4.]
        for value in np.linspace(1,4,113):
            q=interpolate(value,anchors)
            self.assertAlmostEqual(sum(q),1.)
            self.assertAlmostEqual(float(np.dot(q,anchors)),value)
        spec={'anchors':anchors,'train_std':.8}
        rows=[{'primitive':'score','presented_label':1,'value':1.9,'target_probs':interpolate(1.9,anchors)},
              {'primitive':'score','presented_label':3,'value':3.2,'target_probs':interpolate(3.2,anchors)}]
        logits=torch.zeros(2,5,requires_grad=True);losses(logits,rows,spec).mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertFalse(torch.equal(logits.grad[0],logits.grad[1]))
        # Accumulated micro-batches normalize by entities, not micro-batch count.
        other=torch.zeros(2,5,requires_grad=True)
        for i in range(2):(losses(other[i:i+1],rows[i:i+1],spec).sum()/2).backward()
        torch.testing.assert_close(logits.grad,other.grad)
    def test_permutation_identity_and_padding(self):
        r={'id':'x','group_id':'g','task':'structural_class','primitive':'choice','label':1,'choices':['a','b','c'],
           'qtype':0,'head_ids':[1,8,2],'option_ids':[[3,9],[3,10],[3,11]],'body_ids':[2,12,13,2],'length':13}
        canonical=render(r);reverse=render(r,reverse=True)
        self.assertEqual(reverse['presented_label'],1)
        logits=torch.tensor([[1.,3.,2.]])
        x=prediction_records([canonical],logits,{})[0]
        y=prediction_records([reverse],logits.flip(-1),{})[0]
        np.testing.assert_allclose(x["probs"],y["probs"],atol=1e-7,rtol=1e-6)
        self.assertEqual(x["output"]["answer"],y["output"]["answer"])
        self.assertEqual(x["prediction"],y["prediction"])
        torch.manual_seed(7)
        cfg=ModernBertConfig(vocab_size=64,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,
          max_position_embeddings=128,global_attn_every_n_layers=2,local_attention=64,pad_token_id=0,reference_compile=False)
        cfg._attn_implementation='sdpa'
        model=SharedDecision(ModernBertModel(cfg)).eval()
        long=copy.deepcopy(r);long['body_ids']=[2]+[12]*20+[2];long['length']=31
        long=render(long)
        with torch.no_grad():
            alone=model(**pack([canonical],0,'cpu'))[0]
            together=model(**pack([canonical,long],0,'cpu'))[0]
        torch.testing.assert_close(alone,together,atol=2e-5,rtol=2e-5)
        self.assertFalse(any('classifiers' in n or 'regressors' in n for n,_ in model.named_parameters()))

if __name__=='__main__':unittest.main()
