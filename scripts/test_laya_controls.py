"""Label semantics, task isolation, and intervention invariants."""
from pathlib import Path
from types import SimpleNamespace
import sys
from collections import Counter

import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
import laya_formal_experiment as core
from laya_control_experiment import FixedClassModel,make_items
from laya_direct_bpe import DirectBPE
from laya_sequence_diagnostics import canonicalize,perturbed,permutation


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=4);self.emb=nn.Embedding(20,4)
    def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def test_fixed_heads_receive_gradients_and_mask_other_task_classes():
    torch.manual_seed(1)
    base=SimpleNamespace(encoder=TinyEncoder(),head=None,type_emb=nn.Embedding(3,4),
                         scorer=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,4),nn.GELU(),nn.Linear(4,1)))
    model=FixedClassModel(base)
    masks=torch.tensor([[1,1,0,0,0,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
    logits,_=model(torch.tensor([[1,2,3],[1,4,5]]),torch.ones(2,3),torch.zeros(2,7),masks,torch.zeros(2,dtype=torch.long))
    assert logits.shape==(2,7)
    assert torch.all(logits[0,2:]==-1e4)
    nn.functional.cross_entropy(logits,torch.tensor([0,6])).backward()
    assert model.classifiers['2'].weight.grad.norm()>0
    assert model.classifiers['7'].weight.grad.norm()>0
    assert model.encoder.emb.weight.grad.norm()>0


def test_inverse_candidate_mapping_preserves_semantic_label():
    row={'id':'x','choices':['a','b','c'],'label':1,'sequence':'ACGT','task':'toy'}
    order=permutation(row)
    canonical_logits=[.1,2.,-.3]
    raw=[{'id':'x','task':'toy','label':order.index(1),'logits':[canonical_logits[i] for i in order],'n_classes':3}]
    restored=canonicalize(raw,{'x':order})[0]
    assert restored['logits']==canonical_logits and restored['label']==1
    assert sum(restored['probs'])==pytest.approx(1.)


def test_shuffle_is_reproducible_preserves_residues_and_never_mutates_source():
    row={'id':'x','sequence':'MAANNTKACCGGXX'}
    result=perturbed(row,'residue_shuffle_0')
    assert result==perturbed(row,'residue_shuffle_0')
    assert result['sequence']!=row['sequence']
    assert Counter(result['sequence'])==Counter(row['sequence'])
    assert perturbed(row,'sequence_removed')['sequence']==''
    assert row['sequence']=='MAANNTKACCGGXX'


def test_text_only_removes_sequence_and_b1_omits_candidates():
    rep=DirectBPE();_,_,builder=core.import_laya('vendor/laya')
    row=core.load_split(core.ROOT/'artifacts/laya_formal_data','fold_class','train')[0]
    altered=dict(row,sequence='ANNJX')
    text=make_items([row,altered],rep,'text_only',builder,42)
    assert text[0]['ids']==text[1]['ids']
    old_size=len(rep.base)
    assert all(i<old_size for i in text[0]['ids'])
    items=make_items([row],rep,'b1',builder,42)
    assert items[0]['label']==row['label']
    newids=[i for i in items[0]['ids'] if i>=old_size]
    assert ''.join(rep.inverse[i] for i in newids)==row['sequence']
    assert rep.base.mask_token_id not in items[0]['ids']
    assert len(items[0]['markers'])==7
