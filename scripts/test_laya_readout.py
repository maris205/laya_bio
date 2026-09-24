"""Readout padding invariance and objective-isolation regression checks."""
import torch
from laya_multitask_experiment import pool_hidden,score_loss_values


def test_mean_pooling_ignores_padding_and_preserves_cls():
    h=torch.tensor([[[1.,3.],[3.,5.],[999.,-999.]]],requires_grad=True)
    mask=torch.tensor([[1,1,0]])
    assert torch.equal(pool_hidden(h,mask,'cls'),h[:,0])
    result=pool_hidden(h,mask,'mean');assert torch.equal(result,torch.tensor([[2.,4.]]))
    result.sum().backward();assert torch.equal(h.grad[:,2],torch.zeros(1,2))
    assert torch.equal(result,pool_hidden(h[:,:2],mask[:,:2],'mean'))


def test_score_objectives_update_only_supervised_output():
    for objective in ['joint','ce','mse']:
        logits=torch.tensor([[.1,.9],[.4,-.1]],requires_grad=True)
        scalar=torch.tensor([.2,-.7],requires_grad=True)
        labels=torch.tensor([1,0]);target=torch.tensor([.8,-.3])
        loss=score_loss_values(logits,scalar,labels,target,objective)
        ce=torch.nn.functional.cross_entropy(logits,labels,reduction='none');mse=(scalar-target).square()
        assert torch.equal(loss,{'joint':.5*ce+.5*mse,'ce':ce,'mse':mse}[objective])
        loss.sum().backward()
        assert (logits.grad is not None)==(objective!='mse')
        assert (scalar.grad is not None)==(objective!='ce')


def test_adapter_changes_preserve_output_initialization_and_reload(tmp_path,monkeypatch):
    import copy,json
    from types import SimpleNamespace
    from safetensors.torch import save_file
    import laya_multitask_experiment as exp
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(hidden_size=4);self.embedding=torch.nn.Embedding(12,4)
        def gradient_checkpointing_enable(self,**kwargs):pass
        def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    base=torch.nn.Module();base.encoder=Encoder();base.type_emb=torch.nn.Embedding(3,4)
    base.head=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(4,2,8,dropout=0,batch_first=True),1)
    base.scorer=torch.nn.Sequential(torch.nn.LayerNorm(4),torch.nn.Linear(4,4),torch.nn.GELU(),torch.nn.Linear(4,1))
    spec={'fluorescence':{'primitive':'score','n_outputs':5}}
    torch.manual_seed(7);original=exp.SharedHeads(copy.deepcopy(base),spec)
    torch.manual_seed(7);native=exp.SharedHeads(copy.deepcopy(base),spec,'mean','mse',True,True)
    assert torch.equal(original.classifiers['fluorescence'].weight,native.classifiers['fluorescence'].weight)
    assert torch.equal(original.regressors['fluorescence'].weight,native.regressors['fluorescence'].weight)
    assert native.head is None and isinstance(native.readout,torch.nn.LayerNorm)
    model_dir=tmp_path/'model';model_dir.mkdir();(model_dir/'rl_agent_config.json').write_text('{}')
    monkeypatch.setattr(exp.core,'import_laya',lambda p:(None,None,None))
    monkeypatch.setattr(exp.core,'load_model',lambda *args,**kwargs:copy.deepcopy(base))
    model,cfg=exp.build('shared_heads',model_dir,tmp_path,spec,torch.device('cpu'),pooling='mean',score_loss='mse',bypass_head=True,native_readout=True)
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    save_file(model.state_dict(),str(checkpoint/'model.safetensors'));(checkpoint/'rl_agent_config.json').write_text(json.dumps(cfg))
    reloaded,_=exp.build('shared_heads',model_dir,tmp_path,spec,torch.device('cpu'),checkpoint)
    assert reloaded.pooling=='mean' and reloaded.score_loss=='mse' and reloaded.head is None
    batch={'input_ids':torch.tensor([[1,2,3],[4,5,0]]),'attention_mask':torch.tensor([[1,1,1],[1,1,0]]),'qtype':torch.tensor([1,1]),'task':'fluorescence'}
    model.eval();reloaded.eval()
    for expected,actual in zip(model(**batch),reloaded(**batch)):assert torch.equal(expected,actual)
