"""Regression checks for typed batches, panel weighting and paired membership."""
import numpy as np
import torch
import laya_multitask_data as d
import laya_multitask_experiment as e


def test_real_type_ids_survive_collation():
    items=[{'ids':[1,2,3], 'qtype':i} for i in [0,1,2]]
    batch=e.collate(items,0,torch.device('cpu'))
    assert batch['qtype'].tolist()==[0,1,2]
    assert batch['attention_mask'].sum().item()==9


def test_multilabel_is_marginal_and_entity_normalized():
    row={'id':'protein','task':'multilabel_location','primitive':'multilabel','group':'g',
         'inputs':d.inputs(['MABCDE'],['protein']),'labels':[1,1,0,0,0,0,0,0,0,0]}
    items=d.questions(row)
    assert sum(x['label'] for x in items)==2
    assert all(x['primitive']=='noul' and x['choices']==['false','true'] for x in items)
    assert np.isclose(sum(x['weight'] for x in items),1)
    logits=torch.tensor([[.3,.7]]*10,requires_grad=True)
    y=torch.tensor([x['label'] for x in items]);weights=torch.tensor([x['weight'] for x in items])
    weighted=(torch.nn.functional.cross_entropy(logits,y,reduction='none')*weights).sum()
    reference=torch.nn.functional.cross_entropy(logits,y)
    assert torch.allclose(weighted,reference)


def test_endpoint_components_and_reverse_complements():
    assert d.entity_key('AAGC','dna')==d.entity_key('GCTT','dna')
    assert d.entity_key('AAGC','protein')!=d.entity_key('GCTT','protein')
    groups=d.Groups();groups.union('a','b');groups.union('c','b')
    assert groups.find('a')==groups.find('c')
    assert groups.find('a')!=groups.find('x')
    state=d.state({'inputs':d.inputs(['MABC','MXYZ'],['protein','protein'])})
    assert 'protein_1 (protein): MABC' in state and 'protein_2 (protein): MXYZ' in state


def test_score_train_anchors_and_extreme_coverage():
    spec=d.fit_score([{'value':float(i)} for i in range(100)])
    assert spec['fit_entities']==100 and len(spec['anchors'])==5
    row={'value':1000.};d.apply_score(row,spec)
    assert row['label']==4 and row['anchors'][-1]<1000
    assert np.all(np.diff(row['anchors'])>0)


def test_multilabel_metrics_use_all_labels_and_report_missing_support():
    rows=[]
    for n in range(3):
        for i in range(10):
            label=int(i in [0,n+1]);p=.9 if label else .1
            rows.append({'entity_id':str(n),'id':f'{n}:{i}','task':'multilabel_location',
                         'label_index':i,'label':label,'probs':[1-p,p]})
    m=e.metrics(rows,{'multilabel_location':{'primitive':'multilabel','n_outputs':10}})['multilabel_location']
    assert m['micro_auprc']==1 and m['micro_f1_at_0_5']==1
    assert m['n_entities']==3 and m['macro_auprc_label_count']==3
