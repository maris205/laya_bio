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


def test_single_task_replays_joint_examples_and_learning_rate_indices():
    import random
    from collections import Counter
    entities=[{'id':f'{t}:{i}','task':t} for t in d.TASKS for i in range(7)]
    # Independent copy of the previously used joint sampler, including RNG calls.
    rng=random.Random(19);pools={t:[x for x in entities if x['task']==t] for t in d.TASKS}
    for pool in pools.values():rng.shuffle(pool)
    cursor=Counter();cycle=[];expected=[]
    for step in range(1,37):
        if not cycle:cycle=d.TASKS.copy();rng.shuffle(cycle)
        task=cycle.pop();chunk=[];pool=pools[task]
        while len(chunk)<4:
            if cursor[task]>=len(pool):rng.shuffle(pool);cursor[task]=0
            chunk.append(pool[cursor[task]]['id']);cursor[task]+=1
        expected.append((step,task,chunk))
    actual=[(s,t,[r['id'] for r in batch]) for s,t,batch in e.training_schedule(entities,19,36,4)]
    assert actual==expected
    single=[row for row in actual if row[1]=='splice']
    assert single==[row for row in expected if row[1]=='splice'] and len(single)==6
