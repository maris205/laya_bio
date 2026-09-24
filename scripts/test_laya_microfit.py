"""Checks for training-only subset membership and permutation-safe diagnostics."""
import copy
import laya_microfit as m


def test_subset_membership_balance_and_order_invariance():
    rows=[{'id':str(i),'task':'splice','label':i%3,'split':'train'} for i in range(150)]
    selected=m.select_training_subset(rows,'splice')
    reverse=m.select_training_subset(rows[::-1],'splice')
    assert selected==reverse and len(selected)==32
    assert sorted(m.Counter(r['label'] for r in selected).values())==[10,11,11]
    assert {r['id'] for r in selected}<={r['id'] for r in rows}


def test_permuted_predictions_return_to_biological_class_coordinates():
    # Correct answer is canonical class 2, placed at position 0 during training.
    rows=[{'id':'x','label':0,'probs':[.8,.15,.05]}]
    entities=[{'items':[{'id':'x','order':[2,0,1]}]}]
    before=copy.deepcopy(rows)
    result=m.canonicalize(rows,entities,'candidate')
    assert result[0]['label']==2 and result[0]['probs']==[.15,.05,.8]
    assert rows==before


def test_majority_predictions_do_not_pass_fit_threshold():
    assert not m.fit_passed({'accuracy':.375,'nll':1.0})
    assert not m.fit_passed({'accuracy':1.,'nll':.5})
    assert m.fit_passed({'accuracy':31/32,'nll':.1})


def test_order_augmentation_preserves_examples_and_gold_semantics():
    import laya_multitask_data as d
    import laya_multitask_experiment as e
    rows=[{'id':str(i),'task':'splice','primitive':'choice','group':str(i),
           'inputs':d.inputs(['ACGT'],['dna']),'choices':['A','B','C'],
           'question':'Choose a class','label':i%3} for i in range(12)]
    def builder(tok,state,rubric,**kw):
        return [0,1,2,3],[1,2,3]
    entities=e.make_items(rows,None,builder,'candidate',7,True)
    original=copy.deepcopy(entities);orders=[]
    for step in range(1,9):
        out=e.resample_choice_batch(entities,{r['id']:r for r in rows},None,builder,7,step,1024)
        assert [x['id'] for x in out]==[x['id'] for x in entities]
        for entity,row in zip(out,rows):
            item=entity['items'][0]
            assert item['choices'][item['order'][item['label']]]==row['choices'][row['label']]
            assert item['qtype']==0
        orders.append(tuple(out[0]['items'][0]['order']))
    assert len(set(orders))>1 and entities==original
