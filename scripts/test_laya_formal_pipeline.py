"""CPU checks for metadata/label/eligibility contracts before queued M2 starts."""
from argparse import Namespace
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import laya_formal_experiment as experiment
from laya_representation import Representation


@pytest.fixture(scope='module')
def reps():
    path=ROOT/'artifacts/laya_formal_representation'
    return Representation.load(path,False),Representation.load(path,True)


def test_vocab_ids_and_initialization_decompositions(reps):
    base,expanded=reps
    tokens,decomposition=experiment.expansion_spec(expanded.root,base.tokenizer,expanded.tokenizer)
    assert len(tokens)==64
    assert len(expanded.tokenizer)==len(base.tokenizer)+64
    bv=base.tokenizer.get_vocab(); ev=expanded.tokenizer.get_vocab()
    assert all(ev[token]==idx for token,idx in bv.items())
    for token in tokens:
        assert decomposition[token]==base.tokenizer(token,add_special_tokens=False)['input_ids']
        assert expanded.tokenizer(token,add_special_tokens=False)['input_ids']==[ev[token]]


def test_complete_piece_tokens_do_not_match_longer_source_piece(reps):
    base,expanded=reps
    selected=json.loads((expanded.root/'new_tokens.json').read_text())
    for record in selected:
        token=record['token']; kind=record['modality']; piece=record['piece']
        longer=Representation.wrap_piece(kind,piece+'A')
        assert record['expanded_id'] not in expanded.tokenizer(longer,add_special_tokens=False)['input_ids']
    for kind,sequence in [('dna','ACGTACGTN'),('protein','MAAAELKX')]:
        rec={'sequence':sequence,'modality':kind,'context':'Classify this sequence.'}
        assert base.state(rec)==expanded.state(rec)
        assert ''.join(base.source_pieces(rec)[1])==sequence


def test_candidate_permutation_keeps_gold_semantics(reps):
    qtypes,_,builder=experiment.import_laya('vendor/laya')
    base,_=reps
    rows=experiment.load_split(ROOT/'artifacts/laya_formal_data','fold_class','train')[:7]
    for row in rows:
        items,_=experiment.make_items([row],base.tokenizer,builder,qtypes,1024,256,base.state,True,42)
        item=items[0]; target=item['markers'][item['label']]
        expected=base.tokenizer(' '+row['choices'][row['label']],add_special_tokens=False)['input_ids']
        assert item['ids'][target]==base.tokenizer.mask_token_id
        assert item['ids'][target+1:target+1+len(expected)]==expected


def test_common_eligible_counts_and_no_group_overlap():
    root=ROOT/'artifacts/laya_formal_data'
    ids=experiment.load_eligible_ids(str(root/'eligible_ids.json'))
    assert len(ids)==40111
    groups={s:set() for s in ('train','selection_dev','calibration','test')}
    counts={s:0 for s in groups}
    for task in ('promoter_detection','fold_class'):
        for split in groups:
            for line in (root/f'{task}_{split}.jsonl').read_text().splitlines():
                r=json.loads(line)
                if r['id'] in ids:
                    counts[split]+=1;groups[split].add(r['group_id'])
    assert counts=={'train':32050,'selection_dev':1978,'calibration':1986,'test':4097}
    names=list(groups)
    for i,left in enumerate(names):
        for right in names[i+1:]:assert groups[left].isdisjoint(groups[right])


def test_test_gate_precedes_loading_data(monkeypatch,tmp_path):
    args=Namespace(seed=1,device='cpu',data_dir='does_not_exist',model_dir='does_not_exist',
                   output_dir=str(tmp_path),laya_repo='vendor/laya',eval_splits='test',allow_test=False)
    monkeypatch.setattr(experiment,'parse_args',lambda:args)
    def forbidden(*a,**kw):raise AssertionError('Data was read before test gate')
    monkeypatch.setattr(experiment,'load_split',forbidden)
    with pytest.raises(ValueError,match='explicit --allow-test'):
        experiment.main()
