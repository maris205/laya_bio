"""Regression checks for lossless historical BPE reuse and label integrity."""
from pathlib import Path
import json
import sys

import pytest
from tokenizers import Tokenizer

sys.path.insert(0,str(Path(__file__).resolve().parent))
import laya_formal_experiment as core
from laya_direct_bpe import DirectBPE
from laya_direct_experiment import make_items


@pytest.fixture(scope='module')
def rep():
    return DirectBPE()


def test_old_missing_n_regression_and_lossless_repair(rep):
    old = Tokenizer.from_file(str(rep.root/'original_source_tokenizers/protein_bpe_8k.json'))
    assert ''.join(old.encode('MNA').tokens) != 'MNA'
    for kind,sequence in [('protein','MNA'),('protein','ANNAJXUOBZ'),('dna','ACGTNNN')]:
        ids = rep.sequence_ids(sequence,kind)
        assert ''.join(rep.inverse[i] for i in ids) == sequence


def test_original_merges_and_ids_unchanged(rep):
    for name in ['dna_bpe_20k.json','protein_bpe_8k.json']:
        old=json.loads((rep.root/'original_source_tokenizers'/name).read_text())['model']
        new=json.loads((rep.root/'source_tokenizers'/name).read_text())['model']
        assert old['merges']==new['merges']
        assert all(new['vocab'][piece]==idx for piece,idx in old['vocab'].items())
    assert rep.meta['alphabet_repair']=={'dna':['N'],'protein':['J','N']}


def test_initialization_uses_raw_piece_without_metadata_markers(rep):
    for e in rep.entries:
        assert e['base_ids']==rep.base(e['piece'],add_special_tokens=False)['input_ids']
    expanded_vocab=rep.expanded.get_vocab()
    assert all(expanded_vocab[s]==i for s,i in rep.base.get_vocab().items())


def test_raw_equivalence_and_candidate_label_permutation(rep):
    _,_,builder=core.import_laya('vendor/laya')
    for task in ('fold_class','promoter_detection'):
        rows=core.load_split(core.ROOT/'artifacts/laya_formal_data',task,'train')[:3]
        raw=make_items(rows,rep,'raw',builder,42,True)
        full=make_items(rows,rep,'full_bpe',builder,42,True)
        for row,x,y in zip(rows,raw,full):
            assert x['label']==y['label']
            assert x['markers']==y['markers']
            for item in (x,y):
                pos=item['markers'][item['label']]
                answer=rep.base(' '+row['choices'][row['label']],add_special_tokens=False)['input_ids']
                assert item['ids'][pos+1:pos+1+len(answer)]==answer
            q={'t':'choice','ins':'Choose the correct biological label for the sequence.',
               'crit':{c:None for c in row['choices']}}
            ids,markers=rep.build(row,'raw',builder,list(range(len(row['choices']))))
            expected=builder(rep.base,core.raw_state(row),q,max_len=1024,head_max_len=256)
            assert (ids,markers)==expected


def test_natural_language_is_always_base_encoded_and_modalities_disjoint(rep):
    _,_,builder=core.import_laya('vendor/laya')
    row=core.load_split(core.ROOT/'artifacts/laya_formal_data','promoter_detection','train')[0]
    row=dict(row,context='Natural text with <BIO_DNA:A> and protein ACGT.')
    ids,markers=rep.build(row,'full_bpe',builder,[0,1])
    seq=rep.sequence_ids(row['sequence'],'dna')
    assert ids[-len(seq)-1:-1]==seq
    assert all(i<len(rep.base) for i in ids[:-len(seq)-1])
    assert set(rep.maps['dna'].values()).isdisjoint(rep.maps['protein'].values())


def test_overlength_input_is_rejected(rep):
    _,_,builder=core.import_laya('vendor/laya')
    row=core.load_split(core.ROOT/'artifacts/laya_formal_data','promoter_detection','train')[0]
    with pytest.raises(ValueError,match='exceeds'):
        rep.build(row,'full_bpe',builder,[0,1],max_len=20)
