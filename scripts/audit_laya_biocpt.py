#!/usr/bin/env python3
"""Independent artifact audit for the first biological CPT comparison."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,log_loss,recall_score
from safetensors import safe_open
from transformers import AutoTokenizer
from laya_biocpt_data import sha,read_lines,write_json
from laya_biocpt_train import Masker,Stream,SEED


def load(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    root=a.root/'round'
    data=a.root/'data'
    assert load(root/'status.json')['status']=='complete'
    manifest=load(root/'manifest.json')
    checked=[]
    for name,digest in manifest['code_sha256'].items():
        path=root/'frozen_code'/name
        assert sha(path)==digest
        checked.append(str(path))
    assert sha(data/'manifest.json')==manifest['data_manifest_sha256']
    dm=load(data/'manifest.json')
    assert dm['residue_composition_audit']['status']=='pass'
    composition={m:Counter() for m in ['dna','protein']}
    with (data/'raw_train.jsonl').open() as raw:
        for line in raw:
            row=json.loads(line)
            if row['modality'] in composition:
                composition[row['modality']].update(row['content'])
    assert set('ACDEFGHIKLMNPQRSTVWY')<=set(composition['protein'])
    for modality,counts in composition.items():
        assert dict(counts)==dm['residue_composition_audit']['character_counts'][modality]
    for name,digest in dm['output_sha256'].items():
        path=data/name
        assert sha(path)==digest
        checked.append(str(path))
    ct=read_lines(root/'cpt/training_trace.jsonl')
    assert [r['step'] for r in ct]==list(range(1,1153))
    assert sum(r['phase']=='warmup' for r in ct)==128
    assert all(math.isfinite(r['loss']) and r['new_embedding_gradient_norm']>0 and r['mlm_head_gradient_norm']>0 for r in ct)
    assert all(r['encoder_gradient_norm']==0 for r in ct[:128])
    assert all(r['encoder_gradient_norm']>0 for r in ct[128:])
    warm=load(root/'cpt/warmup_embedding_check.json')
    assert warm['old_rows_equal'] and warm['old_row_max_change']==0
    entries=load(data/'representation/new_tokens.json')
    tokenizer=AutoTokenizer.from_pretrained(data/'representation/base_tokenizer')
    masker=Masker(tokenizer,entries)
    rows=read_lines(data/'cpt_train.jsonl')
    val=read_lines(data/'cpt_validation.jsonl')
    assert len({r['id'] for r in rows})==len(rows)
    assert not {r['id'] for r in rows}&{r['id'] for r in val}
    stream=Stream(rows)
    rng=random.Random(SEED+99)
    seen,targets,presentations=Counter(),Counter(),Counter()
    visible_natural,random_replacement=Counter(),Counter()
    input_tokens=0
    for step in range(1152):
        rr=stream.batch()
        presentations.update(r['modality'] for r in rr)
        input_tokens+=sum(len(r['ids']) for r in rr)
        inputs,_,labels,s,t=masker(rr,rng,device='cpu')
        assert int((labels!=-100).sum())==ct[step]['masked_targets']
        for row,corrupted in zip(rr,inputs.tolist()):
            for original,actual in zip(row['ids'],corrupted):
                if actual>=dm['old_vocab_size']:
                    (visible_natural if actual==original else random_replacement)[actual]+=1
        seen.update(s)
        targets.update(t)
    for e in load(root/'cpt/token_training_audit.json'):
        assert seen[e['expanded_id']]==e['observed_input_count']
        assert targets[e['expanded_id']]==e['masked_target_count']
        assert math.isfinite(e['weight_delta_l2']) and e['weight_delta_l2']>0
    assert dict(presentations)=={'dna':16128,'protein':16128,'text':4608}
    exposure=[{'expanded_id':e['expanded_id'],'token':e['token'],
               'natural_occurrences_before_masking':seen[e['expanded_id']],
               'natural_occurrences_visible_after_masking':visible_natural[e['expanded_id']],
               'random_replacement_input_occurrences':random_replacement[e['expanded_id']],
               'masked_target_occurrences':targets[e['expanded_id']]} for e in entries]
    # Every published train/dev metric is independently recomputed with sklearn.
    metric_checks=0
    for name in ['no_cpt_4096','cpt_4096','no_cpt_full','cpt_full']:
        folder=root/name
        cfg=load(folder/'run_config.json')
        status=load(folder/'status.json')
        assert status['status']=='complete' and status['checkpoint_reload_exact']
        trace=read_lines(folder/'training_trace.jsonl')
        expected=192 if name.endswith('4096') else 786
        assert cfg['updates']==expected==len(trace)
        assert [r['step'] for r in trace]==list(range(1,expected+1))
        assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace)
        if name.endswith('full'):
            assert sha(folder/'model.safetensors')==status['checkpoint_sha256']
        for point in read_lines(folder/'learning_curve.jsonl'):
            pred=read_lines(folder/f"{point['split']}_epoch{point['epoch']}_predictions.jsonl")
            size='4096' if name.endswith('4096') else 'full'
            source=read_lines(data/('sft_dev.jsonl' if point['split']=='dev' else 'sft_train_'+size+'.jsonl'))
            assert len(pred)==len(source)
            assert [(r['id'],r['label'],r['group_id']) for r in pred]==[(r['id'],r['label'],r['group_id']) for r in source]
            y=[r['label'] for r in pred]
            yh=[r['prediction'] for r in pred]
            probs=np.array([r['probs'] for r in pred])
            assert np.array_equal(probs.argmax(-1),yh)
            np.testing.assert_allclose(probs.sum(1),1,rtol=0,atol=1e-6)
            np.testing.assert_allclose([accuracy_score(y,yh),f1_score(y,yh,average='macro'),log_loss(y,probs,labels=[0,1])],[point['accuracy'],point['macro_f1'],point['nll']],atol=1e-7,rtol=1e-6)
            assert confusion_matrix(y,yh,labels=[0,1]).tolist()==point['confusion']
            np.testing.assert_allclose(recall_score(y,yh,labels=[0,1],average=None),point['per_class_recall'])
            metric_checks+=1
    for size in ['4096','full']:
        left=load(root/('no_cpt_'+size)/'run_config.json')
        right=load(root/('cpt_'+size)/'run_config.json')
        for field in ['initial_classifier_sha256','batch_order_sha256','seed','epochs','updates','samples','data_manifest_sha256']:
            assert left[field]==right[field]
    for arm in ['no_cpt','cpt']:
        assert sha(root/(arm+'_4096')/'dev_epoch0_predictions.jsonl')==sha(root/(arm+'_full')/'dev_epoch0_predictions.jsonl')
    cptstatus=load(root/'cpt/status.json')
    assert sha(root/'cpt/model.safetensors')==cptstatus['checkpoint_sha256']
    assert (root/'cpt/resume.pt').is_file()
    resume=torch.load(root/'cpt/resume.pt',map_location='cpu',weights_only=False,mmap=True)
    required_rng={'torch_rng','cuda_rng','python_rng','numpy_rng','mask_rng','stream'}
    assert required_rng.issubset(resume)
    assert resume['step']==1152 and resume['phase']=='full'
    assert resume['stream']==stream.state() and resume['mask_rng']==rng.getstate()
    assert {float(v['step']) for v in resume['optimizer']['state'].values()}=={1024.}
    assert len(resume['cuda_rng'])==1
    original_path=a.root.parent/'laya_model/model.safetensors'
    with safe_open(str(original_path),framework='pt',device='cpu') as original, safe_open(str(root/'cpt/model.safetensors'),framework='pt',device='cpu') as adapted:
        change=float((original.get_tensor('encoder.layers.0.mlp.Wi.weight').float()-adapted.get_tensor('model.layers.0.mlp.Wi.weight')).abs().max())
        assert change>0
    result={'status':'pass','hashed_files_checked':len(checked)+1,'CPT_updates_replayed':1152,'CPT_actual_presentations':dict(presentations),'CPT_actual_input_tokens_including_prefix_special':input_tokens,'CPT_actual_masked_targets':sum(r['masked_targets'] for r in ct),'CPT_token_input_and_target_counts_exact':True,'natural_visible_input_token_coverage':sum(v['natural_occurrences_visible_after_masking']>0 for v in exposure),'tokens_without_natural_visible_input':[v['token'] for v in exposure if not v['natural_occurrences_visible_after_masking']],'old_rows_exactly_invariant_during_warmup':True,'SFT_metric_points_independently_recomputed':metric_checks,'all_retained_checkpoint_hashes_verified':True,'paired_classifier_initialization_and_batch_order_match':True,'test_inference':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result['encoder_first_MLP_matrix_max_change']=change
    result['canonical_residue_composition_independently_recomputed']=True
    result['protein_N_characters_in_admitted_training_snapshot']=composition['protein']['N']
    result['resume_sampler_and_mask_RNG_match_exact_replay']=True
    result['resume_optimizer_parameter_states']=len(resume['optimizer']['state'])
    result['resume_optimizer_full_phase_steps']=1024
    result['same_arm_initial_dev_predictions_identical_across_data_sizes']=True
    by_modality={m:sum(targets[e['expanded_id']] for e in entries if e['modality']==m) for m in ['dna','protein']}
    by_modality['text']=result['CPT_actual_masked_targets']-sum(by_modality.values())
    result['CPT_actual_masked_targets_by_modality']=by_modality
    write_json(a.output.with_name('actual_token_exposure.json'),exposure)
    write_json(a.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
