#!/usr/bin/env python3
"""Build a six-task engineering pilot without reading legacy test targets.

Preserves native development splits, guards legacy test sequence memberships,
and applies global exact/RC plus paired-endpoint components. This is NOT a new
homology-disjoint confirmatory benchmark. Native GFP variants share a parent.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

TASKS = ['promoter', 'splice', 'structural_class', 'homology_pair', 'fluorescence', 'multilabel_location']
LOCATIONS = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 'Mitochondrion', 'Plastid',
             'Endoplasmic reticulum', 'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome']
PAIR_PATTERN = r'(?m)^(?:Sequence [12]:|Protein [AB]:|[AB]:|>seq[12])[ \t]*\n?([A-Z*\-]+)[ \t]*$'


def sha(x):
    return hashlib.sha256(x.encode()).hexdigest()


def entity_key(seq, modality):
    seq = seq.upper()
    if modality == 'dna':
        rc = seq.translate(str.maketrans('ACGTRYSWKMBDHVN', 'TGCAYRSWMKVHDBN'))[::-1]
        seq = min(seq, rc)
    return modality + ':' + sha(seq)


class Groups:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def inputs(seqs, modalities):
    return [{'role': 'sequence' if len(seqs) == 1 else f'protein_{i+1}', 'modality': m, 'sequence': s.upper()}
            for i, (s, m) in enumerate(zip(seqs, modalities))]


def state(row):
    return '\n'.join(f"{x['role']} ({x['modality']}): {x['sequence']}" for x in row['inputs'])


def questions(row):
    """One record per proposition, canonical output coordinates, entity weights."""
    common = {'entity_id': row['id'], 'task': row['task'], 'state': state(row), 'group': row['group']}
    if row['primitive'] == 'multilabel':
        return [{**common, 'id': row['id'] + ':' + str(i), 'primitive': 'noul',
                 'question': f"Is this eukaryotic protein localized to {name}?", 'choices': ['false', 'true'],
                 'label': int(row['labels'][i]), 'label_index': i, 'weight': 1 / len(LOCATIONS)}
                for i, name in enumerate(LOCATIONS)]
    out = {**common, 'id': row['id'], 'primitive': row['primitive'], 'question': row['question'],
           'choices': row['choices'], 'label': row['label'], 'weight': 1.0}
    if 'value' in row: out.update(value=row['value'], anchors=row.get('anchors', list(range(len(row['choices'])))))
    return [out]


def rubric(item):
    t = item['primitive']
    if t == 'choice': crit = dict.fromkeys(item['choices'])
    elif t == 'score': crit = item['choices']
    else: crit = {'false': 'No, the statement does not hold.', 'true': 'Yes, the statement holds.'}
    return {'t': t, 'ins': item['question'], 'crit': crit}


def fit_score(rows):
    values = np.asarray([r['value'] for r in rows], dtype=float)
    cuts = np.unique(np.quantile(values, [.2, .4, .6, .8]))
    cuts = cuts[(cuts > values.min()) & (cuts < values.max())]
    bins = np.searchsorted(cuts, values, side='right')
    anchors = [float(values[bins == i].mean()) for i in range(len(cuts) + 1)]
    if not np.isfinite(anchors).all(): raise ValueError('Empty score bin')
    return {'thresholds': cuts.tolist(), 'anchors': anchors, 'train_min': float(values.min()),
            'train_max': float(values.max()), 'fit_entities': len(values)}


def apply_score(row, spec):
    row['label'] = int(np.searchsorted(spec['thresholds'], row['value'], side='right'))
    row['anchors'] = spec['anchors']
    edges = [-float('inf')] + spec['thresholds'] + [float('inf')]
    row['choices'] = [f"log fluorescence in [{edges[i]:.5g}, {edges[i+1]:.5g})" for i in range(len(spec['anchors']))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--laya-repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-cap', type=int, default=1024)
    p.add_argument('--dev-cap', type=int, default=128)
    p.add_argument('--max-length', type=int, default=1024)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    sys.path.insert(0, str(a.laya_repo)); from laya.common import build_sequence
    tok = AutoTokenizer.from_pretrained(a.model_dir / 'tokenizer')
    legacy = a.workspace / 'data/03_sft_biopaws2/jsonl'
    source = a.workspace / 'data/05_task_extension_sources'
    groups = Groups(); reserved = set(); source_hashes = {}; rows = []
    def remember(path): source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    def register(seq_inputs, protected=False):
        keys = [entity_key(x['sequence'], x['modality']) for x in seq_inputs]
        for k in keys: groups.union(keys[0], k)
        if protected: reserved.update(keys)
        return keys[0]
    # Read only input and split fields of protected legacy records. Include all
    # eight single-sequence source views and both pair views in the global guard.
    for path in sorted(legacy.glob('*.jsonl')):
        if path.stem not in {'lg_promoter_detection','lg_fold_class','lg_signal_peptide','lg_npp',
                             'lg_subcellular_loc','lg_splice_site','lg_tf_prediction','lg_core_promoter_detection',
                             'protein_homology_std','protein_homology_remote'}: continue
        remember(path)
        for line in path.open():
            r = json.loads(line); u = next(m['content'] for m in r['messages'] if m['role'] == 'user')
            if path.stem.startswith('protein_homology'):
                seqs = re.findall(PAIR_PATTERN, u)
                if len(seqs) != 2: raise ValueError('Pair parse failure: ' + r['id'])
                inp = inputs(seqs, ['protein','protein'])
            else:
                kind = 'dna' if 'dna' in r['modality'] else 'protein'
                inp = inputs([u.strip().splitlines()[-1].strip()], [kind])
            key = register(inp, r['split'] == 'test')
            if r['split'] == 'test': continue
            task = {'lg_promoter_detection':'promoter', 'lg_fold_class':'structural_class',
                    'protein_homology_std':'homology_pair'}.get(path.stem)
            if task is None: continue
            out = {'id': r['id'], 'task': task, 'inputs': inp, 'group_key': key,
                   'split': 'dev' if r['split'] == 'val' else 'train', 'source_split':r['split']}
            if task == 'promoter':
                out.update(primitive='noul', question='Is this DNA sequence a promoter?', choices=['false','true'], label=int(r['answer_short'] == 'promoter'))
            elif task == 'homology_pair':
                out.update(primitive='noul', question='Are these two proteins homologous (sharing evolutionary ancestry)?', choices=['false','true'], label=int(r['answer_short'] == 'Yes'))
            else:
                out.update(primitive='choice', question='Which structural class describes this protein?', choices=r['choices'], label=r['choices'].index(r['answer_short']))
            rows.append(out)
    path = source / 'dna_splice_site_prediction/data/train-00000-of-00001.parquet'; remember(path)
    for i,r in enumerate(pq.read_table(path).to_pylist()):
        inp = inputs([r['sequence']], ['dna']); key = register(inp)
        split = 'dev' if int(sha('splice-dev-v1:' + key)[:8],16) % 10 == 0 else 'train'
        rows.append({'id':f'splice-upstream:{i}', 'task':'splice', 'inputs':inp,'group_key':key,'split':split,
                     'source_split':'unpartitioned_source_pool', 'primitive':'choice',
                     'question':'Which splice-site category describes this DNA sequence?',
                     'choices':['Non-splice site','Acceptor site','Donor site'], 'label':int(r['label'])})
    for native,split in [('train','train'),('valid','dev')]:
        path = source / f'tape/fluorescence/fluorescence_{native}.json'; remember(path)
        for i,r in enumerate(json.loads(path.read_text())):
            inp = inputs([r['primary']], ['protein']); key = register(inp)
            rows.append({'id':f'fluorescence:{native}:{i}', 'task':'fluorescence', 'inputs':inp,'group_key':key,
                         'split':split,'source_split':native, 'primitive':'score',
                         'question':'What is the log-fluorescence level of this GFP variant?',
                         'value':float(r['log_fluorescence'][0]), 'label':0,
                         'choices':['very low','low','medium','high','very high']})
    path=source/'deeploc2/Swissprot_Train_Validation_dataset.csv'; remember(path)
    for r in csv.DictReader(path.open()):
        partition=int(r['Partition']);inp=inputs([r['Sequence']],['protein']);key=register(inp,partition==4)
        if partition==4: continue # held aside; no calibration/target use in pilot
        labels=[float(r[k]) for k in LOCATIONS]
        if not all(x in (0,1) for x in labels):raise ValueError('Unexpected multilabel targets')
        rows.append({'id':'deeploc2:'+r['ACC'],'task':'multilabel_location','inputs':inp,'group_key':key,
                     'split':'dev' if partition==3 else 'train','source_split':'partition_'+str(partition),
                     'primitive':'multilabel','labels':labels})
    protected={groups.find(k) for k in reserved}; by_group=defaultdict(set);conflicts=defaultdict(set)
    for r in rows:
        r['group']=groups.find(r.pop('group_key'));by_group[r['group']].add(r['split'])
        # Opposite pair orientation describes the same relation. Groups can
        # contain many pairs, so conflict keys use the full unordered pair.
        pairkey='|'.join(sorted(entity_key(x['sequence'],x['modality']) for x in r['inputs']))
        target=r.get('labels',r.get('value',r.get('label')))
        conflicts[(r['task'],pairkey)].add(json.dumps(target))
        r['_conflict_key']=(r['task'],pairkey)
    excluded=Counter(); retained=[]; seen=set()
    for r in rows:
        reason=None
        if r['group'] in protected: reason='protected_legacy_test_or_reserved_partition_component'
        elif r['split']=='train' and 'dev' in by_group[r['group']]: reason='component_in_dev'
        elif len(conflicts[r['_conflict_key']])>1: reason='conflicting_annotations'
        elif (r['task'],r['_conflict_key'][1]) in seen: reason='duplicate_entity_or_pair'
        if reason:excluded[r['task']+':'+reason]+=1;continue
        seen.add((r['task'],r['_conflict_key'][1]));r.pop('_conflict_key')
        full=[]
        for item in questions(r):
            ids,marks=build_sequence(tok,item['state'],rubric(item),max_len=1000000,head_max_len=256)
            if len(marks)!=len(item['choices']):raise ValueError('Missing candidate marker')
            full.append(len(ids))
        if max(full)>a.max_length:excluded[r['task']+':full_input_over_budget']+=1;continue
        r['max_input_tokens']=max(full);retained.append(r)
    sets={}; availability={}
    for split,cap in [('train',a.train_cap),('dev',a.dev_cap)]:
        sets[split]=[];availability[split]={}
        for task in TASKS:
            pool=[r for r in retained if r['task']==task and r['split']==split]
            availability[split][task]=len(pool)
            pool.sort(key=lambda r:sha('pilot-cap-v1:'+r['id']))
            if len(pool)<cap:raise ValueError(f'Insufficient {split} {task}: {len(pool)} < {cap}')
            sets[split].extend(pool[:cap])
    score_spec=fit_score([r for r in sets['train'] if r['primitive']=='score'])
    for values in sets.values():
        for r in values:
            if r['primitive']=='score':apply_score(r,score_spec)
            # Recheck actual numerical rubric, not only provisional level names.
            for item in questions(r):
                ids,_=build_sequence(tok,item['state'],rubric(item),max_len=1000000,head_max_len=256)
                if len(ids)>a.max_length:raise ValueError('Final Score rubric exceeds context')
    tr={r['group'] for r in sets['train']};de={r['group'] for r in sets['dev']}
    assert not tr & de and not (tr|de) & protected
    a.output.mkdir(parents=True)
    outputs={}
    for split,values in sets.items():
        path=a.output/(split+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in values))
        outputs[split]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'entities':len(values),'by_task':dict(Counter(r['task'] for r in values)),
                        'propositions':sum(len(questions(r)) for r in values)}
    manifest={'status':'engineering_pilot_not_confirmatory','tasks':TASKS,'outputs':outputs,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_sha256':source_hashes,'max_length':a.max_length,'available_before_caps':availability,
              'excluded':dict(excluded),'score_spec':score_spec,'locations':LOCATIONS,
              'legacy_test_targets_used':False,'legacy_test_model_evaluation':False,
              'train_dev_exact_component_overlap':0,'protected_test_component_overlap':0,
              'native_splits':{'fluorescence':'train/valid; test not read',
                               'deeploc2':'partitions 0/1/2 train, 3 dev, 4 held aside'},
              'limitations':['No new global homology clustering or genomic-coordinate split.',
                             'GFP native variant split intentionally shares a parent protein.',
                             'Limited source provenance for local paired homology.',
                             'Only Laya full-input eligibility checked; not a common generator subset yet.',
                             'Dev outcomes are engineering diagnostics, not held-out paper test results.']}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','outputs','available_before_caps','excluded','score_spec']},indent=2),flush=True)

if __name__=='__main__':main()
