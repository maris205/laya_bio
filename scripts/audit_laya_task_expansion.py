#!/usr/bin/env python3
"""Read-only task admission audit. No model calls, split changes or test metrics.

Test targets are not inspected. Sequence membership, schema, and provenance are
summarized; exact/RC groups are not a substitute for protein homology clusters.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

TASKS = {
    'promoter_detection': ('dna', 'joint_train', ['Non-promoter', 'promoter']),
    'fold_class': ('protein', 'joint_train', ['All Alpha', 'All Beta', 'Alpha and Beta', 'Alpha plus Beta', 'Multi-domain Proteins', 'Mixed Structures', 'Small Proteins and Peptides']),
    'signal_peptide': ('protein', 'joint_train', ['sec', 'tat']),
    'npp': ('protein', 'joint_train', ['non-NPP', 'NPP']),
    'subcellular_loc': ('protein', 'joint_train', ['CYtoplasmicMembrane', 'Cellwall', 'Cytoplasmic', 'Extracellular', 'OuterMembrane', 'Periplasmic']),
    'splice_site': ('dna', 'held_out_task', ['Non-Splice Sites', 'Acceptor Sites', 'Donor Sites']),
    'tf_prediction': ('dna', 'held_out_task_conditional_on_label_provenance', ['Background Sequences', 'Binding Sites']),
    'core_promoter_detection': ('dna', 'related_domain_transfer_not_new_task_family', ['Non-promoter', 'promoter']),
}

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def length_stats(values):
    a = sorted(values)
    if not a:
        return {'n': 0}
    return {'n': len(a), 'min': a[0], 'median': a[(len(a)-1)//2],
            'p95': a[int((len(a)-1)*.95)], 'max': a[-1],
            'over_512_characters': sum(x > 512 for x in a)}

def audit(root):
    reports = []
    entities = defaultdict(list)
    task_groups = {}
    for task, (modality, role, labels) in TASKS.items():
        path = root / f'lg_{task}.jsonl'
        file_hash = hashlib.sha256()
        counts, splits, errors, templates = (Counter() for _ in range(4))
        lengths = defaultdict(list)
        sources, licenses = Counter(), Counter()
        ids = set()
        split_groups = defaultdict(set)
        trainval_labels = defaultdict(set)
        label_counts = defaultdict(Counter)
        with path.open('rb') as f:
            for raw in f:
                file_hash.update(raw)
                row = json.loads(raw)
                counts['rows'] += 1
                split = row.get('split')
                splits[split] += 1
                sources[str(row.get('source'))] += 1
                licenses[str(row.get('license'))] += 1
                errors['duplicate_id_rows'] += row.get('id') in ids
                ids.add(row.get('id'))
                errors['unexpected_task_id'] += row.get('task_id') != task
                errors['unexpected_choices'] += row.get('choices') != labels
                errors['unexpected_split'] += split not in {'train', 'val', 'test'}
                users = [m.get('content') for m in row.get('messages', []) if m.get('role') == 'user']
                if len(users) != 1 or not isinstance(users[0], str):
                    errors['input_schema_failure'] += 1
                    continue
                lines = [line.strip() for line in users[0].splitlines() if line.strip()]
                if len(lines) != 2:
                    errors['not_exactly_instruction_and_sequence_lines'] += 1
                    continue
                seq = lines[1].upper()
                allowed = 'ACGTRYSWKMBDHVN-' if modality == 'dna' else 'ACDEFGHIKLMNPQRSTVWYBXZJUO*-'
                if not seq or not set(seq) <= set(allowed):
                    errors['sequence_alphabet_failure'] += 1
                    continue
                counts['valid_user_sequence_rows'] += 1
                lengths[split].append(len(seq))
                templates[digest(lines[0])] += 1
                if modality == 'dna':
                    rc = seq.translate(str.maketrans('ACGTRYSWKMBDHVN-', 'TGCAYRSWMKVHDBN-'))[::-1]
                    seq = min(seq, rc)
                key = modality + ':' + digest(seq)
                split_groups[split].add(key)
                entities[key].append((task, split))
                if split in {'train', 'val'}:
                    label = row.get('answer_short')
                    errors['trainval_gold_not_in_choices'] += label not in labels
                    trainval_labels[key].add(label)
                    label_counts[split][str(label)] += 1
        reports.append({'task': task, 'modality': modality, 'proposed_role': role,
                        'source_file': path.name, 'source_sha256': file_hash.hexdigest(),
                        'classes': labels, 'counts': dict(counts), 'raw_splits': dict(splits),
                        'errors': {k: v for k, v in errors.items() if v},
                        'sequence_character_lengths': {k: length_stats(v) for k, v in lengths.items()},
                        'trainval_label_counts': {k: dict(v) for k, v in label_counts.items()},
                        'trainval_conflicting_exact_rc_groups': sum(len(v)>1 for v in trainval_labels.values()),
                        'instruction_template_count': len(templates),
                        'within_task_cross_split_exact_rc_groups': {
                            f'{a}:{b}': len(split_groups[a] & split_groups[b])
                            for a,b in itertools.combinations(('train','val','test'),2)},
                        'sources': dict(sources), 'source_license_declarations': dict(licenses)})
        task_groups[task] = split_groups
    pair_counts = Counter()
    leaks = Counter()
    for members in entities.values():
        unique = set(members)
        tasks = sorted({t for t, s in unique})
        for a, b in itertools.combinations(tasks,2):
            pair_counts[f'{a}:{b}'] += 1
        for a, sa in unique:
            if sa not in {'train','val'}: continue
            for b, sb in unique:
                if a != b and sb == 'test': leaks[f'{a}/{sa} -> {b}/test'] += 1
    joint = [t for t,v in TASKS.items() if v[1] == 'joint_train']
    seen = set().union(*(task_groups[t]['train'] | task_groups[t]['val'] for t in joint))
    heldout_overlap = {t: len(task_groups[t]['test'] & seen) for t,v in TASKS.items() if v[1] != 'joint_train'}
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'admission_audit_only_not_a_frozen_training_split',
            'test_targets_inspected': False, 'model_evaluation_performed': False,
            'task_count': len(TASKS), 'proposed_joint_training_tasks': joint,
            'reports': reports, 'cross_task_exact_rc_shared_groups': dict(pair_counts),
            'cross_task_trainval_to_test_group_overlap': dict(leaks),
            'transfer_test_groups_overlapping_joint_trainval': heldout_overlap,
            'limitations': ['Raw counts precede deduplication, conflicts, homology and tokenizer eligibility.',
                           'No homology clustering or full-input token-length audit in this script.',
                           'NPP ontology and TF binding context require upstream semantic verification.',
                           'Core promoter is related-domain transfer, not an independent new task family.',
                           'Test-only tasks are not converted into supervised training data.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--jsonl-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.jsonl_root);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'tasks':result['task_count'], 'rows':sum(x['counts']['rows'] for x in result['reports']),
                      'errors':{x['task']:x['errors'] for x in result['reports'] if x['errors']},
                      'transfer_overlap':result['transfer_test_groups_overlapping_joint_trainval']},ensure_ascii=False))

if __name__ == '__main__':
    main()
