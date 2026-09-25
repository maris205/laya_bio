#!/usr/bin/env python3
"""Prepare fresh biological vocabulary, filtered CPT samples and full-task SFT data."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, AddedToken
from transformers import AutoTokenizer

SEED = 20260925


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_lines(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')


def read_lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def rc(seq):
    return seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]


def kmers(seq, k):
    return (seq[i:i+k] for i in range(max(0, len(seq)-k+1)))


def key(seq, modality):
    return digest(min(seq, rc(seq)) if modality == 'dna' else seq)


def add_guard(bank, seq, modality):
    k = 31 if modality == 'dna' else 15
    bank.update(kmers(seq, k))
    if modality == 'dna':
        bank.update(kmers(rc(seq), k))


def hits_guard(bank, seq, modality):
    k = 31 if modality == 'dna' else 15
    return any(x in bank for x in kmers(seq, k))


def balanced_subset(rows, count):
    labels = sorted({r['label'] for r in rows})
    chosen = []
    for i, label in enumerate(labels):
        pool = sorted((r for r in rows if r['label'] == label), key=lambda r: digest('biocpt-sft-v1:' + r['id']))
        quota = count // len(labels) + int(i < count % len(labels))
        assert len(pool) >= quota
        chosen.extend(pool[:quota])
    return sorted(chosen, key=lambda r: digest('biocpt-order-v1:' + r['id']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--protein-file', default='protein_lucaone_15g.txt',
                   help='Protein source admitted by canonical-amino-acid composition checks')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    state = {'status': 'preparing', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'test_targets_used': False, 'test_inference': False, 'heldout_sequences_used_for_exclusion_only': True}
    def status(phase, **kwargs):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
        write_json(a.output / 'status.json', state)
        print(json.dumps({'phase': phase, **kwargs}), flush=True)
    formal = a.workspace / 'artifacts/laya_formal_data'
    manifest = json.loads((formal / 'manifest.json').read_text())
    guarded = {'dna': set(), 'protein': set()}
    heldout_counts = {}
    source_files = {}
    for task, modality in [('promoter_detection', 'dna'), ('fold_class', 'protein')]:
        for split in ['selection_dev', 'calibration', 'test']:
            spec = manifest['tasks'][task]['files'][split]
            path = a.workspace / spec['path']
            assert sha(path) == spec['sha256']
            # Only sequence field is used here; no held-out target is inspected.
            sequences = [r['sequence'] for r in read_lines(path)]
            for seq in sequences:
                add_guard(guarded[modality], seq, modality)
            heldout_counts[task + ':' + split] = len(sequences)
            source_files[str(path)] = spec['sha256']
    status('guard_ready', protected_sequence_counts=heldout_counts, guard_kmers={k: len(v) for k,v in guarded.items()})
    counts = {'dna': (32768, 512), 'protein': (32768, 512), 'text': (8192, 128)}
    names = {'dna': 'dna_32g.txt', 'protein': a.protein_file, 'text': 'openwebtext.txt'}
    raw = {'train': [], 'validation': []}
    corpus_stats = {}
    for index, modality in enumerate(['dna', 'protein', 'text']):
        path = a.workspace / 'data/01_raw_cpt' / names[modality]
        size = path.stat().st_size
        rng = random.Random(SEED + index)
        seen = set()
        rejected = Counter()
        ntrain, nval = counts[modality]
        accepted = []
        validation_guard = set()
        with path.open('rb') as f:
            attempts = 0
            while len(accepted) < ntrain + nval:
                attempts += 1
                if attempts > (ntrain + nval) * 80:
                    raise RuntimeError('Excessive sampling rejections: ' + modality)
                f.seek(rng.randrange(max(1, size - 32768)))
                f.readline()
                offset = f.tell()
                payload = f.readline()
                if len(payload) > 100000 or not payload:
                    rejected['line_size'] += 1
                    continue
                try:
                    value = payload.decode('utf-8').strip()
                except UnicodeDecodeError:
                    rejected['encoding'] += 1
                    continue
                if modality != 'text':
                    value = value.upper()
                    alphabet = 'ACGTN' if modality == 'dna' else 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                    if len(value) < 64 or not set(value) <= set(alphabet):
                        rejected['alphabet_or_length'] += 1
                        continue
                    if hits_guard(guarded[modality], value, modality):
                        rejected['downstream_holdout_long_kmer'] += 1
                        continue
                elif len(value) < 200:
                    rejected['short_text'] += 1
                    continue
                identity = key(value, modality)
                if identity in seen:
                    rejected['duplicate_exact_or_dna_rc'] += 1
                    continue
                seen.add(identity)
                is_val = len(accepted) < nval
                if modality != 'text':
                    if not is_val and hits_guard(validation_guard, value, modality):
                        rejected['cpt_validation_long_kmer'] += 1
                        continue
                    if is_val:
                        add_guard(validation_guard, value, modality)
                # Bound raw storage; token-space windows below preserve the actual input span.
                limit = 4096 if modality == 'text' else 2048
                start = int(identity[:8], 16) % max(1, len(value) - limit + 1)
                content = value[start:start+limit]
                row = {'id': modality + ':' + identity, 'modality': modality, 'content': content,
                       'source': str(path), 'byte_offset': offset, 'source_line_sha256': hashlib.sha256(payload).hexdigest(),
                       'raw_character_length': len(value), 'raw_window_start': start, 'split': 'validation' if is_val else 'train'}
                accepted.append(row)
                raw[row['split']].append(row)
                if len(accepted) % 8192 == 0:
                    status('sampling_' + modality, accepted=len(accepted), attempts=attempts)
        corpus_stats[modality] = {'source': str(path), 'source_bytes': size, 'source_mtime_ns': path.stat().st_mtime_ns,
                                  'train': ntrain, 'validation': nval, 'attempts': attempts, 'rejected': dict(rejected),
                                  'sampling': 'deterministic random byte offsets, skip partial line; length-biased, not uniform over records',
                                  'full_source_hash_recomputed': False, 'used_line_hashes_and_bytes_snapshotted': True}
    for split, rows in raw.items():
        write_lines(a.output / ('raw_' + split + '.jsonl'), rows)
    composition = {m: Counter() for m in ['dna','protein']}
    for row in raw['train']:
        if row['modality'] in composition:
            composition[row['modality']].update(row['content'])
    missing_amino_acids = sorted(set('ACDEFGHIKLMNPQRSTVWY') - set(composition['protein']))
    composition_audit = {'character_counts': {m: dict(sorted(c.items())) for m,c in composition.items()},
                         'required_canonical_amino_acids': 'ACDEFGHIKLMNPQRSTVWY',
                         'missing_canonical_amino_acids': missing_amino_acids,
                         'protein_source': str(a.workspace / 'data/01_raw_cpt' / a.protein_file),
                         'status': 'pass' if not missing_amino_acids else 'fail'}
    write_json(a.output / 'residue_composition_audit.json', composition_audit)
    if missing_amino_acids:
        status('failed_source_composition', status='failed', missing_amino_acids=missing_amino_acids)
        raise ValueError('Protein corpus lacks canonical amino acids: ' + str(missing_amino_acids))
    status('fit_vocabulary')
    rep = a.output / 'representation'
    rep.mkdir()
    (rep / 'source_tokenizers').mkdir()
    base = AutoTokenizer.from_pretrained(a.model_dir / 'tokenizer')
    # Transformers 5 may materialize the vocabulary on len(tokenizer).
    # Cache once rather than repeat it for every corpus token.
    base_size = len(base)
    expanded = AutoTokenizer.from_pretrained(a.model_dir / 'tokenizer')
    entries = []
    sources = {}
    for modality in ['dna', 'protein']:
        alphabet = list('ACGTN' if modality == 'dna' else 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        source = Tokenizer(models.BPE(unk_token='[UNK]'))
        source.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=1024, min_frequency=20, special_tokens=['[PAD]', '[UNK]'], initial_alphabet=alphabet, show_progress=False)
        source.train_from_iterator((r['content'] for r in raw['train'] if r['modality'] == modality), trainer=trainer)
        source.save(str(rep / 'source_tokenizers' / (modality + '.json')))
        sources[modality] = source
        for piece, index in sorted(source.get_vocab().items(), key=lambda x: x[1]):
            if piece in ['[PAD]', '[UNK]']:
                continue
            ids = base(piece, add_special_tokens=False)['input_ids']
            assert ids and not any(i in base.all_special_ids for i in ids)
            entries.append({'modality': modality, 'piece': piece, 'source_id': index,
                            'token': '<BIO_' + modality.upper() + ':' + piece + '>', 'base_ids': ids})
    assert expanded.add_tokens([AddedToken(e['token'], normalized=False) for e in entries]) == len(entries)
    vocab = expanded.get_vocab()
    assert all(vocab[s] == i for s, i in base.get_vocab().items())
    for entry in entries:
        entry['expanded_id'] = vocab[entry['token']]
    maps = {m: {e['source_id']: e['expanded_id'] for e in entries if e['modality'] == m} for m in sources}
    base.save_pretrained(rep / 'base_tokenizer')
    expanded.save_pretrained(rep / 'expanded_tokenizer')
    write_json(rep / 'new_tokens.json', entries)
    prefixes = {m: base(('DNA' if m == 'dna' else 'Protein') + ' sequence:', add_special_tokens=False)['input_ids'] for m in sources}
    def encode(content, modality, max_length, identity, crop):
        if modality == 'text':
            prefix = []
            ids = base(content, add_special_tokens=False)['input_ids']
        else:
            encoded = sources[modality].encode(content)
            assert ''.join(encoded.tokens) == content and '[UNK]' not in encoded.tokens
            prefix = prefixes[modality]
            ids = [maps[modality][i] for i in encoded.ids]
        available = max_length - len(prefix) - 2
        start = 0
        if len(ids) > available:
            if not crop:
                raise ValueError('Full supervised input exceeds length: ' + identity)
            start = int(digest(identity + ':token-window')[:8], 16) % (len(ids) - available + 1)
        body = ids[start:start+available]
        return [base.cls_token_id] + prefix + body + [base.sep_token_id], 1+len(prefix), {'full_body_tokens': len(ids), 'token_window_start': start, 'cropped': len(ids) > available}
    token_counts = {'train': Counter(), 'validation': Counter()}
    length_stats = {}
    for split, rows in raw.items():
        encoded_rows = []
        for r in rows:
            ids, start, span = encode(r['content'], r['modality'], 256, r['id'], True)
            token_counts[split].update(i for i in ids if i >= base_size)
            encoded_rows.append({'id': r['id'], 'modality': r['modality'], 'ids': ids, 'body_start': start, **span})
        write_lines(a.output / ('cpt_' + split + '.jsonl'), encoded_rows)
        length_stats[split] = {'rows': len(rows), 'tokens': sum(len(r['ids']) for r in encoded_rows), 'cropped_rows': sum(r['cropped'] for r in encoded_rows)}
    for entry in entries:
        entry['cpt_train_input_occurrences'] = token_counts['train'][entry['expanded_id']]
        entry['cpt_validation_input_occurrences'] = token_counts['validation'][entry['expanded_id']]
    write_json(rep / 'new_tokens.json', entries)
    status('prepare_sft')
    supervised = {}
    for split in ['train', 'selection_dev']:
        spec = manifest['tasks']['promoter_detection']['files'][split]
        path = a.workspace / spec['path']
        assert sha(path) == spec['sha256']
        source_files[str(path)] = spec['sha256']
        supervised[split] = read_lines(path)
    subset = balanced_subset(supervised['train'], 4096)
    sft_counts = {}
    for name, rows in [('train_4096', subset), ('train_full', supervised['train']), ('dev', supervised['selection_dev'])]:
        out = []
        for row in rows:
            ids, start, span = encode(row['sequence'], 'dna', 256, row['id'], False)
            out.append({'id': row['id'], 'ids': ids, 'label': row['label'], 'group_id': row['group_id']})
        write_lines(a.output / ('sft_' + name + '.jsonl'), out)
        sft_counts[name] = {'rows': len(rows), 'labels': dict(Counter(r['label'] for r in rows)), 'max_tokens': max(len(r['ids']) for r in out), 'truncation': 0}
    assert not {r['group_id'] for r in supervised['train']} & {r['group_id'] for r in supervised['selection_dev']}
    metadata = {'schema': 'biocpt-first-round-v2-source-QC', 'seed': SEED, 'source_downstream_sha256': source_files,
                'heldout_sequence_counts_for_exclusion': heldout_counts, 'test_targets_used': False, 'test_inference': False,
                'corpus': corpus_stats, 'cpt_lengths': length_stats, 'sft': sft_counts, 'old_vocab_size': base_size,
                'new_vocab_size': len(expanded), 'added_by_modality': dict(Counter(e['modality'] for e in entries)),
                'unobserved_added_tokens_in_cpt_train': [e['token'] for e in entries if not e['cpt_train_input_occurrences']],
                'vocab_fit': 'fresh DNA/protein BPE fitted only on filtered CPT training samples; 1024 source entries per modality',
                'residue_composition_audit': composition_audit,
                'encoding': 'explicit source-piece to distinct model-ID mapping; natural language uses unchanged base IDs',
                'guards': 'any shared 31-mer (DNA, both strands) or 15-mer (protein) with protected downstream sequences or CPT validation rejects a training corpus line',
                'limitations': ['Fixed promoter/fold heldouts only; future tasks require a new contamination admission audit.',
                                'Long-kmer filtering is conservative but is not a full homology/parent-cluster independence guarantee.',
                                'Downstream development/test memberships have historical prior use; no fresh confirmatory test claim.',
                                'Random-byte sampling is length-biased; corpus upstream provenance remains historical.'],
                'output_sha256': {str(x.relative_to(a.output)): sha(x) for x in sorted(a.output.rglob('*')) if x.is_file() and x.name != 'status.json'}}
    write_json(a.output / 'manifest.json', metadata)
    status('complete', status='complete', sft=sft_counts, new_vocab_size=len(expanded), unobserved_new_tokens=len(metadata['unobserved_added_tokens_in_cpt_train']))


if __name__ == '__main__':
    main()
