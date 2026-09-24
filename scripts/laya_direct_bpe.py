"""Explicit biological BPE -> model ID mapping; natural language uses base BPE.

Source tokenizers are the historical OmniGene BPEs, with missing single-character
alphabet entries appended without altering existing IDs or merges. Every ordinary
source piece (including single characters) receives a separate modality-specific
model ID. Token display names are serialization metadata, never input text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tokenizers import AddedToken, Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/laya_direct_bpe_legacy_representation'
LEGACY = Path('/root/autodl-fs/omnigene_v2/scripts/vocab/trained_bpe')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output=DEFAULT, source_root=LEGACY):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite representation: {output}')
    source_root = Path(source_root)
    source_names = {'dna': 'dna_bpe_20k.json', 'protein': 'protein_bpe_8k.json'}
    repaired_sources, repairs = {}, {}
    for kind,name in source_names.items():
        original = json.loads((source_root/name).read_text())
        vocab = original['model']['vocab']
        # The original protein BPE silently drops N; DNA similarly lacks N.
        # Preserve all learned merges and append only a fixed biological alphabet.
        alphabet = 'ACGTN' if kind=='dna' else 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        missing = [letter for letter in alphabet if letter not in vocab]
        for letter in missing:
            vocab[letter] = max(vocab.values())+1
        repairs[kind] = missing
        repaired_sources[kind] = Tokenizer.from_str(json.dumps(original))
    base = AutoTokenizer.from_pretrained(ROOT / 'artifacts/laya_model/tokenizer')
    expanded = AutoTokenizer.from_pretrained(ROOT / 'artifacts/laya_model/tokenizer')
    entries = []
    for kind, name in source_names.items():
        source = repaired_sources[kind]
        for piece, source_id in sorted(source.get_vocab().items(), key=lambda pair: pair[1]):
            if piece in {'[UNK]', '[PAD]'}:
                continue
            if not piece.isalpha():
                raise ValueError(f'Unexpected source vocabulary piece: {piece!r}')
            ids = base(piece, add_special_tokens=False)['input_ids']
            if not ids or any(i in base.all_special_ids for i in ids):
                raise ValueError(f'Cannot initialize {kind}:{piece} from raw fragment')
            entries.append({'modality': kind, 'source_id': source_id, 'piece': piece,
                            'token': f'<BIO_{kind.upper()}:{piece}>', 'base_ids': ids})
    added = expanded.add_tokens([AddedToken(x['token'], normalized=False) for x in entries])
    if added != len(entries):
        raise ValueError('Duplicate or conflicting vocabulary entries')
    vocab = expanded.get_vocab()
    for entry in entries:
        entry['expanded_id'] = vocab[entry['token']]
    if any(vocab[k] != v for k, v in base.get_vocab().items()):
        raise ValueError('Original token IDs changed')
    output.mkdir(parents=True)
    (output/'source_tokenizers').mkdir()
    (output/'original_source_tokenizers').mkdir()
    for kind,name in source_names.items():
        repaired_sources[kind].save(str(output/'source_tokenizers'/name))
        shutil.copy2(source_root/name, output/'original_source_tokenizers'/name)
    scripts = source_root.parent
    provenance_scripts = [scripts/name for name in ('1-sample_bpe_corpus.py','2-train_bio_bpe.py','3-expand_tokenizer.py')]
    (output/'provenance_scripts').mkdir()
    for path in provenance_scripts:
        if path.exists():shutil.copy2(path,output/'provenance_scripts'/path.name)
    base.save_pretrained(output / 'base_tokenizer')
    expanded.save_pretrained(output / 'expanded_tokenizer')
    (output / 'new_tokens.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2)+'\n')
    meta = {'schema_version': 'laya-direct-bpe-legacy-v1', 'no_cpt': True,
            'fit_policy': 'reuse external historical BPE; no refitting; no neural CPT',
            'historical_fit_recipe': 'sample about 1 GiB each from dna_32g.txt and protein_uni_16.txt, seed42; BPE min_frequency10',
            'historical_fit_corpus_overlap_with_evaluation': 'not established; no train-only claim',
            'source_files': {k:{'path':str(source_root/n),'sha256':digest(source_root/n)} for k,n in source_names.items()},
            'alphabet_repair': repairs, 'existing_merges_preserved':True,
            'old_vocab_size': len(base), 'new_vocab_size': len(expanded),
            'added_by_modality': {k: sum(x['modality'] == k for x in entries) for k in source_names},
            'initialization': 'mean of original tokenizer IDs of RAW biological fragment; no delimiters',
            'encoding': 'source BPE ID mapped directly to added model ID; natural language uses base tokenizer',
            'output_hashes': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({k: meta[k] for k in ('old_vocab_size','new_vocab_size','added_by_modality')}, indent=2))


class DirectBPE:
    def __init__(self, directory=DEFAULT):
        self.root = Path(directory)
        self.meta = json.loads((self.root / 'metadata.json').read_text())
        for name, expected in self.meta['output_hashes'].items():
            if digest(self.root / name) != expected:
                raise ValueError(f'Representation hash mismatch: {name}')
        self.base = AutoTokenizer.from_pretrained(self.root / 'base_tokenizer')
        self.expanded = AutoTokenizer.from_pretrained(self.root / 'expanded_tokenizer')
        self.entries = json.loads((self.root / 'new_tokens.json').read_text())
        self.sources = {k: Tokenizer.from_file(str(self.root / 'source_tokenizers' / name))
                        for k, name in [('dna','dna_bpe_20k.json'),('protein','protein_bpe_8k.json')]}
        self.maps = {k: {e['source_id']: e['expanded_id'] for e in self.entries if e['modality'] == k}
                     for k in self.sources}
        self.inverse = {e['expanded_id']: e['piece'] for e in self.entries}

    def sequence_ids(self, sequence, kind):
        encoded = self.sources[kind].encode(sequence)
        if ''.join(encoded.tokens) != sequence:
            raise ValueError('Source BPE changed the sequence or produced UNK')
        try:
            ids = [self.maps[kind][i] for i in encoded.ids]
        except KeyError as exc:
            raise ValueError('Unmapped source BPE token') from exc
        if ''.join(self.inverse[i] for i in ids) != sequence:
            raise ValueError('Model IDs do not reconstruct the original sequence')
        return ids

    def build(self, row, condition, builder, order, max_len=1024, head_max_len=256):
        q = {'t': 'choice', 'ins': 'Choose the correct biological label for the sequence.',
             'crit': {x: None for x in row['choices']}}
        prefix = f"Task context: {row['context']}\nSequence: "
        if condition == 'raw':
            ids, markers = builder(self.base, prefix + row['sequence'], q,
                                   max_len=1_000_000, head_max_len=head_max_len, option_order=order)
        elif condition == 'full_bpe':
            # Reuse the official header exactly, including candidate markers.
            # Its final SEP terminates the empty state; replace that with our
            # explicitly tokenized state and restore the final SEP.
            header, markers = builder(self.base, '', q, max_len=1_000_000,
                                      head_max_len=head_max_len, option_order=order)
            context_ids = self.base(prefix.replace(self.base.mask_token, ' '), add_special_tokens=False)['input_ids']
            ids = header[:-1] + context_ids + self.sequence_ids(row['sequence'], row['kind']) + [self.base.sep_token_id]
        else:
            raise ValueError(condition)
        if len(ids) > max_len:
            raise ValueError(f"Full input exceeds {max_len}: {row['id']} {condition} {len(ids)}")
        if len(markers) != len(row['choices']):
            raise ValueError('Candidate marker loss')
        for canonical, marker in zip(order, markers):
            expected = self.base(' '+row['choices'][canonical], add_special_tokens=False)['input_ids']
            if ids[marker+1:marker+1+len(expected)] != expected:
                raise ValueError('Candidate text truncated or remapped incorrectly')
        return ids, markers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--source-dir', type=Path, default=LEGACY)
    args=parser.parse_args()
    prepare(args.output,args.source_dir)
