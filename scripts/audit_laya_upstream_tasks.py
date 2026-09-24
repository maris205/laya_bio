#!/usr/bin/env python3
"""Inspect pinned upstream pools and paired inputs; never repartition or train.

Reads public annotations for schema/provenance checks only. Existing split
memberships are compared, not changed. No model outcomes are produced.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import re
import warnings
import pyarrow.parquet as pq
from Bio.Seq import Seq
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)
DNA = {'dna_promoter_300':'promoter_detection', 'dna_core_promoter':'core_promoter_detection',
       'dna_splice_site_prediction':'splice_site', 'dna_transcription_factor_prediction':'tf_prediction'}

def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def dna_key(x):
    x=x.upper();rc=x.translate(str.maketrans('ACGTRYSWKMBDHVN','TGCAYRSWMKVHDBN'))[::-1]
    return digest(min(x,rc))
def stats(x):
    a=sorted(x);return {'min':a[0],'median':a[(len(a)-1)//2],'p95':a[int(.95*(len(a)-1))],'max':a[-1]}

def run(source, legacy):
    dna_reports=[]
    for name,task in DNA.items():
        f=source/name/'data/train-00000-of-00001.parquet';rows=pq.read_table(f).to_pylist()
        grouped=defaultdict(set);exact=defaultdict(set)
        for r in rows:
            grouped[dna_key(r['sequence'])].add(r['label']);exact[digest(r['sequence'].upper())].add(r['label'])
        legacy_seen=defaultdict(set);mapping=defaultdict(Counter);n_legacy=0
        for line in (legacy/f'lg_{task}.jsonl').open():
            r=json.loads(line);u=next(m['content'] for m in r['messages'] if m['role']=='user')
            seq=u.strip().splitlines()[-1].strip().upper();key=dna_key(seq)
            legacy_seen[r['split']].add(key);n_legacy+=1
            for numeric in exact.get(digest(seq),[]):mapping[str(numeric)][r['answer_short']]+=1
        all_seen=set().union(*legacy_seen.values())
        dna_reports.append({'repository':'dnagpt/'+name,'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'task':task,'rows':len(rows),'public_split':'train_only_pool',
                            'label_counts':dict(Counter(str(r['label']) for r in rows)),
                            'character_lengths':stats([len(r['sequence']) for r in rows]),
                            'exact_unique_sequences':len(exact),'rc_unique_groups':len(grouped),
                            'conflicting_exact_groups':sum(len(v)>1 for v in exact.values()),
                            'conflicting_rc_groups':sum(len(v)>1 for v in grouped.values()),
                            'legacy_rows':n_legacy,'legacy_groups_found_by_split':{s:len(v & grouped.keys()) for s,v in legacy_seen.items()},
                            'upstream_groups_not_in_legacy_view':len(grouped.keys()-all_seen),
                            'numeric_to_legacy_label_mapping_counts':{k:dict(v) for k,v in mapping.items()}})
    pair_reports=[];global_members=defaultdict(set)
    for f in sorted((source/'gene_lan_transfer').glob('*/*.parquet')):
        config=f.parent.name;rows=pq.read_table(f).to_pylist();group_labels=defaultdict(set);entities=set()
        lengths1=[];lengths2=[];translations=defaultdict(Counter)
        for r in rows:
            a,b=r['sentence1'].upper(),r['sentence2'].upper();y=str(r['label'])
            group_labels[digest(a+'\0'+b)].add(y);entities.update([digest(a),digest(b)])
            global_members[digest(a)].add(config);global_members[digest(b)].add(config)
            lengths1.append(len(a));lengths2.append(len(b))
            if config.startswith('dna_protein_pair'):
                tr=str(Seq(a[:len(a)//3*3]).translate(table=1)).rstrip('*');target=b.rstrip('*')
                translations[y]['n']+=1;translations[y]['frame0_standard_code_exact']+=tr==target
                translations[y]['exact_ignoring_first_residue']+=(len(tr)==len(target) and tr[1:]==target[1:])
        pair_reports.append({'config':config,'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'rows':len(rows),'label_counts':dict(Counter(str(r['label']) for r in rows)),
                             'length_sentence1':stats(lengths1),'length_sentence2':stats(lengths2),
                             'unique_ordered_pairs':len(group_labels),'conflicting_ordered_pairs':sum(len(v)>1 for v in group_labels.values()),
                             'unique_component_sequences':len(entities),
                             'frame0_translation_diagnostic':{k:dict(v) for k,v in translations.items()},
                             'semantic_status':'binary paired data; construction and label semantics require provenance; diagnostic is not an independent biological benchmark'})
    overlap=Counter()
    for configs in global_members.values():
        configs=sorted(configs)
        for i,a in enumerate(configs):
            for b in configs[i+1:]:overlap[a+' : '+b]+=1
    homology=[]
    for task in ['protein_homology_std','protein_homology_remote']:
        f=legacy/(task+'.jsonl');members=defaultdict(set);splits=Counter();lens=[];pairs=defaultdict(set);errors=Counter()
        for line in f.open():
            r=json.loads(line);splits[r['split']]+=1
            u=next(m['content'] for m in r['messages'] if m['role']=='user')
            segments=re.findall(r'(?m)^(?:Sequence [12]:|Protein [AB]:|[AB]:|>seq[12])[ \t]*\n?([A-Z*\-]+)[ \t]*$',u)
            if len(segments)!=2:errors['pair_parse_failure']+=1;continue
            a,b=segments;lens.append(len(a)+len(b));members[r['split']].update([digest(a),digest(b)])
            pairs[digest('\0'.join(sorted([a,b])))].add(r['split'])
        homology.append({'task':task,'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'rows':sum(splits.values()),'raw_splits':dict(splits),'errors':dict(errors),
                         'combined_character_lengths':stats(lens),'shared_component_sequences':{
                         a+':'+b:len(members[a]&members[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
                         'unordered_pairs_in_multiple_splits':sum(len(v)>1 for v in pairs.values())})
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'software':{k:version(k) for k in ['pyarrow','biopython']},'status':'source_pool_audit_not_experiment_results','test_memberships_preserved':True,
            'public_labels_read_for_provenance_checks':True,'dna':dna_reports,'gene_lan_transfer':pair_reports,
            'gene_config_shared_component_sequences':dict(overlap),'protein_homology':homology,
            'limitations':['Paired examples must be split by component/parent clusters, not by pair rows.',
                           'A direct standard-code translation diagnostic is not validation of source construction or alternative genetic codes.',
                           'All upstream DNA and gene_lan_transfer configs have one train pool; a new grouped protocol is needed.',
                           'Synthetic sequence similarity must not be relabeled as natural homology.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--legacy-jsonl',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d=run(a.source_root,a.legacy_jsonl);a.output.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
 for x in d['dna']:print(x['task'],x['rows'],x['legacy_groups_found_by_split'],x['numeric_to_legacy_label_mapping_counts'])
 for x in d['gene_lan_transfer']:print(x['config'],x['rows'],x['frame0_translation_diagnostic'])
 for x in d['protein_homology']:print(x)
if __name__=='__main__':main()
