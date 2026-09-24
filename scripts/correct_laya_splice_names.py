#!/usr/bin/env python3
"""Version the pilot with SpliceFinder class names, leaving numeric targets intact.

Evidence: original SpliceFinder code plus central motifs in this training subset.
The dnagpt card lacks an explicit upstream mapping: full dataset lineage remains
unverified. Do not overwrite or retrospectively relabel historical run artifacts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

OLD=['Non-splice site','Acceptor site','Donor site']
NEW=['Acceptor site','Donor site','Non-splice site']
SOURCES=[
 'https://github.com/deepomicslab/SpliceFinder/blob/master/SpliceFinder_sourcecode/test/test_Cla.py',
 'https://github.com/deepomicslab/SpliceFinder/blob/master/iteration_process/generate_seq/generate.py',
 'https://huggingface.co/datasets/dnagpt/dna_splice_site_prediction',
]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    original=a.data_dir/'manifest.json';manifest=json.loads(original.read_text());result={};motifs={}
    for split in ['train','dev']:
        path=a.data_dir/manifest['outputs'][split]['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['outputs'][split]['sha256']
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        for r in rows:
            if r['task']=='splice':
                assert r['choices']==OLD;r['choices']=NEW.copy()
        if split=='train':
            for k in range(3):
                seqs=[r['inputs'][0]['sequence'] for r in rows if r['task']=='splice' and r['label']==k]
                motifs[str(k)]={'n':len(seqs),'AG_at_zero_based_198':sum(s[198:200]=='AG' for s in seqs),
                                'GT_at_zero_based_200':sum(s[200:202]=='GT' for s in seqs)}
        result[split]=''.join(json.dumps(r)+'\n' for r in rows)
    correction={'old_choices':OLD,'new_choices':NEW,'numeric_targets_unchanged':True,'rows_and_splits_unchanged':True,
                'source_urls':SOURCES,'training_motif_audit':motifs,
                'evidence_status':'Source-code-supported correction consistent with training motifs; complete dnagpt source lineage remains unverified.',
                'parent_manifest_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                'correction_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest['splice_semantic_correction']=correction
    manifest['limitations'].append('Historical splice names differed from SpliceFinder convention; this version corrects text only. Upstream lineage remains incompletely documented.')
    a.output.mkdir(parents=True)
    for split,content in result.items():
        (a.output/(split+'.jsonl')).write_text(content)
        manifest['outputs'][split]['sha256']=hashlib.sha256(content.encode()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(correction,indent=2))

if __name__=='__main__':main()
