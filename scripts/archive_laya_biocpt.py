#!/usr/bin/env python3
"""Archive completed-run evidence, excluding raw corpora and large model states."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from laya_biocpt_data import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    sources=[]
    def copy(relative):
        source=a.root/relative
        if not source.is_file():
            return
        target=a.output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        assert sha(source)==sha(target)
        sources.append({'source_relative':str(relative),'archive_relative':str(relative),'sha256':sha(source)})
    for name in ['runtime_environment.json','source_release_identity_check.json','base_encoder_config.json','resume_state_audit.json','embedding_exposure_summary.json','preparation_reproducibility_check.json','protein_source_N_audit.json','rejected_protein_source_N_audit.json','negative_admission_test.json','preserved_promoter_inputs_check.json','no_cpt_4096_source_revision_repeat_check.json','data/residue_composition_audit.json','data/manifest.json','data/status.json','data/representation/new_tokens.json','data/representation/source_tokenizers/dna.json','data/representation/source_tokenizers/protein.json','round/manifest.json','round/status.json','round/EXPERIMENT_PLAN.md','round/paired_4096_complete.json']:
        copy(Path(name))
    for name in ['cpt','no_cpt_4096','cpt_4096','no_cpt_full','cpt_full']:
        folder=a.root/'round'/name
        status=folder/'status.json'
        if not status.is_file() or json.loads(status.read_text())['status']!='complete':
            continue
        for source in sorted(folder.iterdir()):
            if source.suffix not in ['.json','.jsonl']:
                continue
            relative=source.relative_to(a.root)
            if source.name.endswith('_predictions.jsonl'):
                target=a.output/(str(relative)+'.gz')
                target.parent.mkdir(parents=True,exist_ok=True)
                content=source.read_bytes()
                target.write_bytes(gzip.compress(content,compresslevel=9,mtime=0))
                assert gzip.decompress(target.read_bytes())==content
                sources.append({'source_relative':str(relative),'archive_relative':str(target.relative_to(a.output)),'source_sha256':hashlib.sha256(content).hexdigest(),'compressed_sha256':sha(target),'compression':'gzip, mtime=0; lossless exact source bytes'})
            else:
                copy(relative)
    warning='\n**Superseded source diagnostic:** this historical protein source has no N characters and was rejected for continued broad protein CPT. Its recorded outcomes are not the corrected-source primary result.\n' if (a.root/'protein_source_N_audit.json').exists() else ''
    (a.output/'README.md').write_text('# Biological CPT evidence archive\n'+warning+'\nThis directory contains completed-run metrics, traces, deterministic gzip-compressed per-example predictions, vocabulary maps, data/recipe hashes, and environment metadata. The round status identifies whether the larger comparison is still running. Raw corpus snapshots, encoded training data, model weights and optimizer/RNG state remain in the local experiment workspace; they are not included here. Predictor JSONL files decompress exactly to the original bytes, verified against `archive_manifest.json`. No test predictions were made.\n\nLocal experiment root: `'+str(a.root)+'`.\n')
    write_json(a.output/'archive_manifest.json',{'source_root':str(a.root),'completed_runs':[p.name for p in (a.output/'round').iterdir() if p.is_dir()],'sources':sources,'weights_included':False,'raw_sequences_included':False,'test_inference':False})
    print(json.dumps({'files':len(sources),'output':str(a.output)}))


if __name__=='__main__':
    main()
