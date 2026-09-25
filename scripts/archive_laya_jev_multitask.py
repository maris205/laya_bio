#!/usr/bin/env python3
"""Publish compact exact evidence, retaining raw sequences and model states locally."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from laya_biocpt_data import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);files=[]
    def copy(rel):
        src=a.root/rel
        if not src.is_file():return
        dest=a.output/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(src)==sha(dest);files.append({'source_relative':str(rel),'archive_relative':str(rel),'sha256':sha(src)})
    for rel in ['data/manifest.json','data/cpt_admission.json','data/status.json','smoke/status.json','smoke/run_config.json','smoke/training_trace.jsonl',
                'round/manifest.json','round/EXPERIMENT_PLAN.md','round/status.json','contract_tests.log',
                'analysis/REPORT.md','analysis/summary.json','analysis/audit.json','analysis/decision_round.png','analysis/decision_round.pdf']:
        copy(Path(rel))
    for name in ['cpt_joint','no_cpt_joint','cpt_promoter','cpt_structural_class','cpt_fluorescence']:
        folder=a.root/'round'/name;status=folder/'status.json'
        if not status.exists() or json.loads(status.read_text())['status']!='complete':continue
        for src in sorted(folder.iterdir()):
            if src.suffix not in ['.json','.jsonl']:continue
            rel=src.relative_to(a.root)
            if src.name.endswith('_predictions.jsonl'):
                content=src.read_bytes();dest=a.output/(str(rel)+'.gz');dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(gzip.compress(content,compresslevel=9,mtime=0));assert gzip.decompress(dest.read_bytes())==content
                files.append({'source_relative':str(rel),'archive_relative':str(dest.relative_to(a.output)),'source_sha256':hashlib.sha256(content).hexdigest(),'compressed_sha256':sha(dest)})
            else:copy(rel)
    for src in sorted((a.root/'round/frozen_code').glob('*.py')):copy(src.relative_to(a.root))
    write_json(a.output/'archive_manifest.json',{'source_root':str(a.root),'files':files,'raw_sequences_included':False,'model_states_included':False,'test_predictions':False})
    (a.output/'README.md').write_text('# Unified Laya JEV-style decision round evidence\n\nOne shared checkpoint per joint arm; three tasks and 8,192 training entities per task. Consult `round/status.json` for completion and `analysis/REPORT.md` for final outcomes when available. The earlier `launch_status.json` is explicitly a historical launch snapshot.\n\nIncludes immutable recipe/code, data-admission hashes, traces, metrics and losslessly compressed predictions. Raw sequences, encoded data, joint model weights and primary CPT-joint optimizer/RNG state remain in `'+str(a.root)+'`. No new test inference. Prediction archives decompress exactly to the recorded source hashes.\n')
    print(json.dumps({'archived_files':len(files),'output':str(a.output)}))

if __name__=='__main__':main()
