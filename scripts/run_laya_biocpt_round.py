#!/usr/bin/env python3
"""Freeze and execute the bounded CPT/no-CPT comparison, stopping on failure."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from laya_biocpt_data import sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--micro',type=int,default=8)
    a=p.parse_args()
    data=a.root/'data'
    assert json.loads((data/'status.json').read_text())['status']=='complete'
    source_qc=json.loads((data/'residue_composition_audit.json').read_text())
    assert source_qc['status']=='pass' and not source_qc['missing_canonical_amino_acids']
    smoke=json.loads((a.root/'smoke/status.json').read_text())
    assert smoke['status']=='complete' and smoke['checkpoint_reload_exact'] and smoke['sft_seconds_per_update']>0
    out=a.root/'round'
    out.mkdir()
    frozen=out/'frozen_code'
    frozen.mkdir()
    source=Path(__file__).resolve().parent
    for name in ['laya_biocpt_data.py','laya_biocpt_train.py','run_laya_biocpt_round.py','test_laya_biocpt.py']:
        shutil.copy2(source/name,frozen/name)
    shutil.copy2(source.parent/'refine-logs/EXPERIMENT_PLAN.md',out/'EXPERIMENT_PLAN.md')
    run_order=[('cpt','cpt',None,None),('no_cpt_4096','sft','4096',None),('cpt_4096','sft','4096',out/'cpt/model.safetensors'),('no_cpt_full','sft','full',None),('cpt_full','sft','full',out/'cpt/model.safetensors')]
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'data_manifest_sha256':sha(data/'manifest.json'),'code_sha256':{x.name:sha(x) for x in frozen.iterdir()},'smoke':smoke,'micro':a.micro,'run_order':[r[0] for r in run_order],'first_round_only':True,'test_access':False,'full_SFT_updates_each':3*((16766+63)//64),'small_SFT_updates_each':192}
    write_json(out/'manifest.json',config)
    for name,mode,samples,checkpoint in run_order:
        command=[sys.executable,'-u',str(frozen/'laya_biocpt_train.py'),mode,'--model-dir',str(a.model_dir),'--data-dir',str(data),'--output',str(out/name),'--micro',str(a.micro)]
        if samples:
            command.extend(['--samples',samples])
        if checkpoint:
            command.extend(['--checkpoint',str(checkpoint)])
        write_json(out/'status.json',{'status':'running','current':name,'command':command})
        with (out/(name+'.log')).open('w') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            write_json(out/'status.json',{'status':'failed','current':name,'returncode':completed.returncode})
            raise RuntimeError(name+' failed; inspect preserved log')
        assert json.loads((out/name/'status.json').read_text())['status']=='complete'
    write_json(out/'status.json',{'status':'complete','runs':[r[0] for r in run_order]})


if __name__=='__main__':
    main()
