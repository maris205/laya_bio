#!/bin/bash
set -u
exec 9>/tmp/blite.lock
flock -n 9 || { echo another-instance-running; exit 0; }
cd /root/autodl-tmp/jev_gene
QD="python3 -u scripts/benchmark_v2/qwen_decision.py"
FAMTASKS() {
python3 - "$1" <<'PY'
import json, sys
F = sys.argv[1]
PF=[("pg_","ProteinGym"),("rnac_","RNAcompete"),("gue_","GUE"),("gb_","GenomicBenchmarks"),("gl_","gene_lan"),("dna_","dnagpt_pools"),("deepstarr","DeepSTARR"),("tape_","TAPE"),("lg_","local_snapshots"),("protein_homology","local_snapshots"),("deeploc","DeepLoc")]
fam=lambda t: next((f for p,f in PF if t.startswith(p)),"other")
m=json.load(open("data/06_benchmark_v2_unified/manifest.json"))
print(" ".join(sorted(t for t,v in m.items() if v.get("status")=="ready" and fam(t)==F
      and v["primitive"] in ("choice","noul","score")
      and len(v["modality"])==1 and v["modality"][0] in ("dna","protein"))))
PY
}
echo "########## B-LITE RERUN START $(date) ##########"
for F in local_snapshots TAPE GenomicBenchmarks; do
  T=$(FAMTASKS $F)
  echo "===== [$(date +%H:%M)] B q06spec heads $F ====="
  $QD --arch heads --tasks $T --model models/Qwen3-0.6B --epochs 12 --batches-per-family 208 \
      --max-per-task 2048 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 200 \
      --dev-max 256 --mb 16 --accum 4 --seed 20261001 --name q06spec_$F > q06spec_$F.log 2>&1 \
    || echo "!!! B $F failed"
  find artifacts/benchmark_v2_train/q06spec_$F -name "*.safetensors" -delete 2>/dev/null
  echo "===== [$(date +%H:%M)] B $F done; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') ====="
done
echo "########## B-LITE RERUN DONE $(date) ##########"
