#!/bin/bash
# Single unified model over ALL 241 supported tasks (both arms, CPT init).
# Family-balanced interleave so ProteinGym's 217 assays cannot flood; global
# best-dev early stop = the honest single-model checkpoint (per-family best kept
# only as diagnostics, not headlined). One seed, documented.
set -u
cd /root/autodl-tmp/jev_gene
TR1="python3 -u scripts/benchmark_v2/train_benchmark_v2.py"
TR2="python3 -u scripts/benchmark_v2/train_taskheads.py"
EV="python3 -u scripts/benchmark_v2/eval_benchmark_v2.py"
ALL=$(python3 - <<'EOF'
import json
PF=[("pg_","ProteinGym"),("rnac_","RNAcompete"),("gue_","GUE"),("gb_","GenomicBenchmarks"),("gl_","gene_lan"),("dna_","dnagpt_pools"),("deepstarr","DeepSTARR"),("tape_","TAPE"),("lg_","local_snapshots"),("protein_homology","local_snapshots"),("deeploc","DeepLoc")]
fam=lambda t:next((f for p,f in PF if t.startswith(p)),"other")
m=json.load(open("data/06_benchmark_v2_unified/manifest.json"))
print(" ".join(sorted(t for t,v in m.items() if v.get("status")=="ready"
      and v["primitive"] in ("choice","noul","score")
      and len(v["modality"])==1 and v["modality"][0] in ("dna","protein"))))
EOF
)
N=$(echo $ALL | wc -w)
COMMON="--batches-per-family 30 --epochs 24 --max-per-task 2048 --min-valid 128 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 500 --dev-max 256 --arm cpt --seed 20261001"
echo "########## JOINT241 START $(date) : $N tasks ##########"
echo "===== [$(date +%H:%M)] arm #2 task heads ====="
$TR2 --tasks $ALL $COMMON --name j241_2 > j241_2.log 2>&1 || echo "!!! j241_2 failed"
echo "===== [$(date +%H:%M)] arm #2 done; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') ====="
echo "===== [$(date +%H:%M)] arm #1 shared scorer ====="
$TR1 --all --family-balance $COMMON --name j241_1 > j241_1.log 2>&1 || echo "!!! j241_1 failed"
$EV --ckpt artifacts/benchmark_v2_train/j241_1/model.safetensors --tasks $ALL \
    --max-per-task 200 --out artifacts/benchmark_v2_eval/j241_1 > j241_1_eval.log 2>&1 \
    || echo "!!! j241_1 eval failed"
find artifacts/benchmark_v2_train/j241_1 artifacts/benchmark_v2_train/j241_2 \
     -name "*.safetensors" -delete 2>/dev/null
echo "########## JOINT241 DONE $(date) ##########"
