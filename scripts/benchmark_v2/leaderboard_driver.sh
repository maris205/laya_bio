#!/bin/bash
# Full-benchmark family-specialized leaderboard (paper_v2 section 5.5 / appendix).
# For each supported family: train #2 (task heads) and #1 (shared scorer) on that
# family ONLY, with the stabilized protocol (warmup .15, clip .5, dev early stop,
# dev>=256), then evaluate the test split. Single training seed (documented).
# Small families first so numbers land early; ProteinGym last (longest).
set -u
cd /root/autodl-tmp/jev_gene
TR1="python3 -u scripts/benchmark_v2/train_benchmark_v2.py"
TR2="python3 -u scripts/benchmark_v2/train_taskheads.py"
EV="python3 -u scripts/benchmark_v2/eval_benchmark_v2.py"
ALLFAM="GUE GenomicBenchmarks ProteinGym TAPE DeepSTARR dnagpt_pools local_snapshots"
ORDER="dnagpt_pools TAPE DeepSTARR local_snapshots GenomicBenchmarks GUE ProteinGym"

tasks_of() {  # enumerate supported tasks for a family from the manifest
python3 - "$1" <<'EOF'
import json, sys
F = sys.argv[1]
PF = [("pg_","ProteinGym"),("rnac_","RNAcompete"),("gue_","GUE"),("gb_","GenomicBenchmarks"),
      ("gl_","gene_lan"),("dna_","dnagpt_pools"),("deepstarr","DeepSTARR"),("tape_","TAPE"),
      ("lg_","local_snapshots"),("protein_homology","local_snapshots"),("deeploc","DeepLoc")]
fam = lambda t: next((f for p, f in PF if t.startswith(p)), "other")
m = json.load(open("data/06_benchmark_v2_unified/manifest.json"))
ts = [t for t, v in m.items() if v.get("status") == "ready" and fam(t) == F
      and v["primitive"] in ("choice", "noul", "score")
      and len(v["modality"]) == 1 and v["modality"][0] in ("dna", "protein")]
print(" ".join(sorted(ts)))
EOF
}

others_of() {  # families to exclude so that only $1 trains
  local keep="$1"; local out=""
  for f in $ALLFAM; do [ "$f" != "$keep" ] && out="$out $f"; done
  echo $out
}

echo "########## LEADERBOARD START $(date) ##########"
for F in $ORDER; do
  T=$(tasks_of "$F"); N=$(echo $T | wc -w)
  MPT=2048; [ "$F" = "ProteinGym" ] && MPT=512   # cap data volume for the 217-assay family
  echo "===== [$(date +%H:%M)] $F ($N tasks) ====="
  # ---- arm #2: task heads (family-specialized) ----
  $TR2 --tasks $T --epochs 12 --batches-per-family 208 --max-per-task $MPT --min-valid 128 \
       --warmup-frac 0.15 --clip 0.5 --dev-eval-every 200 --dev-max 256 --arm cpt \
       --seed 20261001 --name lb2_$F > lb2_$F.log 2>&1 || { echo "!!! lb2_$F failed"; continue; }
  echo "===== [$(date +%H:%M)] $F #2 done; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') ====="
  # ---- arm #1: shared scorer (family-specialized) ----
  $TR1 --all --exclude-family $(others_of "$F") --family-balance --batches-per-family 208 \
       --epochs 12 --max-per-task $MPT --min-valid 128 --warmup-frac 0.15 --clip 0.5 \
       --dev-eval-every 200 --dev-max 256 --per-family-early-stop --arm cpt \
       --seed 20261001 --name lb1_$F > lb1_$F.log 2>&1 || { echo "!!! lb1_$F failed"; continue; }
  $EV --ckpt artifacts/benchmark_v2_train/lb1_$F/model.safetensors --tasks $T \
      --max-per-task 200 --out artifacts/benchmark_v2_eval/lb1_$F > lb1_${F}_eval.log 2>&1 \
      || echo "!!! lb1_${F}_eval failed"
  # ---- prune weights (keep configs/curves/results) ----
  find artifacts/benchmark_v2_train/lb1_$F artifacts/benchmark_v2_train/lb2_$F \
       -name "*.safetensors" -delete 2>/dev/null
  echo "===== [$(date +%H:%M)] $F complete; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') ====="
done
echo "########## LEADERBOARD DONE $(date) ##########"
