#!/bin/bash
# A: Laya 423M (cpt) on the matched low-data protocol (1 row/update x 1680, global
#    best-dev early stop), 3 seeds x 2 arms, 37-task slice -> completes the scale axis
#    with the home backbone at the SAME protocol as the Qwen runs.
# B-lite: Qwen3-0.6B (cpt) family-specialized at the Laya-leaderboard recipe
#    (64 rows/update x 2496, per-family single-family runs), heads arm, 3 families,
#    1 seed -> direct "same good recipe" comparison against Laya leaderboard numbers.
set -u
exec 9>/tmp/laya_vs_qwen.lock
flock -n 9 || { echo another-instance-running; exit 0; }
cd /root/autodl-tmp/jev_gene
TR1="python3 -u scripts/benchmark_v2/train_benchmark_v2.py"
TR2="python3 -u scripts/benchmark_v2/train_taskheads.py"
EV="python3 -u scripts/benchmark_v2/eval_benchmark_v2.py"
QD="python3 -u scripts/benchmark_v2/qwen_decision.py"
COMMON_A="--all --family-balance --batches-per-family 20 --epochs 12 --batch 1 --max-per-task 2048 --min-valid 128 --cap-family ProteinGym=12 GUE=12 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 150 --dev-max 256 --arm cpt"
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

echo "########## A: LAYA 423M MATCHED-LOW-DATA START $(date) ##########"
for s in 20261001 20261002 20261003; do
  echo "===== [$(date +%H:%M)] A #1 seed $s ====="
  $TR1 $COMMON_A --seed $s --name l423_low_s$s > l423_low_s$s.log 2>&1 || echo "!!! A#1 s$s failed"
  $EV --ckpt artifacts/benchmark_v2_train/l423_low_s$s/model.safetensors \
      --tasks $(python3 -c "import json;print(' '.join(json.load(open('artifacts/benchmark_v2_train/l423_low_s$s/run_config.json'))['tasks']))") \
      --max-per-task 200 --out artifacts/benchmark_v2_eval/l423_low_s$s > l423_low_s${s}_ev.log 2>&1 \
      || echo "!!! A#1 s$s eval failed"
  echo "===== [$(date +%H:%M)] A #2 seed $s ====="
  $TR2 --task-config artifacts/benchmark_v2_train/l423_low_s$s/run_config.json \
      --epochs 12 --batches-per-family 20 --batch 1 --max-per-task 2048 --min-valid 128 \
      --warmup-frac 0.15 --clip 0.5 --dev-eval-every 150 --dev-max 256 --arm cpt \
      --seed $s --name l423_low_h_s$s > l423_low_h_s$s.log 2>&1 || echo "!!! A#2 s$s failed"
  find artifacts/benchmark_v2_train/l423_low_s$s artifacts/benchmark_v2_train/l423_low_h_s$s \
       -name "*.safetensors" -delete 2>/dev/null
  echo "===== [$(date +%H:%M)] A seed $s done; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') ====="
done
echo "########## A DONE $(date) ##########"

echo "########## B-lite: QWEN-0.6B SPECIALIZED RECIPE START $(date) ##########"
for F in local_snapshots TAPE GenomicBenchmarks; do
  T=$(FAMTASKS $F)
  echo "===== [$(date +%H:%M)] B q06spec heads $F ====="
  $QD --arch heads --tasks $T --model models/Qwen3-0.6B --epochs 12 --batches-per-family 208 \
      --max-per-task 2048 --min-valid 64 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 200 \
      --dev-max 256 --mb 32 --accum 2 --arm cpt --seed 20261001 --name q06spec_$F > q06spec_$F.log 2>&1 \
    || echo "!!! B $F failed"
  find artifacts/benchmark_v2_train/q06spec_$F -name "*.safetensors" -delete 2>/dev/null
  echo "===== [$(date +%H:%M)] B $F done; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') ====="
done
echo "########## B DONE $(date) ##########"
