#!/bin/bash
set -u
cd /root/autodl-tmp/jev_gene
TR="python3 -u scripts/benchmark_v2/train_benchmark_v2.py"
EV="python3 -u scripts/benchmark_v2/eval_benchmark_v2.py"
COMMON="--all --family-balance --batches-per-family 30 --epochs 12 --max-per-task 2048 --min-valid 128 --cap-family ProteinGym=12 GUE=12 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 200 --dev-max 96 --save-init"
GBT=$(ls -d data/06_benchmark_v2_unified/gb_*/ | sed 's|.*/\(gb_[^/]*\)/|\1|' | tr '\n' ' ')
TAPET="tape_fluorescence tape_stability"

echo "########## XFAM DRIVER START $(date) ##########"
# GB holdout, seeds 2,3 (seed1=bv2_xfam 已有)
for S in 20261002 20261003; do
  echo "=== [$(date +%H:%M)] train GB-holdout seed=$S ==="
  $TR --exclude-family GenomicBenchmarks --seed $S --name bv2_xfam_gb_s$S $COMMON > xfam_gb_s$S.log 2>&1
  echo "=== [$(date +%H:%M)] eval GB trained seed=$S ==="
  $EV --ckpt artifacts/benchmark_v2_train/bv2_xfam_gb_s$S/model.safetensors --tasks $GBT --max-per-task 200 --out artifacts/benchmark_v2_eval/xfam_gb_trained_s$S > xfam_gb_ev_s$S.log 2>&1
done
# TAPE holdout (score family), seed1
echo "=== [$(date +%H:%M)] train TAPE-holdout seed=20261001 ==="
$TR --exclude-family TAPE --seed 20261001 --name bv2_xfam_tape $COMMON > xfam_tape.log 2>&1
echo "=== [$(date +%H:%M)] eval TAPE init+trained ==="
$EV --ckpt artifacts/benchmark_v2_train/bv2_xfam_tape/init.safetensors --tasks $TAPET --max-per-task 200 --out artifacts/benchmark_v2_eval/xfam_tape_init > xfam_tape_ev_init.log 2>&1
$EV --ckpt artifacts/benchmark_v2_train/bv2_xfam_tape/model.safetensors --tasks $TAPET --max-per-task 200 --out artifacts/benchmark_v2_eval/xfam_tape_trained > xfam_tape_ev_tr.log 2>&1
echo "########## XFAM DRIVER DONE $(date) ##########"
