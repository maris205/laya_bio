#!/bin/bash
# Reinforcement driver for the paper plan (PAPER_PLAN.md Next Steps):
# Phase A: recreate s1 (shared scorer) seed-avg checkpoints, run the 6-phrasing
#          noul ablation on each seed's best_GenomicBenchmarks checkpoint,
#          then prune heavy checkpoints (keep only best_GenomicBenchmarks).
# Phase B: TAPE-holdout cross-family transfer, seeds 20261002/20261003
#          (replicates bv2_xfam_tape exactly), eval init+trained, prune.
set -u
cd /root/autodl-tmp/jev_gene
TR="python3 -u scripts/benchmark_v2/train_benchmark_v2.py"
EV="python3 -u scripts/benchmark_v2/eval_benchmark_v2.py"
AB="python3 -u scripts/benchmark_v2/noul_ablation.py"
# exact seed_avg_driver.sh config for #1
COMMON_S1="--all --family-balance --batches-per-family 30 --epochs 12 --max-per-task 2048 --min-valid 128 --cap-family ProteinGym=12 GUE=12 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 150 --dev-max 256 --per-family-early-stop"
# exact xfam_driver.sh config
COMMON_X="--all --family-balance --batches-per-family 30 --epochs 12 --max-per-task 2048 --min-valid 128 --cap-family ProteinGym=12 GUE=12 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 200 --dev-max 96 --save-init"
TAPET="tape_fluorescence tape_stability"

echo "########## REINFORCE START $(date) ##########"
echo "===== PHASE A: noul phrasing on stable seed-avg checkpoints ====="
for S in 20261001 20261002 20261003; do
  echo "=== [A $(date +%H:%M)] retrain s1r_$S ==="
  $TR $COMMON_S1 --seed $S --name s1r_$S > reinforce_s1r_$S.log 2>&1
  CK=artifacts/benchmark_v2_train/s1r_$S/best_GenomicBenchmarks.safetensors
  if [ ! -f "$CK" ]; then echo "!!! missing $CK, skip seed $S"; continue; fi
  echo "=== [A $(date +%H:%M)] phrasing ablation seed $S ==="
  $AB --ckpt $CK --max-per-task 200 > reinforce_ablation_$S.log 2>&1
  cp artifacts/benchmark_v2_eval/noul_ablation.json artifacts/benchmark_v2_eval/noul_ablation_s1r_$S.json
  # prune: keep only best_GenomicBenchmarks
  find artifacts/benchmark_v2_train/s1r_$S -name "*.safetensors" ! -name "best_GenomicBenchmarks.safetensors" -delete
  echo "=== [A $(date +%H:%M)] seed $S done, pruned; disk: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') free ==="
done

echo "===== PHASE B: TAPE-holdout cross-family, seeds 2/3 ====="
for S in 20261002 20261003; do
  echo "=== [B $(date +%H:%M)] train TAPE-holdout seed $S ==="
  $TR --exclude-family TAPE --seed $S --name bv2_xfam_tape_s$S $COMMON_X > reinforce_tape_$S.log 2>&1
  D=artifacts/benchmark_v2_train/bv2_xfam_tape_s$S
  echo "=== [B $(date +%H:%M)] eval TAPE init+trained seed $S ==="
  $EV --ckpt $D/init.safetensors --tasks $TAPET --max-per-task 200 --out artifacts/benchmark_v2_eval/xfam_tape_init_s$S > reinforce_tape_ev_init_$S.log 2>&1
  $EV --ckpt $D/model.safetensors --tasks $TAPET --max-per-task 200 --out artifacts/benchmark_v2_eval/xfam_tape_trained_s$S > reinforce_tape_ev_tr_$S.log 2>&1
  # prune: keep model.safetensors only (init is the deterministic base, seed1 copy retained)
  rm -f $D/init.safetensors $D/model_final.safetensors $D/best.safetensors
  echo "=== [B $(date +%H:%M)] seed $S done, pruned; disk: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') free ==="
done
echo "########## REINFORCE DONE $(date) ##########"
