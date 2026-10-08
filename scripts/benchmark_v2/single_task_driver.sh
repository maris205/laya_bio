#!/bin/bash
# Single-task fine-tuning arm of the granularity gradient:
# single-task > family-joint > all-joint, same protocol/budget order (2400 upd),
# CPT init, both architectures, one seed. Representative tasks across families
# and interfaces (anchors from prior single-task recipes exist for promoter/fold).
set -u
cd /root/autodl-tmp/jev_gene
TR1="python3 -u scripts/benchmark_v2/train_benchmark_v2.py"
TR2="python3 -u scripts/benchmark_v2/train_taskheads.py"
EV="python3 -u scripts/benchmark_v2/eval_benchmark_v2.py"
TASKS="lg_promoter_detection lg_npp lg_fold_class tape_fluorescence gb_demo_coding_vs_intergenomic_seqs"
COMMON="--epochs 24 --batches-per-family 100 --max-per-task 2048 --min-valid 64 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 200 --dev-max 256 --arm cpt --seed 20261001"

echo "########## SINGLE-TASK START $(date) ##########"
for T in $TASKS; do
  echo "===== [$(date +%H:%M)] $T ====="
  $TR2 --tasks $T $COMMON --name st2_$T > st2_$T.log 2>&1 || { echo "!!! st2_$T failed"; continue; }
  $TR1 --tasks $T $COMMON --name st1_$T > st1_$T.log 2>&1 || { echo "!!! st1_$T failed"; continue; }
  $EV --ckpt artifacts/benchmark_v2_train/st1_$T/model.safetensors --tasks $T \
      --max-per-task 200 --out artifacts/benchmark_v2_eval/st1_$T > st1_${T}_eval.log 2>&1 \
      || echo "!!! st1_${T}_eval failed"
  find artifacts/benchmark_v2_train/st1_$T artifacts/benchmark_v2_train/st2_$T \
       -name "*.safetensors" -delete 2>/dev/null
  echo "===== [$(date +%H:%M)] $T complete; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') ====="
done
echo "########## SINGLE-TASK DONE $(date) ##########"
