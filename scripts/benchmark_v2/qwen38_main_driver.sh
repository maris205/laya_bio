#!/bin/bash
# Paper-2 main comparison: Qwen3-8B LoRA, 37-task matched slice (identical task set
# to Paper 1's s1 runs), both decision arms, 3 seeds. Waits for the 8B download.
set -u
cd /root/autodl-tmp/jev_gene
QD="python3 -u scripts/benchmark_v2/qwen_decision.py"
TASKS=$(python3 -c "
import json
c=json.load(open('artifacts/benchmark_v2_train/s1_20261001/run_config.json'))
print(' '.join(c['tasks']))")
COMMON="--model models/Qwen3-8B --epochs 12 --batches-per-family 30 --max-per-task 2048 --dev-max 256 --dev-eval-every 150"

while ! grep -q "ALL DONE" dl_qwen3_8b.log 2>/dev/null; do sleep 60; done
echo "download confirmed $(date)"
echo "########## QWEN38 MAIN START $(date) ##########"
# scale low point first (0.6B, 1 seed, both arms; ~1h total)
for arch in heads shared; do
  $QD --model models/Qwen3-0.6B --arch $arch --tasks $TASKS $COMMON --mb 4 --accum 4 \
      --seed 20261001 --name q06_${arch}_s20261001 > q06_${arch}.log 2>&1 \
      || echo "!!! q06_${arch} failed"
  find artifacts/benchmark_v2_train/q06_${arch}_s20261001 -name "*.safetensors" -delete 2>/dev/null
done
echo "===== 0.6B low point done $(date) ====="
for s in 20261001 20261002 20261003; do
  for arch in heads shared; do
    echo "===== [$(date +%H:%M)] 8B $arch seed $s ====="
    $QD --arch $arch --tasks $TASKS $COMMON --qlora --mb 4 --accum 4 --batches-per-family 20 --seed $s --name q8_${arch}_s$s > q8_${arch}_s$s.log 2>&1 \
      || { echo "!!! q8_${arch}_s$s failed"; continue; }
    find artifacts/benchmark_v2_train/q8_${arch}_s$s -name "*.safetensors" -delete 2>/dev/null
    echo "===== [$(date +%H:%M)] q8_${arch}_s$s done; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') ====="
  done
done
echo "########## QWEN38 MAIN DONE $(date) ##########"
