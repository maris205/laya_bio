#!/bin/bash
# 0.6B scale low-point fixup: the main driver's 0.6B section accidentally ran the
# 8B weights (COMMON carried --model and argparse last-wins). Runs the two 0.6B
# arms properly after the 8B queue drains.
set -u
cd /root/autodl-tmp/jev_gene
QD="python3 -u scripts/benchmark_v2/qwen_decision.py"
TASKS=$(python3 -c "
import json
c=json.load(open('artifacts/benchmark_v2_train/s1_20261001/run_config.json'))
print(' '.join(c['tasks']))")
while pgrep -f "qwen38_main_drive[r].sh" > /dev/null; do sleep 60; done
echo "########## Q06 FIXUP START $(date) ##########"
for arch in heads shared; do
  $QD --arch $arch --tasks $TASKS --model models/Qwen3-0.6B \
      --epochs 12 --batches-per-family 30 --max-per-task 2048 --dev-max 256 \
      --dev-eval-every 150 --mb 8 --accum 2 --seed 20261001 \
      --name q06_${arch}_s20261001 > q06_${arch}_fix.log 2>&1 \
    || echo "!!! q06_${arch} fixup failed"
  find artifacts/benchmark_v2_train/q06_${arch}_s20261001 -name "*.safetensors" -delete 2>/dev/null
  echo "===== [$(date +%H:%M)] q06_${arch} done ====="
done
echo "########## Q06 FIXUP DONE $(date) ##########"
