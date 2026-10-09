#!/bin/bash
# Re-run any q8 run missing eval_results.json after main driver + 0.6B fixup drain.
set -u
cd /root/autodl-tmp/jev_gene
QD="python3 -u scripts/benchmark_v2/qwen_decision.py"
TASKS=$(python3 -c "
import json
c=json.load(open('artifacts/benchmark_v2_train/s1_20261001/run_config.json'))
print(' '.join(c['tasks']))")
while pgrep -f "qwen38_main_drive[r].sh" > /dev/null || pgrep -f "q06_fixu[p].sh" > /dev/null; do sleep 60; done
echo "########## Q8 RETRY START $(date) ##########"
for s in 20261001 20261002 20261003; do
  for arch in heads shared; do
    D=artifacts/benchmark_v2_train/q8_${arch}_s$s
    if [ -f "$D/eval_results.json" ]; then continue; fi
    echo "===== [$(date +%H:%M)] retry q8_${arch}_s$s ====="
    $QD --arch $arch --tasks $TASKS --model models/Qwen3-8B --epochs 12 --batches-per-family 20 \
        --max-per-task 2048 --dev-max 256 --dev-eval-every 150 --qlora --mb 2 --accum 8 \
        --seed $s --name q8_${arch}_s$s > q8_${arch}_s$s.log 2>&1 || echo "!!! retry q8_${arch}_s$s failed"
    find $D -name "*.safetensors" -delete 2>/dev/null
  done
done
echo "########## Q8 RETRY DONE $(date) ##########"
