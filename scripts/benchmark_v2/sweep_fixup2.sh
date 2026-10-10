#!/bin/bash
# After final_sweep drains, re-run any run still missing eval_results.json (once).
set -u
exec 9>/tmp/sweep_fixup2.sh.lock
flock -n 9 || { echo another-instance-running; exit 0; }
cd /root/autodl-tmp/jev_gene
QD="python3 -u scripts/benchmark_v2/qwen_decision.py"
TASKS=$(python3 -c "
import json
c=json.load(open('artifacts/benchmark_v2_train/s1_20261001/run_config.json'))
print(' '.join(c['tasks']))")
while pgrep -f "final_swee[p].sh" > /dev/null; do sleep 60; done
echo "########## FIXUP2 START $(date) ##########"
for spec in "q06_heads_s20261001 heads models/Qwen3-0.6B" "q06_shared_s20261001 shared models/Qwen3-0.6B" \
            "q8_heads_s20261001 heads models/Qwen3-8B" "q8_shared_s20261001 shared models/Qwen3-8B" \
            "q8_heads_s20261002 heads models/Qwen3-8B" "q8_shared_s20261002 shared models/Qwen3-8B" \
            "q8_heads_s20261003 heads models/Qwen3-8B" "q8_shared_s20261003 shared models/Qwen3-8B"; do
  set -- $spec; name=$1; arch=$2; model=$3
  [ -f "artifacts/benchmark_v2_train/$name/eval_results.json" ] && continue
  echo "===== [$(date +%H:%M)] fixup2 $name ====="
  QL=""; case $model in *Qwen3-8B*) QL="--qlora";; esac
  $QD --arch $arch --tasks $TASKS --model $model --epochs 12 --batches-per-family 20 \
      --max-per-task 2048 --dev-max 256 --dev-eval-every 150 --mb 1 --accum 1 \
      $QL --seed ${name##*_s} --name $name > fix2_$name.log 2>&1 \
    || echo "!!! fixup2 $name failed"
  find artifacts/benchmark_v2_train/$name -name "*.safetensors" -delete 2>/dev/null
done
echo "########## FIXUP2 DONE $(date) ##########"
