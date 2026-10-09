#!/bin/bash
# After q06_fixup + q8_retry drain, re-run anything still missing eval_results.json.
set -u
cd /root/autodl-tmp/jev_gene
QD="python3 -u scripts/benchmark_v2/qwen_decision.py"
TASKS=$(python3 -c "
import json
c=json.load(open('artifacts/benchmark_v2_train/s1_20261001/run_config.json'))
print(' '.join(c['tasks']))")
while pgrep -f "q06_fixu[p].sh" > /dev/null || pgrep -f "q8_retr[y].sh" > /dev/null; do sleep 60; done
echo "########## FINAL SWEEP START $(date) ##########"
run_one() {  # name arch model extra...
  local name=$1 arch=$2 model=$3; shift 3
  if [ -f "artifacts/benchmark_v2_train/$name/eval_results.json" ]; then return; fi
  echo "===== [$(date +%H:%M)] sweep $name ====="
  $QD --arch $arch --tasks $TASKS --model $model --epochs 12 --max-per-task 2048 \
      --dev-max 256 --dev-eval-every 150 --seed 20261001 "$@" --name $name > sweep_$name.log 2>&1 \
    || echo "!!! sweep $name failed"
  find artifacts/benchmark_v2_train/$name -name "*.safetensors" -delete 2>/dev/null
}
run_one q06_heads_s20261001 heads models/Qwen3-0.6B --batches-per-family 40 --mb 2 --accum 1
run_one q06_shared_s20261001 shared models/Qwen3-0.6B --batches-per-family 40 --mb 1 --accum 2
for s in 20261001 20261002 20261003; do
  for arch in heads shared; do
    if [ "$arch" = "shared" ]; then EXTRA="--mb 1 --accum 2"; else EXTRA="--mb 2 --accum 1"; fi
    run_one q8_${arch}_s$s $arch models/Qwen3-8B --batches-per-family 40 --qlora $EXTRA --seed $s
  done
done
echo "########## FINAL SWEEP DONE $(date) ##########"
