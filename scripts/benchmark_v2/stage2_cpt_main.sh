#!/bin/bash
# Stage 2: seed-averaged three-way comparison on the CPT arm (biology continued-
# pretrained encoder init), mirroring seed_avg_driver.sh exactly + --arm cpt.
# Names s1c_/s2c_; per-seed weight pruning after eval (disk budget).
set -u
cd /root/autodl-tmp/jev_gene
SEEDS="20261001 20261002 20261003"
COMMON="--all --family-balance --batches-per-family 30 --epochs 12 --max-per-task 2048 --min-valid 128 --cap-family ProteinGym=12 GUE=12 --warmup-frac 0.15 --clip 0.5 --dev-eval-every 150 --dev-max 256 --per-family-early-stop --arm cpt"

echo "########## STAGE2 CPT MAIN START $(date) ##########"
for s in $SEEDS; do
  echo "########## SEED $s : #1 shared scorer (cpt) ##########"
  python3 -u scripts/benchmark_v2/train_benchmark_v2.py $COMMON --seed $s --name s1c_$s > seed_s1c_$s.log 2>&1
  python3 -u - "$s" << 'PY'
import json, subprocess, sys
from collections import defaultdict
s=sys.argv[1]
c=json.load(open(f"artifacts/benchmark_v2_train/s1c_{s}/run_config.json"))
FAM=[("pg_","ProteinGym"),("gue_","GUE"),("gb_","GenomicBenchmarks"),("gl_","gene_lan"),("dna_","dnagpt_pools"),("deepstarr","DeepSTARR"),("tape_","TAPE"),("lg_","local_snapshots"),("protein_homology","local_snapshots"),("deeploc","DeepLoc"),("rnac_","RNAcompete")]
def fam(t):
    for p,f in FAM:
        if t.startswith(p): return f
    return "other"
byfam=defaultdict(list)
for t in c["tasks"]: byfam[fam(t)].append(t)
for f,ts in byfam.items():
    ck=f"artifacts/benchmark_v2_train/s1c_{s}/best_{f}.safetensors"
    subprocess.run(["python3","-u","scripts/benchmark_v2/eval_benchmark_v2.py","--ckpt",ck,"--tasks",*ts,"--max-per-task","200","--out",f"artifacts/benchmark_v2_eval/s1c_{s}_{f}"],capture_output=True)
print(f"  #1 cpt seed {s} per-family eval done")
PY
  find artifacts/benchmark_v2_train/s1c_$s -name "*.safetensors" -delete
  echo "  pruned s1c_$s; disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')"

  echo "########## SEED $s : #2 task heads (cpt) ##########"
  python3 -u scripts/benchmark_v2/train_taskheads.py --task-config artifacts/benchmark_v2_train/s1c_$s/run_config.json \
    --epochs 12 --batches-per-family 30 --max-per-task 2048 --min-valid 128 --eval-max 200 \
    --warmup-frac 0.15 --clip 0.5 --dev-eval-every 150 --dev-max 256 --per-family-early-stop \
    --arm cpt --seed $s --name s2c_$s > seed_s2c_$s.log 2>&1
  find artifacts/benchmark_v2_train/s2c_$s -name "*.safetensors" -delete
  echo "  #2 cpt seed $s done+pruned: $(grep -c '#2 DONE' seed_s2c_$s.log); disk $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')"
done
echo "STAGE2_CPT_ALL_DONE $(date)"
