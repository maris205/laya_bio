#!/bin/bash
set -e
cd /root/autodl-tmp/jev_gene
echo "########## PIPELINE START $(date) ##########"
echo "[1/4] build"; rm -rf data/06_benchmark_v2_unified/*; python3 -u scripts/benchmark_v2/build_benchmark.py
echo "[2/4] isolate"; python3 -u scripts/benchmark_v2/isolate_splits.py
echo "[3/4] admission check"; python3 -u scripts/benchmark_v2/check_admission.py
echo "[4/4] finalize manifest"; python3 -u scripts/benchmark_v2/finalize_manifest.py
echo "########## PIPELINE DONE $(date) ##########"
