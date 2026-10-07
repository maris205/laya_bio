#!/bin/bash
# Wait for the leaderboard driver to finish, then run stage 2 (cpt main comparison).
set -u
cd /root/autodl-tmp/jev_gene
while pgrep -f "leader[b]oard_driver" > /dev/null; do sleep 60; done
echo "leaderboard finished at $(date); starting stage2" >> chain.log
bash scripts/benchmark_v2/stage2_cpt_main.sh >> chain.log 2>&1
