#!/usr/bin/env python3
"""CPT-contribution ablation: joint-37 seed-averaged comparison on the two encoder
inits (no_cpt = seed_avg_three_way.json, cpt = seed_avg_three_way_cpt.json).
Prints and saves the delta table; feeds the S5.5/S6 arm-robustness sentence."""
import json
from paper_plot_style import *

a = load(AE / "seed_avg_three_way.json")
b = load(AE / "seed_avg_three_way_cpt.json")
out = {}
print(f"{'primitive':8s} {'metric':6s} {'no_cpt #1':>10s} {'cpt #1':>8s} {'no_cpt #2':>10s} {'cpt #2':>8s}")
for pr in ("choice", "noul", "score"):
    if pr not in b:
        continue
    m = a[pr]["metric"]
    row = {"metric": m,
           "no_cpt_shared": a[pr]["shared_mean"], "cpt_shared": b[pr]["shared_mean"],
           "no_cpt_head": a[pr]["taskhead_mean"], "cpt_head": b[pr]["taskhead_mean"],
           "no_cpt_sem": a[pr]["shared_sem"], "cpt_sem": b[pr]["shared_sem"]}
    out[pr] = row
    print(f"{pr:8s} {m:6s} {a[pr]['shared_mean']:10.3f} {b[pr]['shared_mean']:8.3f} "
          f"{a[pr]['taskhead_mean']:10.3f} {b[pr]['taskhead_mean']:8.3f}")
w_a, w_b = a.get("per_task_wins_seedmean"), b.get("per_task_wins_seedmean")
out["wins"] = {"no_cpt": w_a, "cpt": w_b}
print("wins no_cpt:", w_a, "| wins cpt:", w_b)
(AE / "cpt_arm_ablation.json").write_text(json.dumps(out, indent=2) + "\n")
print("saved", AE / "cpt_arm_ablation.json")
