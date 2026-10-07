#!/usr/bin/env python3
"""Appendix table: seed-averaged three-way comparison on the CPT encoder init,
side by side with the no_cpt main-table numbers (robustness of the ordering)."""
import json
from paper_plot_style import *

a = load(AE / "seed_avg_three_way.json")       # no_cpt (main)
b = load(AE / "seed_avg_three_way_cpt.json")   # cpt
ab = load(AE / "cpt_arm_ablation.json")

tex = [r"% Table 9: cpt-arm three-way comparison (gen_cpt_table.py)",
       r"\begin{table}[t]", r"\centering",
       r"\caption{Seed-averaged three-way comparison on the biology-CPT encoder init "
       r"(3 seeds, same protocol as \cref{tab:main}). The architecture ordering is "
       r"invariant to the encoder init: per-task heads match or beat the shared scorer on "
       r"every primitive under both inits; CPT lifts both arms on noul "
       r"($+0.041$/$+0.022$) without closing the gap.}",
       r"\label{tab:main_cpt}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
       r"\begin{tabular}{llcccc}", r"\toprule",
       r"Primitive & Metric & $n$ & \#1 shared (cpt) & \#2 heads (cpt) & \#3 frozen \\",
       r"\midrule"]
for g in ("choice", "noul", "score"):
    d = b[g]
    g3 = "--" if d["gen3_mean"] is None else f"{d['gen3_mean']:.3f}"
    tex.append(f"{g} & {d['metric']} & {d['n']} & "
               f"{d['shared_mean']:.3f}$\\pm${d['shared_sem']:.3f} & "
               f"{d['taskhead_mean']:.3f}$\\pm${d['taskhead_sem']:.3f} & {g3} \\\\")
w = b["per_task_wins_seedmean"]
tex += [r"\addlinespace",
        r"Per-task seed-mean wins (\#1 / tie / \#2) & & & \multicolumn{3}{r}{"
        + f"{w['shared']} / {w['tie']} / {w['taskhead']}" + r"} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(FIG_DIR / "TABLE_9_cpt_main.tex").write_text("\n".join(tex))
print("wrote TABLE_9_cpt_main.tex")
