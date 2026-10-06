#!/usr/bin/env python3
"""Fig 4: controlled mechanism ablations.
(a) noul candidate phrasing, 6 texts x 3 seed-averaged stable checkpoints
    (single-variable: rendered candidate text; same checkpoint per seed);
(b) frozen-VLM candidate likelihood: plain vs contrastive decoding (route A pilot)."""
import numpy as np
from paper_plot_style import *

ab = load(AE / "noul_ablation_3seed.json")
sa = load(AE / "seed_avg_three_way.json")
vlm_p = load(AE / "vlm_fold" / "eval_results.json")
vlm_c = load(AE / "vlm_fold_contrastive" / "eval_results.json")

NAME = {"current_false_no": "false: no /\ntrue: yes", "plain_false_true": "false / true",
        "proposition": "proposition\n(\"yes, it does\")", "raw_no_yes": "no / yes\n(trained)",
        "descriptive": "descriptive", "neg_pos_words": "neg / pos"}

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw=dict(width_ratios=[1.45, 1.0], wspace=0.26))

# ---- (a) phrasing x 3 checkpoints ----------------------------------------
ax = axes[0]
items = sorted(ab["per_phrasing"].items(), key=lambda kv: -kv[1]["mean"])
x = np.arange(len(items))
means = [v["mean"] for _, v in items]
stds = [v["std"] for _, v in items]
cols = [C_SHARED if k != ab["trained_phrasing"] else "#7fa8c9" for k, _ in items]
bars = ax.bar(x, means, yerr=stds, color=cols, capsize=2, error_kw=dict(lw=0.8), width=0.6)
ti = [i for i, (k, _) in enumerate(items) if k == ab["trained_phrasing"]][0]
bars[ti].set_hatch("//")
for xi, (k, v) in zip(x, items):
    ax.plot([xi] * len(v["per_seed"]), v["per_seed"], "k.", ms=3, alpha=0.55)
ax.axhline(sa["noul"]["taskhead_mean"], color=C_HEAD, ls="--", lw=1.0)
ax.text(len(items) - 0.45, sa["noul"]["taskhead_mean"] + 0.006,
        "#2 task heads (seed avg) = 0.629", fontsize=6.3, color=C_HEAD, ha="right")
ax.axhline(0.5, color="black", ls=":", lw=0.9)
ax.text(5.45, 0.502, "chance", fontsize=6.3, va="bottom", ha="right")
ax.set_xticks(x); ax.set_xticklabels([NAME[k] for k, _ in items], fontsize=6.3)
ax.set_ylabel("noul AUROC (GB tasks, 3-seed avg ckpts)", fontsize=7.5)
ax.set_ylim(0.40, 0.68)
ax.text(0.0, 1.02, "(a) phrasing: none reaches #2", fontsize=7.2, ha="left", va="bottom",
        transform=ax.transAxes)

# ---- (b) VLM decoding ------------------------------------------------------
ax = axes[1]
vals = [vlm_p["accuracy"], vlm_c["accuracy"]]
np_ = sum(1 for v in vlm_p["pred_dist"].values() if v > 0)
nc_ = sum(1 for v in vlm_c["pred_dist"].values() if v > 0)
labels = [f"plain\n({np_}/7 preds used)", f"contrastive\n({nc_}/7 preds used)"]
ax.bar(range(2), vals, color=["#8c8c8c", "#4d4d4d"], width=0.5)
for i, v in enumerate(vals):
    ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.8)
maj = 0.325; rnd = 1 / 7
ax.axhline(maj, color="black", ls="--", lw=0.9)
ax.text(1.55, maj + 0.005, "majority class", fontsize=6.2, ha="right")
ax.axhline(rnd, color="black", ls=":", lw=0.9)
ax.text(1.55, rnd + 0.005, "chance (1/7)", fontsize=6.2, ha="right")
ax.set_xticks(range(2)); ax.set_xticklabels(labels, fontsize=6.6)
ax.set_ylabel("fold-class accuracy (synthetic property maps)", fontsize=7.5)
ax.set_ylim(0, 0.40)
ax.text(0.0, 1.02, "(b) frozen VLM: decoding fixed, signal absent", fontsize=7.2,
        ha="left", va="bottom", transform=ax.transAxes)
save_fig(fig, "fig4_ablation")
