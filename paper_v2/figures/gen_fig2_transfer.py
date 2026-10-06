#!/usr/bin/env python3
"""Fig 2: the three operational checks on #1's claimed value.
(a) zero-shot transfer gradient (same-family / cross-family / in-domain reference),
(b) cross-family score arm on TAPE, 3 seeds,
(c) adaptation cost: per-task-head few-shot fit vs #1 zero-shot."""
import numpy as np
from paper_plot_style import *

zs = load(AE / "zeroshot_transfer.json")            # same-family held-out
xseed = load(AE / "cross_family_seedavg.json")      # GB cross-family 3-seed
tape3 = load(AE / "cross_family_tape_3seed.json")   # TAPE cross-family 3-seed
paired = load(AE / "full_paired_summary.json")      # in-domain reference
adapt = load(AE / "adapt_cost.json")                # per-task few-shot
asum = load(AE / "adapt_cost_summary.json")

fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), gridspec_kw=dict(width_ratios=[0.95, 1.08, 1.35], wspace=0.36))

# ---- (a) transfer gradient, classification arms --------------------------
ax = axes[0]
gb = xseed["gb_3seed"]
vals = [zs["by_family"]["GUE"]["delta"], zs["by_family"]["ProteinGym"]["delta"],
        gb["mean"], paired["mean_delta"]]
errs = [0, 0, gb["std"] / np.sqrt(len(gb["per_seed"])), 0]
labels = ["same-fam\nGUE", "same-fam\nProtGym", "cross-fam\nGB (3-seed)", "in-domain\nref."]
cols = [C_SHARED, C_SHARED, C_SHARED, C_GEN]
bars = ax.bar(range(4), vals, color=cols, width=0.62)
bars[3].set_hatch("//"); bars[3].set_edgecolor("white")
bars[1].set_alpha(0.55)
ax.errorbar(range(4), vals, yerr=errs, fmt="none", ecolor="black", capsize=2, lw=0.8)
ax.axhline(0, color="black", lw=0.8)
for i, v in enumerate(vals):
    ax.text(i, v + (0.008 if v >= 0 else -0.008), f"{v:+.3f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=6.4)
ax.set_xticks(range(4))
ax.set_xticklabels(["GUE\nsame-fam", "ProtGym\nsame-fam", "GB\ncross-fam", "in-domain\nref."], fontsize=5.6)
ax.set_ylabel("Δ metric (joint-trained − frozen base)", fontsize=7.5)
ax.set_ylim(-0.06, 0.145)
ax.tick_params(axis="y", labelsize=7)
ax.text(0.0, 1.02, "(a) zero-shot transfer", fontsize=7.2, ha="left", va="bottom",
        transform=ax.transAxes)

# ---- (b) TAPE score, 3 seeds ---------------------------------------------
ax = axes[1]
w = 0.32
for i, task in enumerate(["tape_fluorescence", "tape_stability"]):
    t = tape3["per_task"][task]
    fr = np.mean([s["frozen"] for s in t["per_seed"].values()])
    tr = np.mean([s["trained"] for s in t["per_seed"].values()])
    trs = [s["trained"] for s in t["per_seed"].values()]
    sem = t["std_delta"] / np.sqrt(len(trs))
    ax.bar(i - w / 2, fr, w, color=C_GEN, label="frozen base" if i == 0 else "")
    ax.bar(i + w / 2, tr, w, color=C_SHARED, label="joint-trained (#1)" if i == 0 else "")
    ax.plot([i + w / 2] * 3, trs, "k.", ms=3, alpha=0.6)
    ax.errorbar(i + w / 2, tr, yerr=sem, fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.text(i, 0.085, f"Δ={t['mean_delta']:+.3f}\n±{t['std_delta']:.3f}",
            ha="center", fontsize=6.2)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(2))
ax.set_xticklabels(["fluorescence", "stability"], fontsize=7)
ax.set_ylabel("Spearman (vs assay target)", fontsize=7.5)
ax.set_ylim(-0.26, 0.145)
ax.legend(loc="lower left", frameon=False, fontsize=6.2, bbox_to_anchor=(0.0, 0.18))
ax.text(0.0, 1.02, "(b) cross-family score (3-seed)", fontsize=7.2, ha="left", va="bottom",
        transform=ax.transAxes)

# ---- (c) adaptation cost --------------------------------------------------
ax = axes[2]
res = adapt["results"]
names, deltas32, xs128, xs512, is_gue = [], [], [], [], []
for tid, r in res.items():
    short = tid.replace("gue_", "G:").replace("pg_", "P:").split("_")[0]
    names.append(short)
    zs1 = r["zero_shot_1"]
    deltas32.append(r["head_fewshot"].get("32", np.nan) - zs1)
    xs128.append(r["head_fewshot"].get("128", np.nan) - zs1)
    xs512.append(r["head_fewshot"].get("512", np.nan) - zs1)
    is_gue.append(tid.startswith("gue"))
x = np.arange(len(names))
bcols = ["#ff9896" if g else C_HEAD for g in is_gue]
ax.bar(x, deltas32, color=bcols, width=0.55,
       label=None)
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color="#ff9896", label="GUE (acc)"),
                   mpatches.Patch(color=C_HEAD, label="ProtGym (Spearman)")],
          loc="upper right", frameon=False, fontsize=6.0)
ax.plot(x, xs128, "kx", ms=4, mew=0.9, label="N=128")
good512 = [(i, v) for i, v in enumerate(xs512) if not np.isnan(v)]
if good512:
    ax.plot([i for i, _ in good512], [v for _, v in good512], marker="$+$", ls="none",
            ms=5, color="black", label="N=512")
ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=62, ha="right", fontsize=6.0)
ax.set_ylabel("head@32 − #1 zero-shot (per-task metric)", fontsize=7.5)
n32 = asum["match_N_distribution"].get("32", 0)
tot = sum(asum["match_N_distribution"].values())
ax.text(0.02, 0.80, f"{n32}/{tot} tasks catch #1 at N=32", transform=ax.transAxes,
        fontsize=6.6, va="top")
ax.legend(loc="upper left", frameon=False, fontsize=6.0)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylim(-0.24, 0.62)
ax.text(0.0, 1.02, "(c) adaptation cost (held-out tasks)", fontsize=7.2, ha="left",
        va="bottom", transform=ax.transAxes)
save_fig(fig, "fig2_transfer")
