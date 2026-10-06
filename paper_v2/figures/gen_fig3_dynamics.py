#!/usr/bin/env python3
"""Fig 3: training dynamics.
(a) budget scaling and stabilization (mean dev metric), with stabilized dev curve inset;
(b) per-family peak dev steps (same trajectory spans 26x) vs the single global pick;
(c) per-family early stop beats one global checkpoint (test-metric delta)."""
import numpy as np
from paper_plot_style import *

bud = load(AE / "budget_scaling_summary.json")
stab = load(AE / "stabilization_summary.json")
dv = load_jsonl(TR / "bv2_stab" / "dev_curve.jsonl")
pf_dv = load_jsonl(TR / "bv2_pf" / "dev_curve.jsonl")
pfg = load(AE / "per_family_vs_global.json")

fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.05], wspace=0.50))

# ---- (a) budget bars + dev-curve inset -----------------------------------
ax = axes[0]
c = bud["cls_all"]
vals = [c["init"], c["small"], c["big"], stab["cls_all"]["stab"]]
labels = ["init", "2520\nupd", "5600\nnaive", "5600\nstab."]
cols = [C_GEN, C_SHARED, "#ae3a3a", C_ACC]
ax.bar(range(4), vals, color=cols, width=0.6)
ax.axhline(c["init"], color=C_GEN, ls=":", lw=0.9)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.4)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6.9)
ax.set_ylim(0.40, 0.545)
ax.set_ylabel("dev metric (task-macro)", fontsize=7.5)
axins = ax.inset_axes([0.50, 0.62, 0.46, 0.32])
steps = [r["step"] for r in dv]; sc = [r["dev_score"] for r in dv]
axins.plot(steps, sc, color=C_ACC, lw=0.9)
kb = int(np.argmax(sc))
axins.plot(steps[kb], sc[kb], "k*", ms=5)
axins.tick_params(labelsize=5.2, length=1.5)
for sp in ("top", "right"):
    axins.spines[sp].set_visible(True); axins.spines[sp].set_linewidth(0.4)
axins.set_xlabel("updates", fontsize=5.2, labelpad=0.5)
ax.text(0.0, 1.02, "(a) budget & stabilization", fontsize=7.2, ha="left", va="bottom",
        transform=ax.transAxes)

# ---- (b) per-family peak steps on the same trajectory --------------------
ax = axes[1]
fams = sorted({f for r in pf_dv for f in r["per_family"]})
gsc = [r["dev_score"] for r in pf_dv]
gstep = pf_dv[int(np.argmax(gsc))]["step"]
y = np.arange(len(fams))
for i, f in enumerate(fams):
    sc = [(r["step"], r["per_family"].get(f)) for r in pf_dv if f in r["per_family"]]
    st = max(sc, key=lambda t: t[1])[0]
    ax.plot(st, i, "o", color=C_SHARED, ms=4.5)
ax.axvline(gstep, color="#ae3a3a", ls="--", lw=1.0)
ax.text(gstep * 1.06, len(fams) - 0.4, f"global best\nstep {gstep}", fontsize=6.0,
        color="#ae3a3a", va="top")
ax.set_yticks(y)
ax.set_yticklabels([f.replace("_", "-") for f in fams], fontsize=6.4)
ax.set_xscale("log")
ax.set_xticks([200, 500, 1000, 2000, 5000])
ax.set_xticklabels(["200", "500", "1k", "2k", "5k"], fontsize=6.6)
ax.set_xlim(150, 9000)
ax.set_xlabel("dev peak step (same training trajectory)", fontsize=7.5)
ax.grid(axis="x", ls=":", lw=0.4, alpha=0.6)
ax.text(0.0, 1.02, "(b) family peaks: 26x spread", fontsize=7.2, ha="left", va="bottom",
        transform=ax.transAxes)

# ---- (c) per-family vs global checkpoint on test -------------------------
ax = axes[2]
bf = pfg["by_family"]
fams3 = sorted(bf, key=lambda f: -(bf[f]["per_family"] - bf[f]["global"]))
deltas = [bf[f]["per_family"] - bf[f]["global"] for f in fams3]
ax.barh(range(len(fams3)), deltas, color=[C_ACC if d > 0 else "#ae3a3a" for d in deltas],
        height=0.55)
for i, (f, d) in enumerate(zip(fams3, deltas)):
    ax.text(d + (0.004 if d >= 0 else -0.004), i, f"{d:+.3f}", va="center",
            ha="left" if d >= 0 else "right", fontsize=6.2)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(fams3)))
ax.set_yticklabels([f.replace("_", "-") for f in fams3], fontsize=6.4)
ax.set_xlabel("Δ test metric (per-family ckpt − global ckpt)", fontsize=7.5)
ax.set_xlim(-0.02, 0.15)
w = pfg["wins"]
ax.text(0.98, 0.94, f"wins {w['per_family']} / tie {w['tie']} / loss {w['global']}",
        transform=ax.transAxes, fontsize=6.3, va="top", ha="right")
ax.text(0.0, 1.02, "(c) zero-cost early stop", fontsize=7.2, ha="left", va="bottom",
        transform=ax.transAxes)
save_fig(fig, "fig3_dynamics")
