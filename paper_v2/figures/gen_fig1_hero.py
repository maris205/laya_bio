#!/usr/bin/env python3
"""Fig 1 (hero): (a) benchmark panorama 12-source families x 4 decision primitives,
(b) seed-averaged three-architecture comparison per primitive with variance inset."""
import numpy as np
from collections import defaultdict
from paper_plot_style import *

man = load(UNI / "manifest.json")
tasks = {k: v for k, v in man.items() if v.get("status") == "ready"}
QGROUPS = qgroups_conflict()
print(f"ready tasks: {len(tasks)} / {len(man)} | quarantined groups: {QGROUPS}")

# ---- (a) family x primitive count matrix -------------------------------
prims = ["choice", "noul", "score", "multi_noul"]
fam_count = defaultdict(lambda: defaultdict(int))
fam_rows = defaultdict(int)
fam_mods = defaultdict(set)
tot_prim = defaultdict(int)
tot_split = defaultdict(int)
qgroups = 0
for tid, v in tasks.items():
    f = family_of(tid)
    fam_count[f][v["primitive"]] += 1
    fam_rows[f] += 1
    fam_mods[f].update(v["modality"])
    tot_prim[v["primitive"]] += 1
    for s in ("train", "dev", "test"):
        tot_split[s] += v.get(s, 0)
    qgroups += v.get("quarantined_groups", 0)
fams = sorted(fam_count, key=lambda f: -fam_rows[f])
mat = np.array([[fam_count[f][p] for p in prims] for f in fams], dtype=float)
MODTAG = {"dna": "DNA", "protein": "Prot", "rna": "RNA"}
fam_labels = []
for f in fams:
    mods = "+".join(MODTAG[m] for m in ("dna", "protein", "rna") if m in fam_mods[f])
    fam_labels.append(f"{f}  [{mods}]")
print("primitive totals:", dict(tot_prim))
print("split totals:", dict(tot_split), "| quarantined groups:", qgroups)
print("family modality:", fam_labels)

# ---- (b) seed-avg three-way --------------------------------------------
sa = load(AE / "seed_avg_three_way.json")
pf = load(AE / "three_way_pf_summary.json")      # single-run A (bv2_pf)
pf2 = load(AE / "three_way_pf2_summary.json")    # same-config rerun B (bv2_pf2)
groups = [("choice", "acc"), ("noul", "AUROC"), ("score", "Spearman")]

fig = plt.figure(figsize=(6.9, 2.85))
gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.30)

# ---- panel a
axa = fig.add_subplot(gs[0])
im = axa.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=mat.max())
axa.set_xticks(range(len(prims)))
axa.set_xticklabels([p.replace("_", "-") for p in prims], fontsize=7.5)
axa.set_yticks(range(len(fams)))
axa.set_yticklabels(fam_labels, fontsize=7.0)
axa.tick_params(length=0)
for i in range(len(fams)):
    for j in range(len(prims)):
        v = mat[i, j]
        if v > 0:
            axa.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.0,
                     color="white" if v > mat.max() * 0.5 else "#1a3a5c")
        else:
            axa.text(j, i, "–", ha="center", va="center", fontsize=6.5, color="#9aa5b1")
for spine in axa.spines.values():
    spine.set_visible(False)
axa.set_xlabel("decision primitive (cell = number of tasks)")
axa.text(-0.06, 1.06, f"{len(tasks)} leak-audited tasks = {len(fams)} source families x {len(prims)} primitives",
         transform=axa.transAxes, fontsize=7.0, color="#333333", clip_on=False)
axa.text(-0.06, 1.14,
         f"0 train/test sequence leaks  ·  {QGROUPS:,} conflict groups quarantined",
         transform=axa.transAxes, fontsize=7.0, color="#333333", clip_on=False)

# ---- panel b
axb = fig.add_subplot(gs[1])
x = np.arange(len(groups))
w = 0.26
specs = []
for i, (g, _m) in enumerate(groups):
    d = sa[g]
    specs.append((g, d))
    # #1 shared
    axb.bar(i - w, d["shared_mean"], w, yerr=d["shared_sem"], color=C_SHARED,
            capsize=1.5, error_kw=dict(lw=0.8), label="#1 shared (3-seed)" if i == 0 else "")
    # #2 task heads
    axb.bar(i, d["taskhead_mean"], w, yerr=d["taskhead_sem"], color=C_HEAD,
            capsize=1.5, error_kw=dict(lw=0.8), label="#2 task heads (3-seed)" if i == 0 else "")
    # #3 frozen LM
    if d["gen3_mean"] is not None:
        axb.bar(i + w, d["gen3_mean"], w, color=C_GEN,
                label="#3 frozen LM*" if i == 0 else "")
    else:
        axb.text(i + w, 0.02, "n/a*", ha="center", va="bottom", fontsize=6.8, color=C_GEN)
axb.axhline(0.0, color="black", lw=0.8)
axb.plot([-0.48, 1.40], [0.5, 0.5], ls=":", color="black", lw=0.9, clip_on=False)
axb.text(-0.45, 0.505, "chance", fontsize=6.5, va="bottom", ha="left")
axb.set_xticks(x)
axb.set_xticklabels([f"{g}\n({m})" for g, m in groups])
axb.set_ylabel("task-macro metric")
axb.set_ylim(-0.05, 1.08)
axb.legend(loc="upper left", frameon=False, handlelength=1.1, fontsize=6.4)
axb.text(0.99, -0.16, "*#3 not scored for Spearman (anchor protocol differs)",
         transform=axb.transAxes, fontsize=6.2, color="#555555", ha="right", va="top")
# noul significance marker between #1 and #2
d = sa["noul"]
hi = max(d["shared_mean"] + d["shared_sem"], d["taskhead_mean"] + d["taskhead_sem"])
axb.annotate("", xy=(1 + w * 0.5, hi + 0.085), xytext=(1 - w * 0.5, hi + 0.085),
             arrowprops=dict(arrowstyle="|-|", lw=0.8))
axb.text(1, hi + 0.10, "SEM disjoint", fontsize=6.5, ha="center")

# inset: single-run variance case (score Spearman, #1)
axins = axb.inset_axes([0.64, 0.28, 0.36, 0.34])
vals = [pf["by_primitive"]["score"]["shared"],        # run A
        pf2["by_primitive"]["score"]["shared_old"],   # run A copy field naming differs
        sa["score"]["shared_mean"]]
vals[1] = pf2["by_primitive"]["score"]["shared_new"]
labels = ["run\nA", "run\nB", "3-seed\nmean"]
cols = [C_INIT, C_INIT, C_SHARED]
axins.bar(range(3), vals, color=cols, width=0.62)
axins.axhline(sa["score"]["taskhead_mean"], color=C_HEAD, ls="--", lw=0.9)
axins.set_xticks(range(3))
axins.set_xticklabels(labels, fontsize=6.0)
axins.set_ylim(0, 0.13)
axins.set_xlim(-0.75, 2.75)
axins.tick_params(labelsize=6.0, length=2)
for sp in ("top", "right"):
    axins.spines[sp].set_visible(True)
    axins.spines[sp].set_linewidth(0.5)
axins.set_yticks([0, 0.05, 0.1])
axins.text(0.0, 1.03, "Spearman, #1: runs vs mean", fontsize=5.6,
           transform=axins.transAxes, va="bottom")
save_fig(fig, "fig1_hero")
