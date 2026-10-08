#!/usr/bin/env python3
"""Fig 5: granularity gradient. For five representative tasks, the #2-arm metric
under four training granularities: single-task / family-joint / all-241 joint /
joint-37 matched slice (3-seed cpt mean). Shows interference cost monotonically
(as available) and closes the single-task anchor gap."""
import numpy as np
from paper_plot_style import *

TASKS = ["lg_promoter_detection", "lg_npp", "lg_fold_class",
         "tape_fluorescence", "gb_demo_coding_vs_intergenomic_seqs"]
SHORT = ["promoter\n(acc)", "npp\n(acc)", "fold\n(acc)", "fluorescence\n(Spearman)", "coding\n(AUROC)"]
KEY = {"choice": "accuracy", "noul": "auroc", "score": "spearman"}
FAMOF = {"lg_promoter_detection": "local_snapshots", "lg_npp": "local_snapshots",
         "lg_fold_class": "local_snapshots", "tape_fluorescence": "TAPE",
         "gb_demo_coding_vs_intergenomic_seqs": "GenomicBenchmarks"}

def val(res, t):
    r = (res or {}).get(t)
    if not r:
        return None
    k = KEY.get(r.get("primitive", ""))
    return r.get(k) if k else None

def load(p):
    import json
    from pathlib import Path
    p = Path(p)
    return json.loads(p.read_text())["results"] if p.exists() else None

regs = {}
regs["single"] = {t: val(load(TR / f"st2_{t}" / "eval_results.json"), t) for t in TASKS}
regs["family"] = {t: val(load(TR / f"lb2_{FAMOF[t]}" / "eval_results.json"), t) for t in TASKS}
regs["all-241"] = {t: val(load(TR / "j241_2" / "eval_results.json"), t) for t in TASKS}
# joint-37 slice: 3-seed cpt mean per task
j37 = {}
for t in TASKS:
    vs = []
    for s in ("20261001", "20261002", "20261003"):
        r = load(TR / f"s2c_{s}" / "eval_results.json")
        v = val(r, t)
        if v is not None:
            vs.append(v)
    j37[t] = np.mean(vs) if vs else None
regs["joint-37"] = j37

fig, ax = plt.subplots(figsize=(6.9, 2.6))
x = np.arange(len(TASKS))
w = 0.2
cols = [C_ACC, C_SHARED, "#ff9896", C_GEN]
for i, (name, d) in enumerate(regs.items()):
    vals = [d[t] for t in TASKS]
    ax.bar(x + (i - 1.5) * w, [v if v is not None else 0 for v in vals], w,
           color=cols[i], label={"single":"single-task","family":"family-joint","all-241":"all-241 joint","joint-37":"joint-37 slice"}[name])
    for xi, v in zip(x + (i - 1.5) * w, vals):
        if v is None:
            ax.text(xi, 0.01, "n/a", ha="center", fontsize=5.6, rotation=90, va="bottom")
ax.axhline(0, color="black", lw=0.8)
ax.plot([-0.5, 1.5], [0.5, 0.5], ls=":", color="black", lw=0.8)
ax.text(1.52, 0.505, "chance (acc/AUROC)", fontsize=6.2, va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(SHORT, fontsize=6.6)
ax.set_ylabel("task metric (per-task native caliber)")
ax.set_ylim(-0.1, 1.0)
ax.legend(loc="upper right", frameon=False, fontsize=6.2, ncol=2)
save_fig(fig, "fig5_gradient")
print({k: {t: v for t, v in d.items()} for k, d in regs.items()})
