#!/usr/bin/env python3
"""Same-recipe comparison: Qwen3-0.6B LoRA family-specialized (q06spec_*) vs the
Laya-423M family-specialized leaderboard (lb2_*), per task, with deltas.
Both arms: 64 rows/update x 2496 updates, per-family single-family training,
CPT-equivalent priors (Laya bio-CPT init vs Qwen general pretrain), 1 seed."""
import json
from paper_plot_style import *

KEY = {"choice": "accuracy", "noul": "auroc", "score": "spearman"}
FAMS = ["local_snapshots", "TAPE", "GenomicBenchmarks"]

def res(arm, fam):
    p = (TR / f"q06spec_{fam}" / "eval_results.json") if arm == "q" else \
        (TR / f"lb2_{fam}" / "eval_results.json")
    return json.loads(p.read_text())["results"] if p.exists() else None

rows = []
for fam in FAMS:
    rq, rl = res("q", fam), res("l", fam)
    if not rq or not rl:
        continue
    for t in sorted(rl):
        v = rl[t]
        k = KEY.get(v.get("primitive", ""))
        if not k or v.get(k) is None:
            continue
        w = rq.get(t, {})
        q = w.get(k)
        if q is None:
            continue
        rows.append((fam, t.replace("_", "\\_"), v["primitive"], q, v[k], q - v[k]))

tex = [r"% Table 11: same-recipe Laya vs Qwen (gen_laya_vs_qwen.py)",
       r"\begin{table}[t]", r"\centering",
       r"\caption{Same specialized recipe (64 rows/update, 2{,}496 updates, single family, "
       r"single seed): Qwen3-0.6B + LoRA vs the Laya-423M biology-CPT encoder. Positive "
       r"$\Delta$ favors Qwen. Metric per primitive: accuracy (choice), AUROC (noul), "
       r"Spearman (score).}",
       r"\label{tab:layavsqwen}", r"\footnotesize",
       r"\begin{tabular}{lllrrr}", r"\toprule",
       r"Family & Task & Primitive & Qwen-0.6B & Laya-423M & $\Delta$ \\", r"\midrule"]
for fam, t, prim, q, l, d in rows:
    tex.append(f"{fam.replace('_','-')} & {t} & {prim} & {q:.3f} & {l:.3f} & {d:+.3f} \\\\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(FIG_DIR / "TABLE_11_laya_vs_qwen.tex").write_text("\n".join(tex))
wins = sum(1 for r in rows if r[5] > 0)
print(f"rows={len(rows)} qwen-wins={wins} | wrote TABLE_11_laya_vs_qwen.tex")
