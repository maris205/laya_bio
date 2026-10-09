#!/usr/bin/env python3
"""Paper-2 scale table: seed-averaged task-macro metrics per backbone x arm.
Backbones: 423M encoder no_cpt (Paper-1 main), 423M encoder cpt (Paper-1 appendix),
Qwen3-0.6B LoRA, Qwen3-8B QLoRA. Arms: shared scorer / per-task heads.
Reads: seed_avg_three_way.json, seed_avg_three_way_cpt.json,
q06_*_s20261001 (1 seed), q8_{arch}_s* (3 seeds)."""
import json, glob
import numpy as np
from paper_plot_style import *

KEY = {"choice": "accuracy", "noul": "auroc", "score": "spearman"}

def from_three_way(path):
    d = load(path)
    return {p: {"shared": d[p]["shared_mean"], "heads": d[p]["taskhead_mean"],
                "n": d[p]["n"], "sem_s": d[p]["shared_sem"], "sem_h": d[p]["taskhead_sem"]}
            for p in ("choice", "noul", "score") if p in d}

rows = []
rows.append(("423M enc (no_cpt)", from_three_way(AE / "seed_avg_three_way.json")))
rows.append(("423M enc (cpt)", from_three_way(AE / "seed_avg_three_way_cpt.json")))

for label, dirs in [("Qwen3-0.6B LoRA", [str(TR / "q06_shared_s20261001" / "eval_results.json"),
                                         str(TR / "q06_heads_s20261001" / "eval_results.json")]),
                    ("Qwen3-8B QLoRA", [str(TR / f"q8_shared_s{s}" / "eval_results.json") for s in
                                        ("20261001", "20261002", "20261003")] +
                                       [str(TR / f"q8_heads_s{s}" / "eval_results.json") for s in
                                        ("20261001", "20261002", "20261003")])]:
    M = {"shared": {}, "heads": {}}
    prim = {}
    for f in dirs:
        p = Path(f)
        if not p.exists():
            continue
        arm = "shared" if "shared" in p.parent.name else "heads"
        res = load(p)["results"]
        for t, r in res.items():
            k = KEY.get(r.get("primitive", ""))
            if k and r.get(k) is not None and r.get("n", 0) > 0:
                M[arm].setdefault(t, []).append(r[k])
                prim[t] = r["primitive"]
    entry = {}
    for p in ("choice", "noul", "score"):
        ts = [t for t in M["heads"] if prim.get(t) == p and t in M["shared"]]
        if not ts:
            continue
        sh = np.mean([np.mean(M["shared"][t]) for t in ts])
        hd = np.mean([np.mean(M["heads"][t]) for t in ts])
        entry[p] = {"shared": float(sh), "heads": float(hd), "n": len(ts),
                    "seeds": max(len(M["heads"][t]) for t in ts)}
    rows.append((label, entry))

tex = [r"% Table 10: scale x architecture table (gen_scale_table.py)",
       r"\begin{table}[t]", r"\centering",
       r"\caption{Seed-averaged task-macro metrics by backbone and decision arm on the "
       r"37-task matched slice. Qwen rows use LoRA/QLoRA fine-tuning of the causal LM; "
       r"encoder rows are from Paper-1 runs. $n$ = tasks per primitive; seeds in parentheses.}",
       r"\label{tab:scale}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
       r"\begin{tabular}{llcccccc}", r"\toprule",
       r"Backbone & Arm & choice acc & noul AUROC & score Spearman & $n$(c/n/s) \\", r"\midrule"]
for label, entry in rows:
    for arm in ("shared", "heads"):
        cells = []
        ns = []
        for p in ("choice", "noul", "score"):
            e = entry.get(p)
            if e and e.get(arm) is not None:
                cells.append(f"{e[arm]:.3f}")
                ns.append(str(e["n"]))
            else:
                cells.append("--")
                ns.append("-")
        tex.append(f"{label if arm == 'shared' else ''} & \\arch{{{'1' if arm == 'shared' else '2'}}} & "
                   + " & ".join(cells) + f" & {'/'.join(ns)} \\\\")
    tex.append(r"\addlinespace")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(FIG_DIR / "TABLE_10_scale.tex").write_text("\n".join(tex))
for label, entry in rows:
    print(label, {p: {a: round(entry[p][a], 3) for a in ("shared", "heads")} for p in entry})
print("wrote TABLE_10_scale.tex")
