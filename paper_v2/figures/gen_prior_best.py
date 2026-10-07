#!/usr/bin/env python3
"""Prior-best comparison table (A experiment): our family-specialized CPT baselines
vs published bests, with explicit metric-caliber notes (never mix metrics silently).
Reads artifacts/benchmark_v2_eval/prior_best.json + lb2_* eval results."""
import json
from statistics import mean
from paper_plot_style import *

pb = load(AE / "prior_best.json")

def lb2_family_metrics(fam):
    """per-task metrics of the #2 specialized run, split by primitive"""
    p = TR / f"lb2_{fam}" / "eval_results.json"
    if not p.exists():
        return {}
    res = load(p)["results"]
    out = {}
    for t, r in res.items():
        if r.get("n", 0) == 0:
            continue
        key = {"choice": "accuracy", "noul": "auroc", "score": "spearman"}[r["primitive"]]
        if key in r:
            out.setdefault(r["primitive"], []).append(r[key])
    return {k: mean(v) for k, v in out.items()}

# family -> (prior-best display, their metric, comparability note)
ROWS = [
    ("GUE", "DNABERT-2 67.77 / NT-2.5B 66.93", "MCC avg",
     "ours is accuracy: not directly comparable, caliber ref only"),
    ("GenomicBenchmarks", "HyenaDNA best on 7/8 (e.g. worm 96.6, regulatory 93.8)", "top-1 acc",
     "comparable (acc)"),
    ("ProteinGym", "ProteinNPT 0.547 (supervised, fold-contiguous split)", "Spearman",
     "comparable: same official split family"),
    ("TAPE", "fluor 0.68 / stab 0.73 (TAPE-era pretrain; ProtBERT-ft 0.678/0.734)", "Spearman",
     "comparable"),
    ("DeepSTARR", "0.68 / 0.74 (authors' CNN)", "PCC",
     "PCC vs our Spearman; 512-token window truncates ~2kb inputs"),
    ("dnagpt_pools", "LLaMA-Gene category-level 0.83 (DNA class.)", "accuracy",
     "category-level only; per-dataset numbers unverifiable locally"),
    ("local_snapshots", "self-prior: promoter 0.911, fold 0.600 (working ms Table 2)", "accuracy",
     "same-lineage prior; internal record, not peer-reviewed"),
]

tex = [r"% Table 8: prior-best vs our family-specialized baselines (gen_prior_best.py)",
       r"\begin{table}[t]", r"\centering",
       r"\caption{Published bests vs our family-specialized CPT baselines (\arch{2}, "
       r"single seed). Metric calibers differ per family and are \emph{never} averaged "
       r"across rows; the note column states comparability. Sources transcribed from the "
       r"cited papers' tables (see \texttt{prior\_best.json}); unverified numbers are excluded.}",
       r"\label{tab:priorbest}", r"\footnotesize",
       r"\begin{tabular}{p{0.14\textwidth}rp{0.13\textwidth}p{0.30\textwidth}}", r"\toprule",
       r"Family & Ours & Literature best (their metric) & Note \\", r"\midrule"]
for fam, lit, litm, note in ROWS:
    m = lb2_family_metrics(fam)
    if m:
        ours = ", ".join(f"{v:.3f}" for k, v in sorted(m.items()))
        metrics = "/".join({"choice": "acc", "noul": "AUROC", "score": "Sp"}[k] for k in sorted(m))
        ours = f"{ours} ({metrics})"
    else:
        ours = "--"
    tex.append(f"{fam.replace('_','-')} & {ours} & {lit.replace('--','--').replace('%','\\%')} & {note} \\\\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(FIG_DIR / "TABLE_8_prior_best.tex").write_text("\n".join(tex))
print("wrote TABLE_8_prior_best.tex")
