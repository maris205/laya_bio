#!/usr/bin/env python3
"""Family-specialized leaderboard table (B experiment).
Reads lb2_<F>/eval_results.json (train dir) and lb1_<F>/eval_results.json (eval dir)
produced by scripts/benchmark_v2/leaderboard_driver.sh; emits
TABLE_7_leaderboard.tex with per-family task-macro metrics for both arms,
plus a per-task CSV for the appendix. Families with missing results are skipped
and reported, so the script can run incrementally while the driver is still going."""
import json, csv
from pathlib import Path
from statistics import mean, pstdev
from paper_plot_style import *
from paper_plot_style import family_of

FAMS = ["dnagpt_pools", "TAPE", "DeepSTARR", "local_snapshots",
        "GenomicBenchmarks", "GUE", "ProteinGym"]
PRIM_METRIC = {"choice": "accuracy", "noul": "auroc", "score": "spearman"}

SINGLE = {}
def single_results():
    if not SINGLE:
        for arm, p in (("1", AE / "j241_1" / "eval_results.json"),
                       ("2", TR / "j241_2" / "eval_results.json")):
            SINGLE[arm] = json.loads(p.read_text())["results"] if p.exists() else None
    return SINGLE

def arm_results(arm, fam):
    if arm in ("s1", "s2"):
        allr = single_results().get(arm[1])
        if allr is None:
            return None
        return {t: r for t, r in allr.items() if family_of(t) == fam} or None
    p = (AE / f"lb1_{fam}" if arm == "1" else TR / f"lb2_{fam}") / "eval_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["results"]

rows, pending, per_task = [], [], []
for fam in FAMS:
    r1, r2 = arm_results("1", fam), arm_results("2", fam)
    q1, q2 = arm_results("s1", fam), arm_results("s2", fam)
    if r1 is None and r2 is None and q1 is None and q2 is None:
        pending.append(fam); continue
    for prim in ("choice", "noul", "score"):
        out = {"family": fam, "prim": prim, "n": 0}
        for arm, res in (("1", r1), ("2", r2), ("q1", q1), ("q2", q2)):
            if res is None:
                out[arm] = None; continue
            vals = [r[PRIM_METRIC[prim]] for r in res.values()
                    if r.get("primitive") == prim and r.get("n", 0) > 0
                    and r.get(PRIM_METRIC[prim]) is not None]
            out[arm] = mean(vals) if vals else None
            out["n"] = max(out["n"], len(vals))
        if out["n"] == 0:
            continue
        for arm, res in (("1", r1), ("2", r2), ("q1", q1), ("q2", q2)):
            for tid, r in (res or {}).items():
                if r.get("primitive") == prim and r.get(PRIM_METRIC[prim]) is not None:
                    per_task.append([fam, arm, tid, prim, r.get("n"),
                                     round(r[PRIM_METRIC[prim]], 4)])
        rows.append(out)

with open(FIG_DIR / "leaderboard_per_task.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["family", "arm", "task", "primitive", "n_eval", "metric"])
    w.writerows(per_task)

tex = [r"% Table 7: family-specialized leaderboard (gen_leaderboard.py)",
       r"\begin{table}[t]", r"\centering",
       r"\caption{Three training regimes over the full supported benchmark (single training "
       r"seed; stabilized protocol; global best-dev early stop for the single-model columns, "
       r"per-family early stop for the specialized columns): one unified model over all "
       r"${\sim}$243 gated tasks (single), vs one model per family (spec.). Task-macro of "
       r"accuracy (choice), AUROC (noul), Spearman (score). The single-vs-spec.\ gap is the "
       r"measured cost of unification; compare also the joint-37 slice in \cref{tab:main}.}",
       r"\label{tab:leaderboard}", r"\small",
       r"\begin{tabular}{llrrrrr}", r"\toprule",
       r"Family & Primitive & Tasks & \arch{1} single & \arch{2} single & \arch{1} spec. & \arch{2} spec. \\", r"\midrule"]
MNAME = {"choice": "acc", "noul": "AUROC", "score": "Spearman"}
for o in rows:
    def fmt(v): return f"{v:.3f}" if v is not None else "--"
    tex.append(f"{o['family'].replace('_','-')} & {MNAME[o['prim']]} & {o['n']} & "
               f"{fmt(o.get('q1'))} & {fmt(o.get('q2'))} & "
               f"{fmt(o.get('1'))} & {fmt(o.get('2'))} \\\\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(FIG_DIR / "TABLE_7_leaderboard.tex").write_text("\n".join(tex))
print("families done:", [o["family"] for o in rows], "| pending:", pending)
