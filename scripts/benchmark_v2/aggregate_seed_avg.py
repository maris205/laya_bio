#!/usr/bin/env python3
"""Aggregate the seed-averaged three-way comparison (#1 shared scorer, #2 task
heads, #3 frozen generative) into mean +/- std per primitive.

#1: artifacts/benchmark_v2_eval/s1_<seed>_<family>/eval_results.json (per-family
    best-dev checkpoints, one eval dir per family per seed)
#2: artifacts/benchmark_v2_train/s2_<seed>/eval_results.json (self-eval, per-family)
#3: artifacts/benchmark_v2_eval/gen3/eval_results.json (frozen, deterministic)
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from collections import defaultdict

JEV = "/root/autodl-tmp/jev_gene"
EV = f"{JEV}/artifacts/benchmark_v2_eval"
TR = f"{JEV}/artifacts/benchmark_v2_train"
SEEDS = ["20261001", "20261002", "20261003"]


def bucket(v):
    if not v: return (None, None)
    p = v.get("primitive")
    if p == "score": return ("score", v.get("spearman"))
    if p == "noul": return ("noul", v.get("auroc") if v.get("auroc") is not None else v.get("accuracy"))
    if p == "choice": return ("choice", v.get("accuracy"))
    return (None, None)


def load1(seed):
    M = {}
    for d in glob.glob(f"{EV}/s1_{seed}_*"):
        f = f"{d}/eval_results.json"
        if os.path.exists(f):
            M.update(json.load(open(f))["results"])
    return M


def load2(seed):
    f = f"{TR}/s2_{seed}/eval_results.json"
    return json.load(open(f))["results"] if os.path.exists(f) else {}


def main():
    m1 = {s: load1(s) for s in SEEDS}
    m2 = {s: load2(s) for s in SEEDS}
    m3 = json.load(open(f"{EV}/gen3/eval_results.json"))["results"]
    seeds1 = [s for s in SEEDS if m1[s]]
    seeds2 = [s for s in SEEDS if m2[s]]
    print(f"#1 seeds loaded: {seeds1} | #2 seeds loaded: {seeds2}")
    # common tasks across all available seeds of #1 and #2
    def common(mm, seeds):
        sets = [set(t for t in mm[s] if bucket(mm[s][t])[1] is not None) for s in seeds]
        return set.intersection(*sets) if sets else set()
    t1 = common(m1, seeds1); t2 = common(m2, seeds2)
    tasks = sorted(t1 & t2)
    print(f"common tasks with valid metric in all seeds: #1={len(t1)} #2={len(t2)} both={len(tasks)}")

    # per-primitive: for each task, average metric across seeds; then mean/std across tasks
    prim_task_mean = defaultdict(lambda: {"1": [], "2": [], "3": []})
    for t in tasks:
        pr = bucket(m1[seeds1[0]][t])[0]
        v1 = [bucket(m1[s][t])[1] for s in seeds1 if bucket(m1[s][t])[1] is not None]
        v2 = [bucket(m2[s][t])[1] for s in seeds2 if bucket(m2[s][t])[1] is not None]
        if not v1 or not v2: continue
        prim_task_mean[pr]["1"].append(np.mean(v1))
        prim_task_mean[pr]["2"].append(np.mean(v2))
        if pr in ("choice", "noul") and t in m3 and not m3[t].get("skip") and m3[t].get("accuracy") is not None:
            prim_task_mean[pr]["3"].append(m3[t]["accuracy"])

    print("\n=== seed-averaged three-way (per-task seed-mean, then across tasks) ===")
    print(f"{'primitive':9s} {'metric':6s} {'n':>3s} {'#1 shared':>16s} {'#2 heads':>16s} {'#3 frozen':>9s}")
    summary = {}
    for pr in ("choice", "noul", "score"):
        d = prim_task_mean[pr]
        if not d["1"]: continue
        mtr = {"choice": "acc", "noul": "auroc", "score": "spear"}[pr]
        a1 = np.array(d["1"]); a2 = np.array(d["2"])
        # per-task seed-std (averaged) to show run variance
        s1 = f"{a1.mean():.3f}±{a1.std()/max(1,len(a1))**0.5:.3f}"
        s2 = f"{a2.mean():.3f}±{a2.std()/max(1,len(a2))**0.5:.3f}"
        g3 = f"{np.mean(d['3']):.3f}" if d["3"] else "n/a"
        print(f"{pr:9s} {mtr:6s} {len(a1):3d} {s1:>16s} {s2:>16s} {g3:>9s}")
        summary[pr] = {"metric": mtr, "n": len(a1),
                       "shared_mean": float(a1.mean()), "shared_sem": float(a1.std()/len(a1)**0.5),
                       "taskhead_mean": float(a2.mean()), "taskhead_sem": float(a2.std()/len(a2)**0.5),
                       "gen3_mean": float(np.mean(d["3"])) if d["3"] else None}

    # per-task seed variance (how much does one run bounce?)
    print("\n=== per-task run variance (seed std, averaged over tasks) ===")
    for pr in ("choice", "noul", "score"):
        stds1, stds2 = [], []
        for t in tasks:
            if bucket(m1[seeds1[0]][t])[0] != pr: continue
            v1 = [bucket(m1[s][t])[1] for s in seeds1 if bucket(m1[s][t])[1] is not None]
            v2 = [bucket(m2[s][t])[1] for s in seeds2 if bucket(m2[s][t])[1] is not None]
            if len(v1) > 1: stds1.append(np.std(v1))
            if len(v2) > 1: stds2.append(np.std(v2))
        if stds1: print(f"  {pr:8s} #1 seed-std={np.mean(stds1):.3f}  #2 seed-std={np.mean(stds2) if stds2 else float('nan'):.3f}")
        summary.setdefault(pr, {})["run_std"] = {"shared": float(np.mean(stds1)) if stds1 else None,
                                                  "taskhead": float(np.mean(stds2)) if stds2 else None}

    # paired #1 vs #2 per task (on seed-means)
    w1 = w2 = tie = 0
    for t in tasks:
        v1 = np.mean([bucket(m1[s][t])[1] for s in seeds1 if bucket(m1[s][t])[1] is not None])
        v2 = np.mean([bucket(m2[s][t])[1] for s in seeds2 if bucket(m2[s][t])[1] is not None])
        d = v1 - v2
        if d > 0.05: w1 += 1
        elif d < -0.05: w2 += 1
        else: tie += 1
    print(f"\n#1 vs #2 (seed-mean, per task): #1 wins {w1} / tie {tie} / #2 wins {w2}")
    summary["per_task_wins_seedmean"] = {"shared": w1, "tie": tie, "taskhead": w2}
    json.dump(summary, open(f"{EV}/seed_avg_three_way.json", "w"), indent=2)
    print(f"saved {EV}/seed_avg_three_way.json")


if __name__ == "__main__":
    main()
