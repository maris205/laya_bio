#!/usr/bin/env python3
"""Leak-proof split isolation for benchmark v2.

Rationale (matches the catalog's admission caution):
  * Split isolation uses EXACT sequence identity, NOT reverse complement.
    For strand-sensitive tasks (splice donor/acceptor, promoter direction) an
    RC merge would silently flip labels, so RC grouping is deferred to a
    label-aware follow-up. Exact duplicates across splits are unambiguous and
    safe to reconcile.
  * If a sequence appears in more than one split, the whole group is kept in the
    highest-priority split (test > dev > train); copies in lower splits dropped.
  * If a sequence maps to conflicting answers within a task (annotation
    conflict / near-duplicate), the entire group is QUARANTINED (removed from
    every split) rather than resolved by majority vote.

Rewrites <task>/{train,dev,test}.jsonl in place and writes quarantine counts
into isolation_report.json.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict, Counter

OUT = Path("/root/autodl-tmp/jev_gene/data/06_benchmark_v2_unified")
PRIORITY = {"test": 3, "dev": 2, "train": 1}


def seq_key(r):
    return tuple(s["sequence"] for s in r["sequences"])


def answer_of(r):
    p = r["primitive"]
    if p == "multi_noul":
        return tuple(sorted(r["answer"]))
    if p == "score":
        # treat large relative differences as conflicts; exact level match is coarse
        return round(r["gold_value"], 3)
    return r["answer"]


def isolate_task(tdir):
    recs = []
    for split in ("train", "dev", "test"):
        f = tdir / f"{split}.jsonl"
        if f.exists():
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); r["_orig_split"] = split; recs.append(r)
    if not recs:
        return None
    prim = recs[0]["primitive"]
    before = Counter(r["_orig_split"] for r in recs)
    # group by exact sequence
    groups = defaultdict(list)
    for r in recs:
        groups[seq_key(r)].append(r)
    quarantine = set()
    kept_split_for_group = {}
    dup_moves = Counter()
    for key, rows in groups.items():
        answers = {answer_of(r) for r in rows}
        if len(answers) > 1:
            quarantine.add(key)
            continue
        # choose highest-priority split present among rows
        best = max(rows, key=lambda r: PRIORITY[r["_orig_split"]])
        target = best["_orig_split"]
        kept_split_for_group[key] = target
        # count moves: rows whose orig != target and same split existed elsewhere
        splits_here = Counter(r["_orig_split"] for r in rows)
        if len(splits_here) > 1:
            dup_moves["dropped_from_lower_splits"] += sum(v for s, v in splits_here.items() if PRIORITY[s] < PRIORITY[target])
    out = {"train": [], "dev": [], "test": []}
    for r in recs:
        key = seq_key(r)
        if key in quarantine:
            continue
        r.pop("_orig_split", None)
        out[kept_split_for_group[key]].append(r)
    for split in ("train", "dev", "test"):
        with open(tdir / f"{split}.jsonl", "w") as fh:
            for r in out[split]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "primitive": prim,
        "before": {"train": before["train"], "dev": before["dev"], "test": before["test"]},
        "after": {s: len(out[s]) for s in ("train", "dev", "test")},
        "n_groups": len(groups),
        "quarantined_groups": len(quarantine),
        "dropped_from_lower_splits": dup_moves["dropped_from_lower_splits"],
    }


def main():
    report = {}
    tot_q = 0
    for tdir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        res = isolate_task(tdir)
        if res:
            report[tdir.name] = res
            tot_q += res["quarantined_groups"]
    (OUT / "isolation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # summary
    worst = sorted(report.items(), key=lambda kv: -kv[1]["quarantined_groups"])[:10]
    print(f"isolated {len(report)} tasks; total quarantined groups: {tot_q}")
    print("top quarantine tasks:")
    for k, v in worst:
        if v["quarantined_groups"] or v["dropped_from_lower_splits"]:
            print(f"  {k:32s} quar={v['quarantined_groups']:5d} dupdropped={v['dropped_from_lower_splits']:5d} "
                  f"train {v['before']['train']}->{v['after']['train']} "
                  f"test {v['before']['test']}->{v['after']['test']}")


if __name__ == "__main__":
    main()
