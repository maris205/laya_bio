#!/usr/bin/env python3
"""Admission / leak checks for benchmark v2 unified tasks.

Per task:
  - split sizes
  - train/test split overlap by sequence exact-match and by group id (leak check)
  - label conflicts: same sequence -> different answer within a split
  - answer in candidates / label validity for each primitive
  - sequence length distribution (for tokenizer admission)
Global:
  - cross-task exact-duplicate sequences (same sequence reused across tasks)
Writes admission_report.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

OUT = Path("/root/autodl-tmp/jev_gene/data/06_benchmark_v2_unified")


def load_task(task_dir):
    recs = {}
    for split in ("train", "dev", "test"):
        f = task_dir / f"{split}.jsonl"
        if not f.exists():
            recs[split] = []
            continue
        recs[split] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return recs


def seq_key(r):
    return tuple(s["sequence"] for s in r["sequences"])


def answer_of(r):
    p = r["primitive"]
    if p == "multi_noul":
        return tuple(sorted(r["answer"]))
    if p == "score":
        return r["gold_level"]
    return r["answer"]


def check_task(task_dir):
    recs = load_task(task_dir)
    prim = None
    info = {"primitive": None, "modality": None, "n_classes": None}
    seq_len = {"train": [], "dev": [], "test": []}
    group_by_split = {}
    seq_by_split = {}
    label_conflict = 0
    bad_answer = 0
    n = 0
    for split, rows in recs.items():
        prim = rows[0]["primitive"] if rows and prim is None else prim
        groups, seqs = Counter(), defaultdict(set)
        for r in rows:
            n += 1
            if prim is None:
                prim = r["primitive"]
            info["modality"] = r["modality"]
            key = seq_key(r)
            L = max(len(s["sequence"]) for s in r["sequences"])
            seq_len[split].append(L)
            groups[r.get("group", "")] += 1
            seqs[key].add(answer_of(r))
            # validity
            if r["primitive"] == "noul":
                if r["answer"] not in ("yes", "no"):
                    bad_answer += 1
            elif r["primitive"] == "choice":
                if r["answer"] not in r["candidates"]:
                    bad_answer += 1
            elif r["primitive"] == "score":
                if r["gold_level"] not in r["score_levels"]:
                    bad_answer += 1
        for key, answers in seqs.items():
            if len(answers) > 1:
                label_conflict += 1
        group_by_split[split] = set(groups)
        seq_by_split[split] = set(seqs)
    info["primitive"] = prim
    info["n_records"] = n
    # leak: train∩test, train∩dev, dev∩test by group and by seq
    leak = {}
    for a, b in [("train", "test"), ("train", "dev"), ("dev", "test")]:
        leak[f"group_{a}&{b}"] = len(group_by_split[a] & group_by_split[b])
        leak[f"seq_{a}&{b}"] = len(seq_by_split[a] & seq_by_split[b])
    info["split_sizes"] = {s: len(v) for s, v in recs.items()}
    info["split_overlap"] = leak
    info["label_conflicts"] = label_conflict
    info["invalid_answers"] = bad_answer
    for s in seq_len:
        if seq_len[s]:
            a = np.array(seq_len[s])
            info[f"len_{s}"] = {"min": int(a.min()), "median": int(np.median(a)),
                                "p95": int(np.percentile(a, 95)), "max": int(a.max())}
        else:
            info[f"len_{s}"] = None
    return info, recs


def main():
    if not OUT.exists():
        print("no unified dir yet", file=sys.stderr); return 1
    tasks = sorted(p for p in OUT.iterdir() if p.is_dir())
    report = {}
    global_seq = defaultdict(set)  # seq -> set of task_ids
    for t in tasks:
        info, recs = check_task(t)
        report[t.name] = info
        for split, rows in recs.items():
            for r in rows:
                global_seq[seq_key(r)].add(t.name)
    # cross-task duplicates
    cross = {k: sorted(v) for k, v in global_seq.items() if len(v) > 1}
    cross_summary = Counter(tuple(v) for v in cross.values())
    n_leak = sum(1 for k, v in report.items()
                 if v["split_overlap"].get("group_train&test", 0) > 0
                 or v["split_overlap"].get("seq_train&test", 0) > 0)
    summary = {
        "n_tasks": len(report),
        "tasks_with_train_test_leak": n_leak,
        "n_cross_task_duplicate_sequences": len(cross),
        "top_cross_task_pairs": [
            {"tasks": list(pair), "n_shared": cnt}
            for pair, cnt in cross_summary.most_common(15)
        ],
        "per_task_flags": {
            k: [f for f in ("train_test_leak" if (v["split_overlap"].get("group_train&test",0) or v["split_overlap"].get("seq_train&test",0)) else "",
                            "label_conflict" if v["label_conflicts"] else "",
                            "invalid_answer" if v["invalid_answers"] else "")
                 if f]
            for k, v in report.items()
        },
    }
    out = {"summary": summary, "tasks": report}
    (OUT / "admission_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("n_tasks:", len(report), "| leak tasks:", n_leak, "| cross-task dup seqs:", len(cross))
    for k, flags in summary["per_task_flags"].items():
        if flags:
            print(f"  FLAG {k}: {flags}  overlap={report[k]['split_overlap']} conflict={report[k]['label_conflicts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
