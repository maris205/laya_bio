#!/usr/bin/env python3
"""Benchmark v2 slice 3: ProteinGym (217 substitution assays, Score) using the
official fold_contiguous_5 split, and RNAcompete (14 RBP binding-preference
Score tasks) for RNA modality coverage.

ProteinGym: GleghornLab/ProteinGym_DMS by_dms_id parquets carry the official
per-mutant fold columns. We use fold_contiguous_5 (MSA-contiguous, homology-
aware): fold0=test, fold1=dev, folds2-4=train. Rows with fold==-100 (multiple
mutants, outside the single-mutant benchmark) are excluded. Anchors per assay
from train only.

RNAcompete: probe_metadata.tsv (probe_id->RNA seq) x probe_zscore.tsv
(probe x 14 HybID RBP z-scores). One Score task per RBP. Probes subsampled to
a fixed random subset (documented) and split by probe-hash bucket.
"""
from __future__ import annotations
import json, glob, hashlib, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

JEV = Path("/root/autodl-tmp/jev_gene")
RAW = JEV / "data/05_benchmark_v2_raw"
OUT = JEV / "data/06_benchmark_v2_unified"
RECEIPT = json.loads((RAW / "download_receipt.json").read_text())
SCORE_N_ANCHORS = 5
PG_FOLD_COL = "fold_contiguous_5"
RNA_PROBE_SUBSAMPLE = 25000  # fixed random probe subset per RBP task
RNA_SEED = 20261001

REV = {}
for d in RECEIPT.get("datasets", []):
    REV[d["subdir"]] = {"revision": d["revision"], "license": d.get("license", "?"), "source": d["repo_id"]}
for d in RECEIPT.get("github_clones", []):
    REV[d["name"]] = {"revision": d.get("commit"), "license": "?", "source": d.get("task")}


def prov(subdir, origin=None):
    r = REV.get(subdir, {"revision": None, "license": "?"})
    return {"revision": r["revision"], "license": r["license"], "source": r.get("source", subdir), "origin": origin}


def write_records(task_id, records):
    d = OUT / task_id
    d.mkdir(parents=True, exist_ok=True)
    counts = Counter(r["split"] for r in records)
    for split in ("train", "dev", "test"):
        with open(d / f"{split}.jsonl", "w") as fh:
            for r in records:
                if r["split"] == split:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return counts


def base(task_id, primitive, modality, sequences, question, split, provenance, group=""):
    return {"task_id": task_id, "primitive": primitive, "modality": modality,
            "sequences": sequences, "question": question, "split": split,
            "group": group, "provenance": provenance}


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def score_record(task_id, modality, seq, question, split, provenance, value, anchors):
    lvl = str(sum(1 for a in anchors if value > a))
    r = base(task_id, "score", modality,
             [{"role": "query", "modality": modality[0], "sequence": seq}],
             question, split, provenance, group=hashlib.sha1(seq.encode()).hexdigest()[:12])
    r["score_levels"] = [str(i) for i in range(SCORE_N_ANCHORS)]
    r["anchors"] = [float(a) for a in anchors]
    r["gold_value"] = float(value)
    r["gold_level"] = lvl
    return r


# ---------- ProteinGym ----------
def conv_proteingym(manifest):
    files = [f for f in glob.glob(str(RAW / "proteingym_dms_folds/by_dms_id/*.parquet")) if "__" not in f]
    n = 0
    for f in files:
        d = pd.read_parquet(f, columns=["DMS_id", "is_indel", PG_FOLD_COL, "DMS_score", "mutated_seq", "mutant"])
        sub = d[(d["is_indel"] == False) & (d[PG_FOLD_COL] >= 0) & (d[PG_FOLD_COL] <= 4)]
        if len(sub) == 0:
            continue
        dms_id = str(sub["DMS_id"].iloc[0])
        task_id = f"pg_{sanitize(dms_id)}"
        # anchors from train folds (2,3,4)
        train_vals = sub.loc[sub[PG_FOLD_COL].isin([2, 3, 4]), "DMS_score"].astype(float).values
        if len(train_vals) < 10:
            continue
        anchors = np.quantile(train_vals, np.linspace(0, 1, SCORE_N_ANCHORS + 1)[1:-1]).tolist()
        q = (f"Estimate the mutational fitness (DMS score) of this protein variant "
             f"(assay {dms_id}); higher = more fit.")
        recs = []
        for fold, seq, val in zip(sub[PG_FOLD_COL], sub["mutated_seq"].astype(str), sub["DMS_score"].astype(float)):
            split = "test" if fold == 0 else ("dev" if fold == 1 else "train")
            recs.append(score_record(task_id, ["protein"], seq, q, split,
                                     prov("proteingym_dms_folds", origin=f"ProteinGym/{dms_id}"), val, anchors))
        counts = write_records(task_id, recs)
        manifest[task_id] = {"primitive": "score", "modality": ["protein"], "n_levels": SCORE_N_ANCHORS,
                             "anchors": [float(a) for a in anchors], "dms_id": dms_id,
                             **{s: counts[s] for s in ("train", "dev", "test")},
                             "provenance": prov("proteingym_dms_folds", origin=dms_id),
                             "status": "ready", "split_protocol": "fold_contiguous_5: f0=test f1=dev f2-4=train"}
        n += 1
    return n


# ---------- RNAcompete ----------
def conv_rnacompete(manifest):
    rc = RAW / "github/RNAcompete"
    meta = pd.read_csv(rc / "rnacompete/data/probe_metadata.tsv", sep="\t")
    # probe seq map
    probe_seq = dict(zip(meta["probe_id"].astype(str), meta["seq"].astype(str)))
    zpath = rc / "test/root/HybID00025_00103_subset/probe_zscore.tsv"
    z = pd.read_csv(zpath, sep="\t", index_col=0)
    rbps = list(z.columns)
    probe_ids = list(z.index.astype(str))
    # fixed random subsample of probes present in both
    rng = np.random.default_rng(RNA_SEED)
    valid = [p for p in probe_ids if p in probe_seq]
    if len(valid) > RNA_PROBE_SUBSAMPLE:
        valid = list(rng.choice(valid, size=RNA_PROBE_SUBSAMPLE, replace=False))
    n = 0
    for rbp in rbps:
        task_id = f"rnac_{sanitize(rbp)}"
        rows = []
        for p in valid:
            v = z.at[p, rbp] if p in z.index else None
            if v is None or pd.isna(v):
                continue
            rows.append((p, probe_seq[p], float(v)))
        if len(rows) < 100:
            continue
        train_vals = [v for _, _, v in rows]  # anchors from all (no native split; probe-hash split below)
        # split by probe-hash bucket
        def bucket(p): return int(hashlib.sha1(p.encode()).hexdigest(), 16) % 1000
        tr = [v for p, _, v in rows if bucket(p) >= 200]
        anchors = np.quantile(np.array(tr), np.linspace(0, 1, SCORE_N_ANCHORS + 1)[1:-1]).tolist()
        q = (f"Estimate the binding preference (z-score) of RNA-binding protein {rbp} "
             f"for this RNA sequence; higher = stronger binding.")
        recs = []
        for p, seq, v in rows:
            b = bucket(p)
            split = "test" if b < 100 else ("dev" if b < 200 else "train")
            recs.append(score_record(task_id, ["rna"], seq, q, split,
                                     prov("RNAcompete", origin=f"RNAcompete/{rbp}"), v, anchors))
        counts = write_records(task_id, recs)
        manifest[task_id] = {"primitive": "score", "modality": ["rna"], "n_levels": SCORE_N_ANCHORS,
                             "anchors": [float(a) for a in anchors], "rbp": rbp,
                             **{s: counts[s] for s in ("train", "dev", "test")},
                             "provenance": prov("RNAcompete", origin=rbp), "status": "ready",
                             "note": f"probe subsample {len(valid)} of {len(probe_ids)}; split by probe-hash"}
        n += 1
    return n


def main():
    mp = OUT / "manifest.json"
    manifest = json.loads(mp.read_text()) if mp.exists() else {}
    pg = conv_proteingym(manifest)
    print(f"ProteinGym assays added: {pg}")
    rna = conv_rnacompete(manifest)
    print(f"RNAcompete RBP tasks added: {rna}")
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"total manifest tasks: {len(manifest)}")


if __name__ == "__main__":
    main()
