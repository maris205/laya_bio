#!/usr/bin/env python3
"""Build benchmark v2 in the unified Jev-style interface.

Unified record schema (one JSON object per line):
  task_id, primitive (noul|choice|score|multi_noul), modality (list),
  sequences: [{role, modality, sequence}], question,
  candidates (choice/noul), answer (choice/noul),
  score_levels + anchors + gold_value + gold_level (score),
  labels (multi_noul, {label: 0/1}),
  group (entity id for leak-proof split), split (train|dev|test),
  provenance {source, revision, license, origin}

Slice 1 converts sources with native or cleanly-derived splits. Sources that
need entity/cluster isolation or an external official split are registered with
status="needs_work" and are not emitted here.
"""
from __future__ import annotations
import json, os, re, sys, glob, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

JEV = Path("/root/autodl-tmp/jev_gene")
RAW = JEV / "data/05_benchmark_v2_raw"
EXT = JEV / "data/05_task_extension_sources"
SFT = JEV / "data/03_sft_biopaws2/jsonl"
OUT = JEV / "data/06_benchmark_v2_unified"
RECEIPT = json.loads((RAW / "download_receipt.json").read_text())

# revision + license lookup keyed by subdir / name
REV = {}
for d in RECEIPT.get("datasets", []):
    REV[d["subdir"]] = {"revision": d["revision"], "license": d.get("license", "?"), "source": d["repo_id"]}
for d in RECEIPT.get("github_clones", []):
    REV[d["name"]] = {"revision": d.get("commit"), "license": "?", "source": d.get("task")}

SCORE_N_ANCHORS = 5


def prov(subdir, origin=None):
    r = REV.get(subdir, {"revision": None, "license": "?"})
    return {"revision": r["revision"], "license": r["license"],
            "source": r.get("source", subdir), "origin": origin}


def write_records(task_id, records):
    """records: list of dicts each already carrying a 'split' field."""
    d = OUT / task_id
    d.mkdir(parents=True, exist_ok=True)
    counts = Counter(r["split"] for r in records)
    for split in ("train", "dev", "test"):
        with open(d / f"{split}.jsonl", "w") as fh:
            for r in records:
                if r["split"] == split:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return counts


def base(task_id, primitive, modality, sequences, question, split, provenance, group=None):
    return {"task_id": task_id, "primitive": primitive, "modality": modality,
            "sequences": sequences, "question": question, "split": split,
            "group": group or "", "provenance": provenance}


# ---------- GUE: DNA, choice, native train/dev/test ----------
def conv_gue(manifest):
    tasks = {}
    for tf in sorted(glob.glob(str(RAW / "gue/**/train*.csv"), recursive=True)):
        # reconstruct readable task id (sanitize slashes from nested GUE tasks)
        rel = Path(tf).parent.relative_to(RAW / "gue")
        name = "_".join(rel.parts)  # e.g. prom_prom_300_tata, tf_0, H3K4me3, mouse_0
        task_id = f"gue_{name}"
        dev = glob.glob(str(Path(tf).parent / "dev*.csv"))
        test = glob.glob(str(Path(tf).parent / "test*.csv"))
        tr = pd.read_csv(tf)
        recs = []
        labels = sorted(pd.concat([tr]+[pd.read_csv(x) for x in dev+test])["label"].unique().tolist())
        cand = [str(l) for l in labels]
        q = f"Classify the following DNA sequence for the GUE task '{name}'. Candidate classes are: {', '.join(cand)}."
        for split, dfs in [("train", [tr]), ("dev", [pd.read_csv(x) for x in dev]), ("test", [pd.read_csv(x) for x in test])]:
            for df in dfs:
                for seq, lab in zip(df["sequence"].astype(str), df["label"]):
                    r = base(task_id, "choice", ["dna"],
                             [{"role": "query", "modality": "dna", "sequence": seq}],
                             q, split, prov("gue", origin="GUE/DNABERT-2"),
                             group=hashlib.sha1(seq.encode()).hexdigest()[:12])
                    r["candidates"] = cand
                    r["answer"] = str(lab)
                    recs.append(r)
        counts = write_records(task_id, recs)
        tasks[task_id] = {"primitive": "choice", "modality": ["dna"],
                               "n_classes": len(cand), **{s: counts[s] for s in ("train","dev","test")},
                               "provenance": prov("gue", origin="GUE/DNABERT-2"), "status": "ready"}
    manifest.update(tasks)
    return len(tasks)


# ---------- Genomic Benchmarks: DNA, noul(binary)/choice(3), native train/test ----------
def conv_genomic(manifest):
    tasks = {}
    for sub in sorted(p for p in (RAW / "genomic_benchmarks").iterdir() if p.is_dir() and not p.name.startswith(".")):
        name = sub.name
        trp = glob.glob(str(sub / "data/train*.parquet"))
        tep = glob.glob(str(sub / "data/test*.parquet"))
        tr = pd.concat([pd.read_parquet(f) for f in trp])
        te = pd.concat([pd.read_parquet(f) for f in tep])
        labels = sorted(set(tr["label"].unique().tolist()) | set(te["label"].unique().tolist()))
        binary = len(labels) == 2 and set(labels) == {0, 1}
        prim = "noul" if binary else "choice"
        cand = ["no", "yes"] if binary else [str(l) for l in labels]
        # carve dev from train by hash bucket (test stays the official held-out)
        tr = tr.reset_index(drop=True)
        bucket = (tr["seq"].astype(str).map(lambda s: int(hashlib.sha1(s.encode()).hexdigest(), 16) % 1000))
        dev_mask = bucket < 100
        q = (f"Does this DNA sequence belong to the positive class for '{name}'?" if binary
             else f"Classify the following DNA sequence for '{name}'. Classes: {', '.join(cand)}.")
        recs = []
        for split, df in [("train", tr[~dev_mask]), ("dev", tr[dev_mask]), ("test", te)]:
            for seq, lab in zip(df["seq"].astype(str), df["label"]):
                lab = int(lab)
                ans = ("yes" if lab == 1 else "no") if binary else str(lab)
                r = base(f"gb_{name}", prim, ["dna"],
                         [{"role": "query", "modality": "dna", "sequence": seq}],
                         q, split, prov(f"genomic_benchmarks/{name}", origin="GenomicBenchmarks"),
                         group=hashlib.sha1(seq.encode()).hexdigest()[:12])
                r["candidates"] = cand
                r["answer"] = ans
                recs.append(r)
        counts = write_records(f"gb_{name}", recs)
        tasks[f"gb_{name}"] = {"primitive": prim, "modality": ["dna"], "n_classes": len(cand),
                               **{s: counts[s] for s in ("train","dev","test")},
                               "provenance": prov(f"genomic_benchmarks/{name}"), "status": "ready"}
    manifest.update(tasks)
    return len(tasks)


def _quantile_anchors(values, n):
    qs = np.linspace(0, 1, n + 1)[1:-1]
    return np.quantile(values, qs).tolist()


def _assign_level(value, anchors):
    for i, a in enumerate(anchors):
        if value <= a:
            return str(i)
    return str(len(anchors))


# ---------- Score: TAPE stability, DeepSTARR (native splits) ----------
def _emit_score(task_id, prim_rows, question, provenance, value_col):
    """prim_rows: list of (split, seq, value). Anchors from train only."""
    recs = []
    train_vals = [v for s, seq, v in prim_rows if s == "train"]
    anchors = _quantile_anchors(np.array(train_vals), SCORE_N_ANCHORS)
    levels = [str(i) for i in range(SCORE_N_ANCHORS)]
    for s, seq, v in prim_rows:
        r = base(task_id, "score", _modality_of(task_id),
                 [{"role": "query", "modality": _modality_of(task_id)[0], "sequence": seq}],
                 question, s, provenance, group=hashlib.sha1(seq.encode()).hexdigest()[:12])
        r["score_levels"] = levels
        r["anchors"] = [float(a) for a in anchors]
        r["gold_value"] = float(v)
        r["gold_level"] = _assign_level(v, anchors)
        recs.append(r)
    return recs, anchors


def _modality_of(task_id):
    if task_id.startswith(("tape_stability", "tape_fluo")):
        return ["protein"]
    if task_id.startswith("deepstarr"):
        return ["dna"]
    return ["dna"]


def conv_tape_stability(manifest):
    rows = []
    for split, fn in [("train","stability_train.json"),("dev","stability_valid.json"),("test","stability_test.json")]:
        data = json.load(open(EXT / "tape/stability/stability" / fn))
        for it in data:
            val = it["stability_score"]
            val = float(val[0]) if isinstance(val, list) else float(val)
            rows.append((split, it["primary"], val))
    recs, anchors = _emit_score("tape_stability", rows,
        "Estimate the stability of the following protein sequence on its native experimental scale (higher = more stable).",
        prov("tape_fluorescence", origin="TAPE/stability"), None)
    counts = write_records("tape_stability", recs)
    manifest["tape_stability"] = {"primitive":"score","modality":["protein"],
        "n_levels":SCORE_N_ANCHORS,"anchors":[float(a) for a in anchors],
        **{s:counts[s] for s in ("train","dev","test")},
        "provenance":prov("tape_fluorescence",origin="TAPE/stability"),"status":"ready"}
    return 1


def conv_deepstarr(manifest):
    tr = pd.read_parquet(RAW / "deepstarr_enhancer_activity/train.parquet")
    dev = pd.read_parquet(RAW / "deepstarr_enhancer_activity/valid.parquet")
    te = pd.read_parquet(RAW / "deepstarr_enhancer_activity/test.parquet")
    n = 0
    for col, nm, desc in [("Dev_log2_enrichment_scaled", "dev", "developmental"),
                          ("Hk_log2_enrichment_scaled", "hk", "housekeeping")]:
        rows = []
        for split, df in [("train", tr), ("dev", dev), ("test", te)]:
            for seq, v in zip(df["sequence"].astype(str), df[col]):
                rows.append((split, seq, float(v)))
        recs, anchors = _emit_score(f"deepstarr_{nm}", rows,
            f"Estimate the {desc} enhancer activity (log2 enrichment) of this DNA sequence.",
            prov("deepstarr_enhancer_activity", origin="DeepSTARR"), None)
        counts = write_records(f"deepstarr_{nm}", recs)
        manifest[f"deepstarr_{nm}"] = {"primitive":"score","modality":["dna"],
            "n_levels":SCORE_N_ANCHORS,"anchors":[float(a) for a in anchors],
            **{s:counts[s] for s in ("train","dev","test")},
            "provenance":prov("deepstarr_enhancer_activity"),"status":"ready"}
        n += 1
    return n


# ---------- DeepLoc 2.0: protein, multi_noul (11 locs), Partition groups ----------
DEEPLOC_LOCS = ['Membrane','Cytoplasm','Nucleus','Extracellular','Cell membrane',
                'Mitochondrion','Plastid','Endoplasmic reticulum','Lysosome/Vacuole',
                'Golgi apparatus','Peroxisome']

def conv_deeploc(manifest):
    d = pd.read_csv(EXT / "deeploc2/Swissprot_Train_Validation_dataset.csv")
    # constructed split: partition 0 -> test, 1-3 -> train, 4 -> dev (documented)
    def sp(p):
        return {0:"test"}.get(p, "dev" if p == 4 else "train")
    recs = []
    seqs = d["Sequence"].astype(str).tolist()
    parts = d["Partition"].astype(int).tolist()
    label_cols = {loc: d[loc].astype(int).tolist() for loc in DEEPLOC_LOCS}
    for i in range(len(d)):
        seq = seqs[i]
        labs = {loc: label_cols[loc][i] for loc in DEEPLOC_LOCS}
        r = base("deeploc_multi", "multi_noul", ["protein"],
                 [{"role":"query","modality":"protein","sequence":seq}],
                 "Which of the following subcellular locations does this protein reside in? Report yes/no for each.",
                 sp(parts[i]), prov("deeploc2", origin="DeepLoc2.0"),
                 group=hashlib.sha1(seq.encode()).hexdigest()[:12])
        r["candidates"] = DEEPLOC_LOCS
        r["labels"] = labs
        r["answer"] = [loc for loc in DEEPLOC_LOCS if labs[loc] == 1]
        recs.append(r)
    counts = write_records("deeploc_multi", recs)
    manifest["deeploc_multi"] = {"primitive":"multi_noul","modality":["protein"],
        "n_labels":len(DEEPLOC_LOCS),"labels":DEEPLOC_LOCS,
        **{s:counts[s] for s in ("train","dev","test")},
        "provenance":prov("deeploc2"),"status":"ready","note":"split is constructed from Partition column (p0=test, p1-3=train, p4=dev)"}
    return 1


# ---------- local LLaMA-Gene / BioPAWS snapshots: extract sequence ----------
def _extract_seqs(content, modality):
    """Pull raw sequence(s) from the question text. Returns (seqs, ok)."""
    # paired (two 'Sequence N:')
    if re.search(r"Sequence 1:", content):
        s1 = re.search(r"Sequence 1:\s*([A-Za-z*\-]+)", content)
        s2 = re.search(r"Sequence 2:\s*([A-Za-z*\-]+)", content)
        if s1 and s2:
            return [s1.group(1), s2.group(1)], True
        return [], False
    # single: last whitespace-separated token run that looks like a bare sequence
    tail = content.rsplit("\n", 1)[-1].strip()
    if re.fullmatch(r"[A-Za-z*\-]+", tail) and len(tail) >= 10:
        return [tail], True
    return [], False


def conv_local_sft(manifest):
    specs = [
        ("lg_fold_class", "lg_fold_class", "choice", ["protein"], "F5_structure"),
        ("lg_signal_peptide", "lg_signal_peptide", "choice", ["protein"], "F2_functional"),
        ("lg_subcellular_loc", "lg_subcellular_loc", "choice", ["protein"], "F2_functional"),
        ("lg_npp", "lg_npp", "choice", ["protein"], "F2_functional"),
        ("protein_homology_std", "protein_homology_std", "noul", ["protein"], "F1_pairwise"),
        ("protein_homology_remote", "protein_homology_remote", "noul", ["protein"], "F1_pairwise"),
    ]
    n = 0
    for fname, task_id, prim, modality, fam in specs:
        recs, skipped = [], 0
        lines = (SFT / f"{fname}.jsonl").read_text().splitlines()
        for line in lines:
            d = json.loads(line)
            content = d["messages"][0]["content"]
            seqs, ok = _extract_seqs(content, modality)
            if not ok:
                skipped += 1
                continue
            nseq = len(seqs)
            q = content.rsplit("\n", 1)[0].strip() if nseq == 1 else content.split("Sequence 1:")[0].strip()
            provx = {"revision": None, "license": d.get("license", "?"),
                     "source": d.get("source", fname), "origin": "llama-gene/biopaws snapshot"}
            split = "dev" if d["split"] == "val" else d["split"]
            r = base(task_id, prim, modality,
                     [{"role": "query" if nseq == 1 else f"query_{i+1}", "modality": modality[0], "sequence": s}
                      for i, s in enumerate(seqs)],
                     q, split, provx,
                     group=hashlib.sha1("".join(seqs).encode()).hexdigest()[:12])
            cand = d.get("choices", [])
            if prim == "noul":
                # map to yes/no if choices are Yes/No
                r["candidates"] = ["no", "yes"]
                ans = d["answer_short"]
                r["answer"] = "yes" if str(ans).lower().startswith("y") else ("no" if str(ans).lower().startswith("n") else str(ans))
            else:
                r["candidates"] = cand
                r["answer"] = str(d["answer_short"])
            recs.append(r)
        # local snapshots use train/val/test already
        counts = write_records(task_id, recs)
        manifest[task_id] = {"primitive":prim,"modality":modality,"n_classes":len(cand),
            **{s:counts[s] for s in ("train","dev","test")},"skipped_extract_fail":skipped,
            "provenance":provx,"status":"ready"}
        n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    report = {}
    report["gue"] = conv_gue(manifest)
    report["genomic_benchmarks"] = conv_genomic(manifest)
    report["tape_stability"] = conv_tape_stability(manifest)
    report["deepstarr"] = conv_deepstarr(manifest)
    report["deeploc"] = conv_deeploc(manifest)
    report["local_sft"] = conv_local_sft(manifest)
    # write manifest
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Converted source groups:", json.dumps(report))
    print("Total tasks emitted:", len(manifest))


if __name__ == "__main__":
    main()
