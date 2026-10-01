#!/usr/bin/env python3
"""Benchmark v2 slice 2: DNA pools with old-test isolation, double-sequence
gene_lan_transfer, TAPE fluorescence Score. Appends into the same unified dir.

Isolation principle (catalog): the dnagpt upstream pools are unsplit 'train'
pools that CONTAIN the paper's old blind-test members. We do NOT randomly
re-split a pool and call it a blind test. Instead:
  * test  = the old lg_* test split (the paper's blind test), kept verbatim;
  * train/dev = pool sequences MINUS every old-test member (isolated out);
  * any sequence whose pool int-label disagrees with its lg test text-label is
    a genuine annotation conflict -> quarantined, not majority-voted.
Label int->text mapping is the one empirically consistent with the lg test on
all shared sequences (splice: old convention; see build protocol caveat).
"""
from __future__ import annotations
import json, hashlib
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
SCORE_N_ANCHORS = 5

REV = {}
for d in RECEIPT.get("datasets", []):
    REV[d["subdir"]] = {"revision": d["revision"], "license": d.get("license", "?"), "source": d["repo_id"]}


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


def seqgroup(seq):
    return hashlib.sha1(seq.encode()).hexdigest()[:12]


# ---------- DNA pool with old-test isolation ----------
def load_lg_test_text(lg_file):
    """seq -> answer_short (text) for the lg test split."""
    out = {}
    for l in open(SFT / f"{lg_file}.jsonl"):
        d = json.loads(l)
        if d["split"] != "test":
            continue
        c = d["messages"][0]["content"]
        tail = c.rsplit("\n", 1)[-1].strip()
        out[tail] = str(d["answer_short"])
    return out


def conv_dna_pool(task_id, pool_dir, lg_file, int2text, question, modality=("dna",)):
    lg_test = load_lg_test_text(lg_file)
    pool = pd.read_parquet(EXT / pool_dir / "data/train-00000-of-00001.parquet")
    pool_seq2int = {}
    for s, lab in zip(pool["sequence"].astype(str), pool["label"]):
        pool_seq2int[s] = int(lab)
    cand = sorted(set(int2text.values()) | set(lg_test.values()))
    old_test_seqs = set(lg_test)
    recs = []
    conflict = 0
    # TEST = old lg test (blind), verify pool int agrees (else quarantine)
    for s, ans in lg_test.items():
        if s in pool_seq2int:
            mapped = int2text.get(pool_seq2int[s])
            if mapped is not None and mapped != ans:
                conflict += 1
                continue  # annotation conflict -> quarantine (drop from test too)
        r = base(task_id, "choice", list(modality),
                 [{"role": "query", "modality": modality[0], "sequence": s}],
                 question, "test", prov(pool_dir, origin="dnagpt pool + lg blind test"), group=seqgroup(s))
        r["candidates"] = cand; r["answer"] = ans
        recs.append(r)
    # TRAIN/DEV = pool minus old-test members, split by seq-hash bucket
    train_pool = [(s, i) for s, i in pool_seq2int.items() if s not in old_test_seqs]
    dev_frac = 0.1
    for s, i in train_pool:
        bucket = int(hashlib.sha1(s.encode()).hexdigest(), 16) % 1000
        split = "dev" if bucket < dev_frac * 1000 else "train"
        ans = int2text.get(i)
        if ans is None:
            continue
        r = base(task_id, "choice", list(modality),
                 [{"role": "query", "modality": modality[0], "sequence": s}],
                 question, split, prov(pool_dir, origin="dnagpt pool"), group=seqgroup(s))
        r["candidates"] = cand; r["answer"] = ans
        recs.append(r)
    counts = write_records(task_id, recs)
    return {"primitive": "choice", "modality": list(modality), "n_classes": len(cand),
            **{s: counts[s] for s in ("train", "dev", "test")},
            "quarantined_conflicts": conflict, "provenance": prov(pool_dir),
            "status": "ready", "note": "test=old lg blind test; train/dev=pool minus old-test members"}


# ---------- gene_lan_transfer: double-sequence Noul, grouped by endpoint components ----------
class DSU:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def conv_gene_lan(task_id, config, question, modality_pair):
    p = glob_one(EXT / "gene_lan_transfer" / config)
    d = pd.read_parquet(p)
    dsu = DSU()
    pairs = []
    for s1, s2, lab in zip(d["sentence1"].astype(str), d["sentence2"].astype(str), d["label"]):
        dsu.union(s1, s2)
        pairs.append((s1, s2, int(lab)))
    # component -> deterministic bucket for split
    comp_of = {x: dsu.find(x) for x in list(dsu.p)}
    comp_bucket = {}
    for x, c in comp_of.items():
        if c not in comp_bucket:
            comp_bucket[c] = int(hashlib.sha1(c.encode()).hexdigest(), 16) % 1000
    recs = []
    for s1, s2, lab in pairs:
        comp = dsu.find(s1)
        b = comp_bucket[comp]
        split = "test" if b < 100 else ("dev" if b < 200 else "train")
        seqs = [{"role": "query_1", "modality": modality_pair[0], "sequence": s1},
                {"role": "query_2", "modality": modality_pair[1], "sequence": s2}]
        r = base(task_id, "noul", list(modality_pair), seqs, question, split,
                 prov("gene_lan_transfer", origin=f"dnagpt/gene_lan_transfer:{config}"),
                 group=seqgroup(s1 + "|" + s2))
        r["candidates"] = ["no", "yes"]; r["answer"] = "yes" if lab == 1 else "no"
        recs.append(r)
    counts = write_records(task_id, recs)
    return {"primitive": "noul", "modality": list(modality_pair), "n_classes": 2,
            **{s: counts[s] for s in ("train", "dev", "test")},
            "provenance": prov("gene_lan_transfer", origin=config), "status": "ready",
            "note": "constructed-similarity diagnostic; grouped split by endpoint connected-components; cross-config endpoint reuse remains"}


def glob_one(d):
    g = list(Path(d).glob("train*.parquet"))
    return g[0]


# ---------- TAPE fluorescence Score ----------
def conv_tape_fluo():
    rows = []
    for split, fn in [("train", "fluorescence_train.json"), ("dev", "fluorescence_valid.json"), ("test", "fluorescence_test.json")]:
        data = json.load(open(EXT / "tape/fluorescence" / fn))
        for it in data:
            val = it["log_fluorescence"]
            val = float(val[0]) if isinstance(val, list) else float(val)
            rows.append((split, it["primary"], val))
    train_vals = [v for s, seq, v in rows if s == "train"]
    anchors = np.quantile(np.array(train_vals), np.linspace(0, 1, SCORE_N_ANCHORS + 1)[1:-1]).tolist()
    recs = []
    for s, seq, v in rows:
        r = base("tape_fluorescence", "score", ["protein"],
                 [{"role": "query", "modality": "protein", "sequence": seq}],
                 "Estimate the fluorescence level (log scale) of this GFP variant.",
                 s, prov("tape_fluorescence", origin="TAPE/fluorescence"), group=seqgroup(seq))
        lvl = str(sum(1 for a in anchors if v > a))
        r["score_levels"] = [str(i) for i in range(SCORE_N_ANCHORS)]
        r["anchors"] = [float(a) for a in anchors]
        r["gold_value"] = float(v); r["gold_level"] = lvl
        recs.append(r)
    counts = write_records("tape_fluorescence", recs)
    return {"primitive": "score", "modality": ["protein"], "n_levels": SCORE_N_ANCHORS,
            "anchors": [float(a) for a in anchors],
            **{s: counts[s] for s in ("train", "dev", "test")},
            "provenance": prov("tape_fluorescence"), "status": "ready"}


def main():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    report = {}
    # DNA pools with old-test isolation (splice/tf/core); promoter uses lg native splits (slice1-style, added below)
    report["dna_splice"] = conv_dna_pool("dna_splice", "dna_splice_site_prediction", "lg_splice_site",
        {0: "Non-Splice Sites", 1: "Acceptor Sites", 2: "Donor Sites"},
        "Classify the following DNA sequence's splice-site type. Candidates: Non-Splice Sites, Acceptor Sites, Donor Sites.")
    report["dna_tf"] = conv_dna_pool("dna_tf", "dna_transcription_factor_prediction", "lg_tf_prediction",
        {0: "Background Sequences", 1: "Binding Sites"},
        "Is the following DNA sequence a transcription-factor binding site? Candidates: Background Sequences, Binding Sites.")
    report["dna_core"] = conv_dna_pool("dna_core", "dna_core_promoter", "lg_core_promoter_detection",
        {0: "Non-promoter", 1: "promoter"},
        "Is the following DNA sequence a core promoter? Candidates: Non-promoter, promoter.")
    # gene_lan_transfer 8 configs
    gl_specs = [
        ("gl_dna_protein_pair", "dna_protein_pair", ("dna", "protein"),
         "Do the following DNA sequence and protein sequence form a correct coding pair?"),
        ("gl_dna_protein_rand", "dna_protein_pair_rand", ("dna", "protein"),
         "Do the following DNA sequence and protein sequence form a correct coding pair?"),
        ("gl_dna_protein_rand_v2", "dna_protein_pair_rand_v2", ("dna", "protein"),
         "Do the following DNA sequence and protein sequence form a correct coding pair?"),
        ("gl_dna_sim_150", "dna_sim_pair_150bp", ("dna", "dna"),
         "Are the following two DNA sequences similar (homologous by construction)?"),
        ("gl_dna_sim_50", "dna_sim_pair_50bp", ("dna", "dna"),
         "Are the following two DNA sequences similar (homologous by construction)?"),
        ("gl_dna_sim_simple_150", "dna_sim_pair_simple_150bp", ("dna", "dna"),
         "Are the following two DNA sequences similar (homologous by construction)?"),
        ("gl_protein_sim_150", "protein_sim_pair_150bp", ("protein", "protein"),
         "Are the following two protein sequences similar (homologous by construction)?"),
        ("gl_protein_sim_450", "protein_sim_pair_450bp", ("protein", "protein"),
         "Are the following two protein sequences similar (homologous by construction)?"),
    ]
    for tid, cfg, mods, q in gl_specs:
        report[tid] = conv_gene_lan(tid, cfg, q, mods)
    report["tape_fluorescence"] = conv_tape_fluo()
    manifest.update(report)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("slice2 tasks added:", len(report), "| total manifest tasks:", len(manifest))
    for k, v in report.items():
        print(f"  {k:26s} {v['primitive']:10s} train={v['train']:6d} dev={v['dev']:5d} test={v['test']:6d} quar={v.get('quarantined_conflicts',0)}")


if __name__ == "__main__":
    main()
