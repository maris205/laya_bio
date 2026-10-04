#!/usr/bin/env python3
"""Route A multimodal pilot: render proteins as amino-acid property heatmaps.

We have sequences + fold labels (lg_fold_class) but no 3D structures, so we
render a SYNTHETIC image: a 2D heatmap of per-residue physicochemical properties
(rows = properties, cols = sequence position). This is an honest synthetic
rendering (not a real structure/microscopy image) — flagged as such. It lets a
VLM attempt the same fold-classification question from an image, for a
sequence-vs-image cross-modal comparison.

Emits PNGs + a unified-format manifest with modality ["image"].
"""
from __future__ import annotations
import json, hashlib, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JEV = Path("/root/autodl-tmp/jev_gene")
SFT = JEV / "data/03_sft_biopaws2/jsonl/lg_fold_class.jsonl"
OUTDIR = JEV / "data/07_multimodal"

# 8 physicochemical properties per amino acid (normalized 0..1 where possible)
AA = "ACDEFGHIKLMNPQRSTVWY"
HYDRO = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,"H":-3.2,
         "I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
CHARGE = {"D":-1,"E":-1,"K":1,"R":1,"H":0.1}
VOLUME = {"G":60,"A":89,"S":89,"C":109,"D":111,"P":113,"N":114,"T":116,"E":138,"V":140,
          "Q":144,"H":153,"M":163,"I":167,"L":167,"K":169,"F":190,"R":174,"W":228,"Y":194}
POLAR = {"G":1,"A":0,"S":1,"C":0,"D":1,"P":0,"N":1,"T":1,"E":1,"V":0,"Q":1,"H":1,"M":0,"I":0,"L":0,"K":1,"F":0,"R":1,"W":0,"Y":1}
AROM = {"F":1,"W":1,"Y":1,"H":0.5}
HELIX = {"A":1.42,"R":0.98,"N":0.67,"D":1.01,"C":0.70,"Q":1.11,"E":1.51,"G":0.57,"H":1.00,
         "I":1.08,"L":1.21,"K":1.16,"M":1.45,"F":1.13,"P":0.57,"S":0.77,"T":0.83,"W":1.08,"Y":0.69,"V":1.06}
BETA = {"A":0.83,"R":0.93,"N":0.89,"D":0.54,"C":1.19,"Q":1.10,"E":0.37,"G":0.75,"H":0.87,
        "I":1.60,"L":1.30,"K":0.75,"M":1.05,"F":1.38,"P":0.55,"S":0.75,"T":1.19,"W":1.37,"Y":1.47,"V":1.70}

def norm(d, keys):
    vals = [d.get(k, 0.0) for k in keys]
    lo, hi = min(vals), max(vals)
    return {k: (d.get(k, 0.0) - lo) / (hi - lo + 1e-9) for k in keys}
Hn = norm(HYDRO, AA); Vn = norm(VOLUME, AA); Hxn = norm(HELIX, AA); Bn = norm(BETA, AA)
PROP_ROWS = [
    ("hydrophobicity", lambda a: Hn.get(a, 0.5)),
    ("charge", lambda a: (CHARGE.get(a, 0) + 1) / 2),
    ("volume", lambda a: Vn.get(a, 0.5)),
    ("polarity", lambda a: POLAR.get(a, 0)),
    ("aromaticity", lambda a: AROM.get(a, 0)),
    ("helix_prop", lambda a: Hxn.get(a, 0.5)),
    ("beta_prop", lambda a: Bn.get(a, 0.5)),
]

def render(seq, path, max_len=400):
    s = seq[:max_len].upper()
    M = np.zeros((len(PROP_ROWS), len(s)))
    for i, (_, fn) in enumerate(PROP_ROWS):
        for j, a in enumerate(s):
            M[i, j] = fn(a)
    fig, ax = plt.subplots(figsize=(max(3, len(s) * 0.04), 2.2), dpi=64)
    ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(PROP_ROWS))); ax.set_yticklabels([p[0] for p in PROP_ROWS], fontsize=5)
    ax.set_xticks([]); ax.set_xlabel("residue position", fontsize=6)
    ax.set_title("protein property map (synthetic render)", fontsize=6)
    fig.tight_layout(pad=0.3); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args()
    imgdir = OUTDIR / "fold_images"; imgdir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in SFT.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == a.split]
    rng = np.random.default_rng(a.seed)
    if len(rows) > a.n:
        rows = [rows[i] for i in rng.choice(len(rows), size=a.n, replace=False)]
    manifest = []
    import re
    for r in rows:
        content = r["messages"][0]["content"]
        seq = content.rsplit("\n", 1)[-1].strip()
        if not re.fullmatch(r"[A-Za-z*]+", seq):
            continue
        g = hashlib.sha1(seq.encode()).hexdigest()[:12]
        p = imgdir / f"{g}.png"
        if not p.exists():
            render(seq, p)
        manifest.append({"task_id": "fold_image", "primitive": "choice", "modality": ["image"],
            "images": [{"role": "query", "format": "png", "path": str(p)}],
            "question": "This image is a synthetic property map of a protein sequence. Which structural fold class does the protein belong to?",
            "candidates": r["choices"], "answer": r["answer_short"], "group": g, "split": a.split,
            "provenance": {"source": "lg_fold_class rendered (route A synthetic)", "license": r.get("license", "?")},
            "_seqlen": len(seq)})
    (OUTDIR / "fold_image_tasks.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n")
    print(f"rendered {len(manifest)} images -> {imgdir}; manifest {OUTDIR/'fold_image_tasks.jsonl'}")
    from collections import Counter
    print("label dist:", dict(Counter(m["answer"] for m in manifest)))
    print("seqlen med:", int(np.median([m["_seqlen"] for m in manifest])))

if __name__ == "__main__":
    main()
