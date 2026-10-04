#!/usr/bin/env python3
"""Noul candidate-phrasing ablation for #1 (shared typed-decision scorer).

#1's noul weakness (AUROC 0.577 vs #2's dedicated-head 0.620) may come from the
candidate TEXT the shared scorer must discriminate: the harness builds
["false: no", "true: yes"] which is redundant. Because #1 scores dynamic
candidate text, we can change the phrasing at INFERENCE time (no retraining) and
measure noul AUROC per phrasing. AUROC is rank-based, so temperature/calibration
cannot change it — only the discriminability of the candidate representations.

Loads a #1 checkpoint and evaluates the GB noul tasks under several phrasings.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

JEV = Path("/root/autodl-tmp/jev_gene")
FROZEN = JEV / "artifacts/laya_jev_multitask_v1/round/frozen_code"
MODEL = JEV / "artifacts/laya_model"
CPT = JEV / "artifacts/laya_biocpt_v2"
OUT = JEV / "data/06_benchmark_v2_unified"
sys.path.insert(0, str(FROZEN))
import torch
from safetensors.torch import load_file
from laya_jev_multitask_data import Representation
import laya_jev_multitask_train as t
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score

MAXLEN = 512

# phrasing variants: (neg_text, pos_text); index 0 = neg(label0), 1 = pos(label1)
PHRASINGS = {
    "current_false_no":   ("false: no", "true: yes"),
    "raw_no_yes":         ("no", "yes"),
    "plain_false_true":   ("false", "true"),
    "descriptive":        ("false: this is not a positive-class sequence",
                           "true: this is a positive-class sequence"),
    "proposition":        ("no, it does not", "yes, it does"),
    "neg_pos_words":      ("negative", "positive"),
}


def load_model(ckpt):
    model = t.build(MODEL, CPT, "no_cpt")
    model.load_state_dict(load_file(str(ckpt)), strict=True)
    model.eval()
    return model


@torch.no_grad()
def eval_noul(model, rep, pad, task_dir, neg, pos, maxn=200, seed=20261001):
    f = task_dir / "test.jsonl"
    if not f.exists(): f = task_dir / "dev.jsonl"
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(seed)
    if len(recs) > maxn:
        recs = [recs[i] for i in rng.choice(len(recs), size=maxn, replace=False)]
    rows = []
    for rec in recs:
        seqs = rec["sequences"]
        if len(seqs) != 1 or seqs[0]["modality"] not in ("dna", "protein"): continue
        s = seqs[0]["sequence"]; m = seqs[0]["modality"]
        label = 1 if rec["answer"] == "yes" else 0
        row = {"id": rec.get("group",""), "primitive": "noul", "question": rec["question"],
               "modality": m, "sequence": s, "choices": [neg, pos], "label": label}
        try: row = rep.prepare(row)
        except Exception: continue
        if row["length"] > MAXLEN: continue
        rows.append(row)
    if len(rows) < 8: return None
    ys, pos_prob = [], []
    for st in range(0, len(rows), 16):
        rr = [t.render(r) for r in rows[st:st+16]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**t.pack(rr, pad))
        for row, logit in zip(rr, logits):
            p = logit[:2].float().softmax(-1).cpu().numpy()
            canon = np.zeros(2)
            for j, idx in enumerate(row["order"]): canon[idx] = p[j]
            ys.append(row["label"]); pos_prob.append(canon[1])
    ys = np.array(ys); pp = np.array(pos_prob)
    if len(set(ys.tolist())) < 2: return {"auroc": None, "acc": float((pp>0.5).astype(int).eq(ys).mean()), "n": len(ys)}
    return {"auroc": float(roc_auc_score(ys, pp)), "acc": float(((pp>0.5).astype(int)==ys).mean()), "n": len(ys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=JEV/"artifacts/benchmark_v2_train/bv2_pf/model.safetensors")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--max-per-task", type=int, default=200)
    a = ap.parse_args()
    tasks = a.tasks or [d.name for d in sorted(OUT.iterdir())
                        if d.is_dir() and d.name.startswith("gb_")
                        and (d/"test.jsonl").exists()
                        and json.loads(open(d/"test.jsonl").readline())["primitive"] == "noul"]
    print(f"[ablation] ckpt={a.ckpt.name} tasks={len(tasks)}: {tasks}", flush=True)
    model = load_model(a.ckpt)
    rep = Representation(CPT / "data")
    pad = AutoTokenizer.from_pretrained(CPT / "data/representation/base_tokenizer").pad_token_id
    table = {}
    for name, (neg, pos) in PHRASINGS.items():
        aurocs, accs = [], []
        for tk in tasks:
            r = eval_noul(model, rep, pad, OUT/tk, neg, pos, a.max_per_task)
            if r and r.get("auroc") is not None:
                aurocs.append(r["auroc"]); accs.append(r["acc"])
        table[name] = {"neg": neg, "pos": pos, "n_tasks": len(aurocs),
                       "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
                       "mean_acc": float(np.mean(accs)) if accs else None,
                       "per_task_auroc": {tk: None for tk in tasks}}
        # per-task detail
        pt = {}
        for tk in tasks:
            r = eval_noul(model, rep, pad, OUT/tk, neg, pos, a.max_per_task)
            pt[tk] = r.get("auroc") if r else None
        table[name]["per_task_auroc"] = pt
        print(f"  {name:20s} mean AUROC={table[name]['mean_auroc']:.3f} acc={table[name]['mean_acc']:.3f}  (neg='{neg[:25]}' pos='{pos[:25]}')", flush=True)
    best = max((k for k in table if table[k]["mean_auroc"] is not None), key=lambda k: table[k]["mean_auroc"])
    print(f"\n[best phrasing] {best}: AUROC {table[best]['mean_auroc']:.3f}")
    outp = JEV/"artifacts/benchmark_v2_eval/noul_ablation.json"
    outp.write_text(json.dumps({"ckpt": str(a.ckpt), "tasks": tasks, "best": best, "table": table}, ensure_ascii=False, indent=2)+"\n")
    print(f"saved {outp}")


if __name__ == "__main__":
    main()
