#!/usr/bin/env python3
"""Evaluate a Laya-JEV checkpoint on benchmark v2 (unified interface).

Reuses the frozen MVP1 code (Representation / build / render / pack /
prediction_records) so the model consumes benchmark v2 records through the
same typed-decision path. The MVP1 SharedDecision encoder is single-sequence
and DNA/protein-only with a 512-token budget, so this harness supports:
  * primitives noul / choice / score,
  * modality single {dna, protein},
and reports skip reasons for multi_noul, rna, double-sequence, and over-budget.

This is a zero-shot generalization probe of the shared decision interface on
tasks the checkpoint was NOT trained on (only promoter/structural/fluorescence
were trained). Metrics are per task; aggregates are macro over tasks.
"""
from __future__ import annotations
import argparse, json, sys, importlib
from pathlib import Path
from collections import defaultdict
import numpy as np

JEV = Path("/root/autodl-tmp/jev_gene")
FROZEN = JEV / "artifacts/laya_jev_multitask_v1/round/frozen_code"
MODEL = JEV / "artifacts/laya_model"
CPT = JEV / "artifacts/laya_biocpt_v2"
DATA_DIR = JEV / "artifacts/laya_jev_multitask_v1/data"
OUT = JEV / "data/06_benchmark_v2_unified"

sys.path.insert(0, str(FROZEN))
import torch
from safetensors.torch import load_file
from laya_jev_multitask_data import Representation
import laya_jev_multitask_train as t

MAXLEN = 512
TYPE_IDS = {"choice": 0, "score": 1, "noul": 2}
SUPPORTED = {"noul", "choice", "score"}


def load_model(ckpt):
    model = t.build(MODEL, CPT, "no_cpt")
    state = load_file(str(ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def to_row(rec, task_anchors):
    """benchmark v2 record -> MVP1 row schema consumed by Representation."""
    seqs = rec["sequences"]
    if len(seqs) != 1:
        return None, "double_sequence"
    s = seqs[0]
    if s["modality"] not in ("dna", "protein"):
        return None, f"modality_{s['modality']}"
    prim = rec["primitive"]
    if prim == "multi_noul":
        return None, "multi_noul"
    q = rec["question"]
    if prim == "noul":
        # raw candidates (["no","yes"]); "false:/true:" prefix hurt noul AUROC (noul_ablation)
        choices = list(rec["candidates"])
        label = 1 if rec["answer"] == "yes" else 0
    elif prim == "choice":
        choices = list(rec["candidates"])
        if rec["answer"] not in choices:
            return None, "answer_not_in_candidates"
        label = choices.index(rec["answer"])
    elif prim == "score":
        if task_anchors is None:
            return None, "no_anchors"
        # benchmark stores interior cut points (n_levels-1); MVP1 needs n anchor
        # VALUES. Derive 5 representative anchors: bin midpoints + extrapolated ends.
        c = list(task_anchors)
        if len(c) + 1 != len(rec.get("score_levels", c)):  # fall back to cuts+1 levels
            pass
        if len(c) >= 2:
            step = (c[-1] - c[0]) / (len(c) - 1)
            a5 = [c[0] - step / 2, (c[0] + c[1]) / 2, (c[1] + c[2]) / 2, (c[2] + c[3]) / 2, c[-1] + step / 2]
        else:
            a5 = [c[0]] * 5 if c else [0.0] * 5
        row_anchors = a5
        choices = [f"value = {a:.6g}" for a in a5]
        label = int(rec["gold_level"])
        if not 0 <= label < len(choices):
            return None, "level_out_of_range"
    else:
        return None, "unknown_primitive"
    row = {"id": rec.get("group", "") + ":" + str(hash(s["sequence"])) [:8], "primitive": prim,
           "question": q, "modality": s["modality"], "sequence": s["sequence"],
           "choices": choices, "label": label}
    if prim == "score":
        row["value"] = float(rec["gold_value"])
        row["_anchors"] = row_anchors  # the 5 derived anchor values
        row["target_probs"] = None  # not needed for eval
    return row, None


@torch.no_grad()
def run_task(model, rep, task_dir, maxn, score_spec_holder, pad, seed=20261001):
    # Read the full test split (fall back to dev), then sample RANDOMLY with a
    # fixed seed. Taking the first N is biased: several source test files are
    # class-ordered, which degenerates AUROC/accuracy on the capped subset.
    recs = []
    for split in ("test", "dev"):
        f = task_dir / f"{split}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line); r["_split"] = split; recs.append(r)
        if recs:
            break  # use test if present, else dev; do not mix
    if not recs:
        return None
    rng = np.random.default_rng(seed)
    if len(recs) > maxn:
        idx = rng.choice(len(recs), size=maxn, replace=False)
        recs = [recs[i] for i in sorted(idx)]
    if not recs:
        return None
    # per-task anchors from the first record carrying them
    anchors = recs[0].get("anchors") if recs[0]["primitive"] == "score" else None
    rows, skipped = [], defaultdict(int)
    for rec in recs:
        row, why = to_row(rec, anchors)
        if why:
            skipped[why] += 1; continue
        try:
            row = rep.prepare(row)
        except Exception as e:
            skipped["encode_fail:" + type(e).__name__] += 1; continue
        if row["length"] > MAXLEN:
            skipped["over_budget"] += 1; continue
        if rec["primitive"] == "score":
            row["_gold_value"] = rec["gold_value"]
            # _anchors already set to the 5 derived values inside to_row
        row["_split"] = rec["_split"]; row["_primitive"] = rec["primitive"]
        rows.append(row)
    if not rows:
        return {"skip_reasons": dict(skipped), "note": "no supported rows"}
    # run in micro batches, score-decode with per-task anchors
    out_probs, out_labels, out_goldv, out_preds = [], [], [], []
    micro = 16
    for start in range(0, len(rows), micro):
        rr = [t.render(r) for r in rows[start:start + micro]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**t.pack(rr, pad))
        for row, logit in zip(rr, logits):  # rendered rows carry 'order' and _primitive/_gold_value/_anchors
            k = len(row["choices"])
            p = logit[:k].float().softmax(-1).cpu().numpy()
            canon = np.zeros(k)
            for j, idx in enumerate(row["order"]):
                canon[idx] = p[j]
            out_probs.append(canon); out_labels.append(row["label"])
            if row["_primitive"] == "score":
                val = float(np.dot(canon, row["_anchors"]))
                out_goldv.append(row["_gold_value"]); out_preds.append(val)
    y = np.array(out_labels); P = np.array(out_probs); pred = P.argmax(-1)
    res = {"skip_reasons": dict(skipped), "n": len(rows), "primitive": rows[0]["_primitive"]}
    if res["primitive"] in ("noul", "choice"):
        acc = float((pred == y).mean())
        res["accuracy"] = acc
        res["nll"] = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-30, 1)).mean())
        classes = sorted(set(y.tolist()) | set(pred.tolist()))
        f1s = []
        for c in classes:
            tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
            prec = tp / (tp + fp) if tp + fp else 0; rec_ = tp / (tp + fn) if tp + fn else 0
            f1s.append(2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0)
        res["macro_f1"] = float(np.mean(f1s)) if f1s else None
        if res["primitive"] == "noul" and len(set(y.tolist())) > 1:
            # AUROC using prob of positive class
            from sklearn.metrics import roc_auc_score
            res["auroc"] = float(roc_auc_score(y, P[:, -1])) if P.shape[1] >= 2 else None
    elif res["primitive"] == "score":
        g = np.array(out_goldv); pv = np.array(out_preds)
        res["rmse"] = float(np.sqrt(((pv - g) ** 2).mean()))
        res["mae"] = float(np.abs(pv - g).mean())
        res["prediction_std"] = float(pv.std())
        try:
            from scipy.stats import spearmanr
            rho = spearmanr(g, pv).statistic; res["spearman"] = float(rho) if np.isfinite(rho) else None
        except Exception:
            res["spearman"] = None
        res["gold_std"] = float(g.std())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None, help="trained model.safetensors; default = recommended recipe")
    ap.add_argument("--tasks", nargs="*", default=None, help="task_ids to eval; default = all supported dirs")
    ap.add_argument("--max-per-task", type=int, default=200)
    ap.add_argument("--out", type=Path, default=JEV / "artifacts/benchmark_v2_eval")
    a = ap.parse_args()
    ckpt = a.ckpt or next(iter(sorted((JEV / "artifacts/laya_jev_joint_path_diagnostics/run_gfp_loss2x_xseed/round").glob("*/model.safetensors"))), None)
    if ckpt is None or not Path(ckpt).exists():
        print("no checkpoint found"); return 1
    print(f"[harness] checkpoint = {ckpt}", flush=True)
    model = load_model(ckpt)
    rep = Representation(CPT / "data")
    from transformers import AutoTokenizer
    pad = AutoTokenizer.from_pretrained(CPT / "data/representation/base_tokenizer").pad_token_id

    all_tasks = sorted(p for p in OUT.iterdir() if p.is_dir())
    if a.tasks:
        all_tasks = [p for p in all_tasks if p.name in set(a.tasks)]
    results = {}
    for td in all_tasks:
        try:
            r = run_task(model, rep, td, a.max_per_task, None, pad)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        if r is not None:
            results[td.name] = r
        tag = r or {}
        line = f"  {td.name:34s}"
        if "primitive" in tag:
            line += f" {tag['primitive']:6s} n={tag.get('n',0):4d}"
            for k in ("accuracy","macro_f1","auroc","rmse","mae","spearman"):
                if k in tag and tag[k] is not None:
                    line += f" {k}={tag[k]:.3f}"
        elif "error" in tag:
            line += f" ERR {tag['error'][:40]}"
        else:
            line += f" SKIP {tag}"
        print(line, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "eval_results.json").write_text(json.dumps({"checkpoint": str(ckpt), "max_per_task": a.max_per_task,
                                                          "results": results}, ensure_ascii=False, indent=2) + "\n")
    # aggregate by primitive
    byprim = defaultdict(list)
    for name, r in results.items():
        if "primitive" in r:
            byprim[r["primitive"]].append(r)
    summary = {}
    for prim, rs in byprim.items():
        m = {"n_tasks": len(rs)}
        for k in ("accuracy","macro_f1","auroc","rmse","mae","spearman"):
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals: m["mean_"+k] = float(np.mean(vals))
        summary[prim] = m
    print("\n[aggregate by primitive]"); print(json.dumps(summary, indent=2))
    (a.out / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
