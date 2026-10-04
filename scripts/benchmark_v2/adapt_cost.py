#!/usr/bin/env python3
"""Adaptation-cost comparison: #1 zero-shot (0 labels) vs #2 new-head few-shot.

#1 (shared typed-decision scorer) handles a NEW task with zero task-specific
parameters or training — feed the question+candidates. #2 (per-task heads)
structurally cannot: it must add a new head and train it on that task's labeled
data. This quantifies #2's adaptation cost via a linear probe: freeze the #2
encoder (trained on the 37 seen tasks), precompute pooled features once, train a
new head on N labeled examples of a held-out task, and compare to #1's zero-shot
number on the same task. How many labels does #2 need to match #1's zero-shot?
"""
from __future__ import annotations
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

JEV = Path("/root/autodl-tmp/jev_gene")
FROZEN = JEV/"artifacts/laya_jev_multitask_v1/round/frozen_code"
MODEL = JEV/"artifacts/laya_model"; CPT = JEV/"artifacts/laya_biocpt_v2"
sys.path.insert(0, str(FROZEN)); sys.path.insert(0, str(JEV/"scripts/benchmark_v2"))
from laya_jev_multitask_data import Representation
import laya_biocpt_train as bio
from train_taskheads import pack2
from train_benchmark_v2 import load_task_rows, BATCH
from transformers import AutoTokenizer
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--ns", nargs="*", type=int, default=[32, 128, 512])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--enc-ckpt", type=Path, default=JEV/"artifacts/benchmark_v2_train/bv2_taskheads_pf/model.safetensors")
    ap.add_argument("--zs1", type=Path, default=JEV/"artifacts/benchmark_v2_eval/zs_trained/eval_results.json")
    ap.add_argument("--out", type=Path, default=JEV/"artifacts/benchmark_v2_eval/adapt_cost.json")
    a = ap.parse_args()
    tasks = a.tasks or json.load(open("/tmp/heldout_sel.json"))
    torch.manual_seed(20261001); np.random.seed(20261001); random.seed(20261001)
    rep = Representation(CPT/"data")
    pad = AutoTokenizer.from_pretrained(CPT/"data/representation/base_tokenizer").pad_token_id
    mlm, _, _ = bio.build(MODEL, CPT/"data", None)
    encoder = mlm.model; hidden = encoder.config.hidden_size
    st = load_file(str(a.enc_ckpt))
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in st.items() if k.startswith("encoder.")}, strict=True)
    encoder.to("cuda").eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    zs1 = json.load(open(a.zs1))["results"]

    @torch.no_grad()
    def feats_of(rows):
        outs = []
        for s in range(0, len(rows), BATCH):
            pk = pack2(rows[s:s+BATCH], pad)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = encoder(**pk).last_hidden_state
            am = pk["attention_mask"].unsqueeze(-1).to(o.dtype)
            outs.append(((o*am).sum(1)/am.sum(1).clamp(min=1)).float())
        return torch.cat(outs, 0)

    results = {}
    for tk in tasks:
        tr_rows, spec = load_task_rows(rep, tk, max(a.ns), split="train")
        te_rows, _ = load_task_rows(rep, tk, 200, split="test")
        if len(te_rows) < 8 or not tr_rows: continue
        prim = te_rows[0]["primitive"]
        nout = 1 if prim == "score" else len(te_rows[0]["choices"])
        zkey = "spearman" if prim == "score" else ("auroc" if prim == "noul" else "accuracy")
        zval = zs1.get(tk, {}).get(zkey) if prim != "noul" else (zs1.get(tk, {}).get("auroc", zs1.get(tk, {}).get("accuracy")))
        trF = feats_of(tr_rows); teF = feats_of(te_rows)
        tr_lab = torch.tensor([r["label"] for r in tr_rows], device="cuda")
        tr_val = torch.tensor([r.get("value", r["label"]) for r in tr_rows], device="cuda", dtype=torch.float)
        te_lab = [r["label"] for r in te_rows]; te_val = [r.get("value", r["label"]) for r in te_rows]
        std = (spec or {}).get("train_std", float(tr_val.std())) or 1.0
        per_n = {}
        for N in a.ns:
            if N > len(tr_rows): continue
            head = torch.nn.Linear(hidden, nout).to("cuda")
            opt = torch.optim.Adam(head.parameters(), lr=1e-3)
            Fs, ls, vs = trF[:N], tr_lab[:N], tr_val[:N]
            for _ in range(a.steps):
                idx = torch.randint(0, N, (min(BATCH, N),), device="cuda")
                lg = head(Fs[idx])
                loss = ((lg.squeeze(-1)-vs[idx])/std).square().mean() if prim == "score" else F.cross_entropy(lg, ls[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad(): lg = head(teF)
            if prim == "score":
                e = lg.squeeze(-1).cpu().tolist(); g = np.array(te_val)
                rho = spearmanr(g, np.array(e)).statistic if len(g) > 2 else None
                per_n[N] = round(float(rho), 3) if rho is not None and np.isfinite(rho) else None
            else:
                pv = F.softmax(lg, -1).cpu().numpy(); pred = lg.argmax(-1).cpu().tolist()
                if prim == "noul" and len(set(te_lab)) > 1:
                    per_n[N] = round(float(roc_auc_score(te_lab, pv[:, -1])), 3)
                else:
                    per_n[N] = round(float((np.array(pred) == np.array(te_lab)).mean()), 3)
        results[tk] = {"primitive": prim, "metric": zkey if prim != "noul" else "auroc", "zero_shot_1": zval, "head_fewshot": per_n}
        print(f"  {tk:32s} {prim:6s} #1零样本={zval} #2新头(N)={per_n}", flush=True)
    a.out.write_text(json.dumps({"enc_ckpt": str(a.enc_ckpt), "steps": a.steps, "ns": a.ns, "results": results}, ensure_ascii=False, indent=2)+"\n")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
