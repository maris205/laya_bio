#!/usr/bin/env python3
"""#2 baseline: same Laya encoder + per-task heads (matched to #1 shared scorer).

Isolates the architecture variable vs the benchmark v2 shared typed-decision
scorer (#1): identical backbone, data, family-balanced schedule, update count,
seed, and eval subsets; the ONLY difference is the decision mechanism —
#1 scores dynamic candidate markers with one shared scorer; #2 uses a per-task
output head (softmax over each task's fixed class set for noul/choice, a scalar
regression head for score) and does not consume candidate text as scored
markers.

Runs the same selected tasks as a #1 run (default: bv2_full2 task list), trains,
then evaluates on the same test/dev subsets and writes per-task metrics.
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

JEV = Path("/root/autodl-tmp/jev_gene")
FROZEN = JEV / "artifacts/laya_jev_multitask_v1/round/frozen_code"
MODEL = JEV / "artifacts/laya_model"
CPT = JEV / "artifacts/laya_biocpt_v2"
OUT = JEV / "data/06_benchmark_v2_unified"
sys.path.insert(0, str(FROZEN))

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya_jev_multitask_data import Representation, TYPE_IDS
import laya_biocpt_train as bio
from train_benchmark_v2 import (family_of, derive_anchors, discover_supported, schedule_family,
                                LR_ENC, LR_HEAD, SEED, BATCH, MAXLEN)

# reuse the SAME prepared-row builder from #1's trainer for identical data
from train_benchmark_v2 import load_task_rows  # returns (rows with head_ids/option_ids/body_ids/label/...)


def enc_ids(row):
    # #2 input = question tokens (head) + sequence tokens (body); no candidate markers.
    return row["head_ids"] + row["body_ids"]


class TaskHeadModel(nn.Module):
    def __init__(self, encoder, hidden, task_nout):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, n) for t, n in task_nout.items()})
    def forward(self, input_ids, attention_mask, task):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.heads[task](pooled)


def pack2(rows, pad, device="cuda"):
    width = max(len(enc_ids(r)) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    att = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        e = enc_ids(r)
        ids[i, :len(e)] = torch.tensor(e); att[i, :len(e)] = 1
    return {"input_ids": ids.to(device), "attention_mask": att.to(device)}


def lr_factor(step, total):
    warmup = max(1, int(.05 * total))
    if step <= warmup: return step / warmup
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def make_optimizer(model):
    enc, head = [], []
    for name, p in model.named_parameters():
        (enc if name.startswith("encoder.") else head).append(p)
    return torch.optim.AdamW([{"params": enc, "lr": LR_ENC, "base_lr": LR_ENC, "weight_decay": .01},
                              {"params": head, "lr": LR_HEAD, "base_lr": LR_HEAD, "weight_decay": .01}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--task-config", type=Path, default=JEV/"artifacts/benchmark_v2_train/bv2_full2/run_config.json",
                    help="reuse a #1 run's task list for a matched comparison")
    ap.add_argument("--max-per-task", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batches-per-family", type=int, default=30)
    ap.add_argument("--min-valid", type=int, default=128)
    ap.add_argument("--eval-max", type=int, default=200)
    ap.add_argument("--name", default="bv2_taskheads")
    ap.add_argument("--out", type=Path, default=JEV/"artifacts/benchmark_v2_train")
    a = ap.parse_args()
    if a.tasks:
        tasks = list(a.tasks)
    else:
        tasks = json.load(open(a.task_config))["tasks"]
    torch.set_num_threads(8)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    out = a.out / a.name; out.mkdir(parents=True, exist_ok=True)
    rep = Representation(CPT / "data")
    pad = AutoTokenizer.from_pretrained(CPT / "data/representation/base_tokenizer").pad_token_id
    mlm, _, _ = bio.build(MODEL, CPT / "data", None)
    encoder = mlm.model
    hidden = encoder.config.hidden_size
    # lm_head (MLM decoder) is unused by #2; TaskHeadModel only wraps encoder + heads.
    encoder.to("cuda")

    bytask = {}; specs = {}; nout = {}; prep_fail = Counter()
    for tk in tasks:
        rows, spec = load_task_rows(rep, tk, a.max_per_task)
        if len(rows) < max(BATCH, a.min_valid):
            prep_fail[tk] = len(rows); continue
        prim = rows[0]["primitive"]
        nout[tk] = 1 if prim == "score" else len(rows[0]["choices"])
        bytask[tk] = rows; specs[tk] = spec
    nfam = len({family_of(x) for x in bytask})
    total = a.epochs * a.batches_per_family * nfam
    print(f"[#2] tasks={len(bytask)} families={nfam} updates≈{total} gated={len(prep_fail)}", flush=True)
    model = TaskHeadModel(encoder, hidden, nout).to("cuda")
    rng = random.Random(SEED + 41)
    opt = make_optimizer(model)
    step = 0; trace = []; t0 = time.monotonic()
    for epoch, tk, rr in schedule_family(bytask, a.epochs, BATCH, a.batches_per_family, rng):
        step += 1
        for g in opt.param_groups: g["lr"] = g["base_lr"] * lr_factor(step, total)
        model.train(); opt.zero_grad(set_to_none=True)
        labels = torch.tensor([r["label"] for r in rr], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**pack2(rr, pad), task=tk)
        prim = rr[0]["primitive"]
        if prim == "score":
            pred = logits.squeeze(-1).float()
            true = torch.tensor([r["value"] for r in rr], device="cuda")
            std = (specs[tk] or {}).get("train_std", float(true.std())) or 1.0
            loss = ((pred - true) / std).square().mean() * len(rr)
        else:
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        (loss / len(rr)).backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False))
        opt.step()
        trace.append({"step": step, "epoch": epoch, "task": tk, "loss": float(loss)/len(rr), "grad": grad})
        if step % 40 == 0 or step == 1:
            print(json.dumps({**trace[-1], "sec": round(time.monotonic()-t0,1)}), flush=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out/"model.safetensors"))
    (out/"train_trace.jsonl").write_text("\n".join(json.dumps(r) for r in trace)+"\n")

    # ---- eval on same test/dev subsets as #1 (seeded random) ----
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    results = {}
    evrng = np.random.default_rng(20261001)
    for tk in bytask:
        d = OUT / tk
        recs = []
        for split in ("test","dev"):
            f = d / f"{split}.jsonl"
            if f.exists():
                for line in f.read_text().splitlines():
                    if line.strip(): recs.append(json.loads(line))
                if recs: break
        if not recs: continue
        prim = recs[0]["primitive"]
        rows = []
        for rec in recs:
            seqs = rec["sequences"]
            if len(seqs)!=1 or seqs[0]["modality"] not in ("dna","protein"): continue
            s = seqs[0]["sequence"]; m = seqs[0]["modality"]
            if prim=="noul":
                choices=["false: "+rec["candidates"][0],"true: "+rec["candidates"][1]]; label=1 if rec["answer"]=="yes" else 0
            elif prim=="choice":
                choices=list(rec["candidates"]);
                if rec["answer"] not in choices: continue
                label=choices.index(rec["answer"])
            else:
                choices=[f"value = {x:.6g}" for x in (rec.get('anchors') or [])]
                label=int(rec["gold_level"])
            row={"id":tk,"primitive":prim,"modality":m,"question":rec["question"],"sequence":s,"choices":choices,"label":label}
            if prim=="score": row["value"]=float(rec["gold_value"])
            try: row=rep.prepare(row)
            except Exception: continue
            if row["length"]>MAXLEN: continue
            rows.append(row)
        if len(rows)<1: continue
        # sample max eval-max for parity with #1
        if len(rows)>a.eval_max:
            rows=[rows[i] for i in np.random.default_rng(20261001).choice(len(rows),size=a.eval_max,replace=False)]
        model.eval(); preds=[]; ys=[]; gold=[]; probs=[]
        with torch.no_grad():
            for st in range(0,len(rows),BATCH):
                b=rows[st:st+BATCH]
                with torch.autocast("cuda",dtype=torch.bfloat16): logits=model(**pack2(b,pad),task=tk)
                if prim=="score":
                    preds+=logits.squeeze(-1).float().cpu().tolist(); gold+=[r["value"] for r in b]
                else:
                    lg=logits.float(); pv=F.softmax(lg,-1).cpu().tolist(); probs+=pv
                    preds+=lg.argmax(-1).cpu().tolist(); ys+=[r["label"] for r in b]
        if prim=="score":
            pv=np.array(preds); g=np.array(gold)
            rho=spearmanr(g,pv).statistic
            results[tk]={"primitive":"score","n":len(rows),"rmse":float(np.sqrt(((pv-g)**2).mean())),
                "mae":float(np.abs(pv-g).mean()),"spearman":float(rho) if np.isfinite(rho) else None,
                "gold_std":float(g.std()),"prediction_std":float(pv.std())}
        else:
            pred=np.array(preds); y=np.array(ys); P=np.array(probs)
            acc=float((pred==y).mean())
            cls=sorted(set(y.tolist())|set(pred.tolist())); f1s=[]
            for c in cls:
                tp=((pred==c)&(y==c)).sum();fp=((pred==c)&(y!=c)).sum();fn=((pred!=c)&(y==c)).sum()
                pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
                f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
            r={"primitive":prim,"n":len(rows),"accuracy":acc,"macro_f1":float(np.mean(f1s))}
            if prim=="noul" and len(set(y.tolist()))>1 and P.shape[1]>=2:
                try: r["auroc"]=float(roc_auc_score(y,P[:,-1]))
                except Exception: pass
            results[tk]=r
    (out/"eval_results.json").write_text(json.dumps({"name":a.name,"matched_to":str(a.task_config),"results":results},ensure_ascii=False,indent=2)+"\n")
    (out/"run_config.json").write_text(json.dumps({"tasks":list(bytask),"nout":nout,"epochs":a.epochs,
        "batches_per_family":a.batches_per_family,"total_updates":step,"elapsed_sec":round(time.monotonic()-t0,1)},indent=2)+"\n")
    print(f"[#2 DONE] trained {step} updates in {round(time.monotonic()-t0,1)}s; eval saved", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
