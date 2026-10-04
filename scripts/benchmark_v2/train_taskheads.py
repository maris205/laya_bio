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


def lr_factor(step, total, warmup_frac=0.05):
    warmup = max(1, int(warmup_frac * total))
    if step <= warmup: return step / warmup
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def make_optimizer(model):
    enc, head = [], []
    for name, p in model.named_parameters():
        (enc if name.startswith("encoder.") else head).append(p)
    return torch.optim.AdamW([{"params": enc, "lr": LR_ENC, "base_lr": LR_ENC, "weight_decay": .01},
                              {"params": head, "lr": LR_HEAD, "base_lr": LR_HEAD, "weight_decay": .01}])


def eval_subset(model, tasks, split, maxn, rep, pad, want=("accuracy","macro_f1","auroc","spearman")):
    """Evaluate the head model on `split` for the given tasks -> {task: metrics}.
    Reused for dev early-stopping and final test eval (per-family checkpoints)."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    results = {}
    model.eval()
    for tk in tasks:
        f = OUT / tk / f"{split}.jsonl"
        if not f.exists():
            f2 = OUT / tk / ("dev.jsonl" if split == "test" else "test.jsonl")
            f = f2 if f2.exists() else f
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs: continue
        prim = recs[0]["primitive"]
        rows = []
        for rec in recs:
            seqs = rec["sequences"]
            if len(seqs) != 1 or seqs[0]["modality"] not in ("dna", "protein"): continue
            s = seqs[0]["sequence"]; m = seqs[0]["modality"]
            if prim == "noul":
                choices = ["false: "+rec["candidates"][0], "true: "+rec["candidates"][1]]; label = 1 if rec["answer"] == "yes" else 0
            elif prim == "choice":
                choices = list(rec["candidates"])
                if rec["answer"] not in choices: continue
                label = choices.index(rec["answer"])
            else:
                choices = [f"value = {x:.6g}" for x in (rec.get("anchors") or [])]; label = int(rec["gold_level"])
            row = {"id": tk, "primitive": prim, "modality": m, "question": rec["question"], "sequence": s, "choices": choices, "label": label}
            if prim == "score": row["value"] = float(rec["gold_value"])
            try: row = rep.prepare(row)
            except Exception: continue
            if row["length"] > MAXLEN: continue
            rows.append(row)
        if len(rows) < 1: continue
        if len(rows) > maxn:
            rows = [rows[i] for i in np.random.default_rng(20261001).choice(len(rows), size=maxn, replace=False)]
        preds=[]; ys=[]; gold=[]; probs=[]
        with torch.no_grad():
            for st in range(0, len(rows), BATCH):
                b = rows[st:st+BATCH]
                with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(**pack2(b, pad), task=tk)
                if prim == "score":
                    preds += logits.squeeze(-1).float().cpu().tolist(); gold += [r["value"] for r in b]
                else:
                    lg = logits.float(); pv = F.softmax(lg, -1).cpu().tolist(); probs += pv
                    preds += lg.argmax(-1).cpu().tolist(); ys += [r["label"] for r in b]
        if prim == "score":
            pv = np.array(preds); g = np.array(gold)
            if len(g) < 3: continue
            rho = spearmanr(g, pv).statistic
            results[tk] = {"primitive": "score", "n": len(rows), "spearman": float(rho) if np.isfinite(rho) else None,
                           "rmse": float(np.sqrt(((pv-g)**2).mean())), "mae": float(np.abs(pv-g).mean()),
                           "gold_std": float(g.std()), "prediction_std": float(pv.std())}
        else:
            pred = np.array(preds); y = np.array(ys); P = np.array(probs)
            acc = float((pred==y).mean())
            cls = sorted(set(y.tolist())|set(pred.tolist())); f1s = []
            for c in cls:
                tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum()
                pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
                f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
            r = {"primitive": prim, "n": len(rows), "accuracy": acc, "macro_f1": float(np.mean(f1s))}
            if prim == "noul" and len(set(y.tolist()))>1 and P.shape[1]>=2:
                try: r["auroc"] = float(roc_auc_score(y, P[:,-1]))
                except Exception: pass
            results[tk] = r
    model.train()
    return results


def dev_metric(res):
    """map a per-task eval result to a [0,1] scalar for early stopping."""
    import numpy as _np
    vals = []
    for tk, v in res.items():
        if v.get("primitive") == "score":
            s = v.get("spearman"); vals.append((s+1)/2 if s is not None else 0.5)
        elif v.get("primitive") == "noul":
            vals.append(v.get("auroc") if v.get("auroc") is not None else v.get("accuracy", 0))
        else:
            vals.append(v.get("accuracy", 0))
    return float(_np.mean(vals)) if vals else 0.0, {tk: ( (v.get("spearman")+1)/2 if v.get("primitive")=="score" and v.get("spearman") is not None else (v.get("auroc") if v.get("primitive")=="noul" and v.get("auroc") is not None else v.get("accuracy",0)) ) for tk, v in res.items()}


@torch.no_grad()
def _dev_eval_heads(model, devrows, pad, micro=32):
    """Eval prepared dev rows through the head model -> {task: {primitive, accuracy|spearman|auroc}}."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    model.eval()
    out = {}
    for tk, rows in devrows.items():
        if not rows: continue
        prim = rows[0]["primitive"]
        ys, preds, probs, gold, est = [], [], [], [], []
        for st in range(0, len(rows), micro):
            b = rows[st:st+micro]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**pack2(b, pad), task=tk)
            if prim == "score":
                est += logits.squeeze(-1).float().cpu().tolist(); gold += [r["value"] for r in b]
            else:
                lg = logits.float(); pv = F.softmax(lg, -1).cpu().tolist()
                probs += pv; preds += lg.argmax(-1).cpu().tolist(); ys += [r["label"] for r in b]
        if prim == "score":
            if len(gold) < 3: continue
            rho = spearmanr(np.array(gold), np.array(est)).statistic
            out[tk] = {"primitive": "score", "spearman": float(rho) if np.isfinite(rho) else None}
        else:
            y = np.array(ys); p = np.array(preds); P = np.array(probs)
            r = {"primitive": prim, "accuracy": float((p == y).mean())}
            if prim == "noul" and len(set(y.tolist())) > 1 and P.shape[1] >= 2:
                try: r["auroc"] = float(roc_auc_score(y, P[:, -1]))
                except Exception: pass
            out[tk] = r
    model.train()
    return out


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
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dev-eval-every", type=int, default=0)
    ap.add_argument("--dev-max", type=int, default=96)
    ap.add_argument("--per-family-early-stop", action="store_true")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--name", default="bv2_taskheads")
    ap.add_argument("--out", type=Path, default=JEV/"artifacts/benchmark_v2_train")
    a = ap.parse_args()
    global SEED; SEED = a.seed
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
    # dev rows for early stopping (prepared once)
    devrows = {}
    if a.dev_eval_every > 0:
        for tk in bytask:
            drows, _ = load_task_rows(rep, tk, a.dev_max, split="dev")
            devrows[tk] = drows
    fam_best = {}; fam_best_step = {}; best_dev = -1.0; best_step = 0; dev_curve = []
    step = 0; trace = []; t0 = time.monotonic()
    for epoch, tk, rr in schedule_family(bytask, a.epochs, BATCH, a.batches_per_family, rng):
        step += 1
        for g in opt.param_groups: g["lr"] = g["base_lr"] * lr_factor(step, total, a.warmup_frac)
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
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip, error_if_nonfinite=False))
        opt.step()
        trace.append({"step": step, "epoch": epoch, "task": tk, "loss": float(loss)/len(rr), "grad": grad})
        if a.dev_eval_every > 0 and step % a.dev_eval_every == 0:
            # dev eval on the union of dev rows via eval_subset-style (reuse prepared devrows)
            dres = _dev_eval_heads(model, devrows, pad)
            gscore, per_task = dev_metric(dres)
            fscores = defaultdict(list)
            for tkk, mv in per_task.items(): fscores[family_of(tkk)].append(mv)
            fscores = {f: float(np.mean(v)) for f, v in fscores.items()}
            dev_curve.append({"step": step, "dev_score": gscore, "per_family": fscores})
            if gscore > best_dev:
                best_dev = gscore; best_step = step
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out/"best.safetensors"))
            if a.per_family_early_stop:
                for f, sc in fscores.items():
                    if sc > fam_best.get(f, -1.0):
                        fam_best[f] = sc; fam_best_step[f] = step
                        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out/f"best_{f}.safetensors"))
            print(json.dumps({"dev_eval": step, "dev_score": round(gscore,4), "best_step": best_step,
                              "fam_best": {f: round(s,3) for f,s in fam_best.items()}}), flush=True)
        if step % 40 == 0 or step == 1:
            print(json.dumps({**trace[-1], "sec": round(time.monotonic()-t0,1)}), flush=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out/"model.safetensors"))
    (out/"train_trace.jsonl").write_text("\n".join(json.dumps(r) for r in trace)+"\n")
    # promote global best-dev if early stopping ran
    if a.dev_eval_every > 0 and best_step > 0 and (out/"best.safetensors").exists():
        import shutil
        shutil.copy2(out/"model.safetensors", out/"model_final.safetensors")
        shutil.copy2(out/"best.safetensors", out/"model.safetensors")
    if dev_curve:
        (out/"dev_curve.jsonl").write_text("\n".join(json.dumps(r) for r in dev_curve)+"\n")

    # ---- final test eval ----
    results = {}
    fam_of = {tk: family_of(tk) for tk in bytask}
    if a.per_family_early_stop:
        # eval each family's tasks with that family's own best checkpoint
        for f in sorted({fam_of[t] for t in bytask}):
            ck = out/f"best_{f}.safetensors"
            ftasks = [t for t in bytask if fam_of[t] == f]
            if ck.exists():
                model.load_state_dict(load_file(str(ck)), strict=True)
            res = eval_subset(model, ftasks, "test", a.eval_max, rep, pad)
            results.update(res)
    else:
        results = eval_subset(model, list(bytask), "test", a.eval_max, rep, pad)
    (out/"eval_results.json").write_text(json.dumps({"name":a.name,"matched_to":str(a.task_config),
        "per_family_early_stop":a.per_family_early_stop,"results":results},ensure_ascii=False,indent=2)+"\n")
    (out/"run_config.json").write_text(json.dumps({"tasks":list(bytask),"nout":nout,"epochs":a.epochs,
        "batches_per_family":a.batches_per_family,"total_updates":step,"warmup_frac":a.warmup_frac,"clip":a.clip,
        "dev_eval_every":a.dev_eval_every,"best_step":best_step,"best_dev":best_dev,
        "per_family_early_stop":a.per_family_early_stop,"fam_best_dev":fam_best,"fam_best_step":fam_best_step,
        "elapsed_sec":round(time.monotonic()-t0,1)},indent=2)+"\n")
    print(f"[#2 DONE] trained {step} updates in {round(time.monotonic()-t0,1)}s; eval saved (per_family={a.per_family_early_stop})", flush=True)
    return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
