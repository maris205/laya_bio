#!/usr/bin/env python3
"""Paper-2 trainer/evaluator: typed-decision arms on causal LM backbones (Qwen3).

Arm "shared" (#1Q): question + sequence + each candidate rendered as text with a
trailing [MARK] token; a shared scalar scorer reads the [MARK] hidden state; softmax
over candidates -> CE (score tasks: 5 anchor candidates, expectation vs standardized
gold adds an MSE term).
Arm "heads"  (#2Q): question + sequence only; pooled last hidden -> per-task Linear
(softmax CE for choice/noul; scalar MSE on standardized gold for score).

Protocol mirrors Paper 1: family-balanced interleave, warmup .15, clip .5, dev early
stop (dev>=256), per-family best checkpoints, 512-token full inputs, seed averaging
done by the aggregator outside this script. LoRA (peft) bf16; --qlora falls back to
4-bit if VRAM is tight.
"""
from __future__ import annotations
import argparse, json, math, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from safetensors.torch import save_file, load_file

JEV = Path("/root/autodl-tmp/jev_gene")
UNI = JEV / "data/06_benchmark_v2_unified"
MARK = "[MARK]"
MAXLEN = 512
BATCH = 64

PREFIX_FAMILY = [("pg_", "ProteinGym"), ("rnac_", "RNAcompete"), ("gue_", "GUE"),
                 ("gb_", "GenomicBenchmarks"), ("gl_", "gene_lan"), ("dna_", "dnagpt_pools"),
                 ("deepstarr", "DeepSTARR"), ("tape_", "TAPE"), ("lg_", "local_snapshots"),
                 ("protein_homology", "local_snapshots"), ("deeploc", "DeepLoc")]

def family_of(tid):
    for pre, f in PREFIX_FAMILY:
        if tid.startswith(pre):
            return f
    return "other"

# ---------------- data ----------------
def derive_anchors(levels, train_vals):
    """5 representative anchor values from 4 interior train quantile cuts."""
    qs = np.quantile(train_vals, [0.1, 0.3, 0.5, 0.7, 0.9])
    return [float(x) for x in qs]

def load_task(rec_dir, maxn, split="train", rng=None):
    rows = [json.loads(l) for l in (rec_dir / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    if rng: rng.shuffle(rows)
    rows = rows[:maxn]
    out = []
    prim = rows[0]["primitive"]
    for r in rows:
        seqs = " ".join(s["sequence"] for s in r["sequences"])
        q = r["question"]
        if prim == "score":
            vals = r.get("gold_value", r.get("value"))
            out.append({"prim": prim, "seq": seqs, "q": q, "gold": float(vals),
                        "levels": r.get("score_levels")})
        else:
            cands = r["candidates"]
            out.append({"prim": prim, "seq": seqs, "q": q, "cands": cands,
                        "y": cands.index(r["answer"]) if r["answer"] in cands else None})
    return [r for r in out if r.get("y") is not None or r["prim"] == "score"]

def task_stats(rows):
    if rows and rows[0]["prim"] == "score":
        g = np.array([r["gold"] for r in rows])
        return {"mean": float(g.mean()), "std": float(g.std() + 1e-8),
                "anchors": derive_anchors(None, g)}
    return None

# ---------------- model ----------------
def build_model(model_dir, qlora):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_dir)
    if MARK not in tok.get_vocab():
        tok.add_tokens([MARK])
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if qlora:
        from transformers import BitsAndBytesConfig
        kw.update(torch_dtype=torch.bfloat16,
                  quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                      bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
    model = AutoModel.from_pretrained(model_dir, **kw)  # no lm_head: hidden states only
    model.resize_token_embeddings(len(tok))
    if not qlora:
        model.gradient_checkpointing_enable()
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.FEATURE_EXTRACTION,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return tok, model

class Scorer(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.scorer = torch.nn.Sequential(torch.nn.Linear(hidden, hidden // 2),
                                          torch.nn.GELU(), torch.nn.Linear(hidden // 2, 1))
    def forward(self, h):
        return self.scorer(h).squeeze(-1)

# ---------------- rendering ----------------
def render_shared(row, stats):
    """one text per candidate; returns list of (text, mark_pos_target)."""
    if row["prim"] == "score":
        cands = [f"value = {a:.2f}" for a in stats["anchors"]]
    else:
        cands = row["cands"]
    base = f"{row['q']}\nSequence: {row['seq']}\nAnswer:"
    return [f"{base} {c} {MARK}" for c in cands], cands

def render_heads(row):
    return f"{row['q']}\nSequence: {row['seq']}"

# ---------------- batches ----------------
def pack(tok, texts, maxlen=MAXLEN):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
    return enc

# ---------------- eval ----------------
@torch.no_grad()
def eval_tasks(model, tok, heads, scorer, tasks, arch, maxn=200):
    model.eval()
    res = {}
    dev = next(model.parameters()).device
    mark_id = tok.convert_tokens_to_ids(MARK)
    for tid, (rows, stats) in tasks.items():
        rows = rows[:maxn]
        prim = rows[0]["prim"]
        if arch == "shared":
            scores = []
            for r in rows:
                ts, cands = render_shared(r, stats)
                enc = pack(tok, ts)
                enc = {k: v.to(dev) for k, v in enc.items()}
                h = model(**enc, use_cache=False).last_hidden_state
                pos = (enc["input_ids"] == mark_id).float().argmax(dim=1)
                logits = scorer(h.gather(1, pos[:, None, None].expand(-1, 1, h.shape[-1])).squeeze(1))
                logits = logits.view(len(ts) // len(cands), len(cands)) if len(cands) > 1 else logits
                scores.append(logits)
            S = torch.cat(scores) if prim != "score" else torch.cat(scores)
            if prim == "score":
                av = torch.tensor(stats["anchors"], device=dev)
                pred = (torch.softmax(S.float(), -1) * av).sum(-1)
                gold = torch.tensor([r["gold"] for r in rows], device=dev)
                from scipy.stats import spearmanr
                rho = spearmanr(pred.cpu().numpy(), gold.cpu().numpy()).statistic
                res[tid] = {"primitive": prim, "n": len(rows), "spearman": float(rho)}
            else:
                y = torch.tensor([r["y"] for r in rows], device=dev)
                p = torch.softmax(S.float(), -1)
                acc = float((p.argmax(-1) == y).float().mean())
                if p.shape[1] == 2:
                    yi = 1 if cands[1].lower().startswith(("yes", "true")) else 0
                    pn = p[:, yi]
                else:
                    pn = p.gather(1, y[:, None]).squeeze()
                from sklearn.metrics import roc_auc_score
                try:
                    auroc = float(roc_auc_score((y == yi).cpu().numpy(), pn.cpu().numpy())) if p.shape[1] == 2 else None
                except Exception:
                    auroc = None
                res[tid] = {"primitive": prim, "n": len(rows), "accuracy": acc, "auroc": auroc}
        else:
            logits = []
            for i in range(0, len(rows), 32):
                chunk = rows[i:i+32]
                enc = pack(tok, [render_heads(r) for r in chunk])
                enc = {k: v.to(dev) for k, v in enc.items()}
                out = model(**enc, use_cache=False).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (out * mask).sum(1) / mask.sum(1)
                logits.append(heads[tid](pooled))
            logits = torch.cat(logits)
            if prim == "score":
                pred = logits.squeeze(-1).float()
                gold = torch.tensor([(r["gold"] - stats["mean"]) / stats["std"] for r in rows], device=dev)
                from scipy.stats import spearmanr
                rho = spearmanr(pred.cpu().numpy(), gold.cpu().numpy()).statistic
                res[tid] = {"primitive": prim, "n": len(rows), "spearman": float(rho)}
            else:
                y = torch.tensor([r["y"] for r in rows], device=dev)
                p = torch.softmax(logits.float(), -1)
                acc = float((p.argmax(-1) == y).float().mean())
                auroc = None
                if logits.shape[1] == 2:
                    yi = 1 if rows[0]["cands"][1].lower().startswith(("yes", "true")) else 0
                    from sklearn.metrics import roc_auc_score
                    try:
                        auroc = float(roc_auc_score((y == yi).cpu().numpy(),
                                  torch.softmax(logits.float(), -1)[:, yi].cpu().numpy()))
                    except Exception:
                        auroc = None
                res[tid] = {"primitive": prim, "n": len(rows), "accuracy": acc,
                            "auroc": auroc}
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(JEV / "models/Qwen3-0.6B"))
    ap.add_argument("--arch", choices=["shared", "heads"], required=True)
    ap.add_argument("--tasks", nargs="*", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batches-per-family", type=int, default=30)
    ap.add_argument("--max-per-task", type=int, default=2048)
    ap.add_argument("--dev-max", type=int, default=256)
    ap.add_argument("--dev-eval-every", type=int, default=200)
    ap.add_argument("--warmup-frac", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="stop after N updates")
    ap.add_argument("--mb", type=int, default=16, help="rows per forward (VRAM unit)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation forwards per update")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--name", default="qwen_run")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = JEV / "artifacts/benchmark_v2_train" / a.name
    out.mkdir(parents=True, exist_ok=True)
    tok, model = build_model(a.model, a.qlora)
    mark_id = tok.convert_tokens_to_ids(MARK)
    dev = "cuda"
    model.to(dev)
    hidden = model.config.hidden_size

    train, devrows, stats = {}, {}, {}
    for tid in a.tasks:
        rows = load_task(UNI / tid, a.max_per_task, "train", random.Random(a.seed))
        if len(rows) < 64: continue
        train[tid] = rows
        stats[tid] = task_stats(rows)
        devrows[tid] = load_task(UNI / tid, a.dev_max, "dev", random.Random(20261001))
    fams = sorted({family_of(t) for t in train})
    total = a.epochs * a.batches_per_family * len(fams)
    if a.smoke: total = min(total, a.smoke)

    scorer = Scorer(hidden).to(dev, dtype=torch.bfloat16) if a.arch == "shared" else None
    heads = None
    if a.arch == "heads":
        heads = torch.nn.ModuleDict({
            t: (torch.nn.Linear(hidden, 1) if train[t][0]["prim"] == "score"
                else torch.nn.Linear(hidden, len(train[t][0]["cands"]))).to(dev, dtype=torch.bfloat16)
            for t in train})
    params = [p for p in model.parameters() if p.requires_grad]
    if scorer is not None: params += list(scorer.parameters())
    if heads is not None: params += list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4 if a.arch == "heads" else 5e-5)

    pools = {f: [t for t in train if family_of(t) == f] for f in fams}
    step = 0; best_dev = -1.0; trace = []
    t0 = time.monotonic()
    while step < total:
        for f in fams:
            for _ in range(a.batches_per_family):
                if step >= total: break
                tid = random.choice(pools[f])
                opt.zero_grad()
                loss_acc = 0.0
                oom = False
                for _acc in range(a.accum):
                    rows_b = [random.choice(train[tid]) for _ in range(a.mb)]
                    prim = rows_b[0]["prim"]
                    if a.arch == "shared":
                        allT, cands = [], None
                        for r in rows_b:
                            ts_, cands = render_shared(r, stats[tid]); allT += ts_
                        enc = pack(tok, allT)
                        lens = enc["attention_mask"].sum(1).tolist()
                        # chunk texts so each checkpointed forward stays under the
                        # recompute-memory budget (~1800 tokens at 8B QLoRA)
                        chunks, cur, cur_len = [], [], 0
                        for i, L in enumerate(lens):
                            if cur and cur_len + L > 1200:
                                chunks.append(cur); cur, cur_len = [], 0
                            cur.append(i); cur_len += L
                        if cur: chunks.append(cur)
                        lgs = []
                        for ch in chunks:
                            e = {k: v[ch] for k, v in enc.items()}
                            e = {k: v.to(dev) for k, v in e.items()}
                            h = ckpt(lambda ids, mask: model(input_ids=ids, attention_mask=mask,
                                     use_cache=False).last_hidden_state,
                                     e["input_ids"], e["attention_mask"], use_reentrant=False)
                            pos = (e["input_ids"] == mark_id).float().argmax(dim=1)
                            lgs.append(scorer(h.gather(1, pos[:, None, None].expand(-1, 1,
                                     h.shape[-1])).squeeze(1)))
                        lg = torch.cat(lgs).view(len(rows_b), len(cands))
                        if prim == "score":
                            gold = torch.tensor([(r["gold"] - stats[tid]["mean"]) / stats[tid]["std"]
                                                 for r in rows_b], device=dev)
                            av = torch.tensor([(x - stats[tid]["mean"]) / stats[tid]["std"]
                                               for x in stats[tid]["anchors"]], device=dev)
                            p = torch.softmax(lg, -1)
                            loss = F.mse_loss((p * av).sum(-1).float(), gold)
                        else:
                            y = torch.tensor([r["y"] for r in rows_b], device=dev)
                            loss = F.cross_entropy(lg.float(), y)
                    else:
                        enc = pack(tok, [render_heads(r) for r in rows_b])
                        enc = {k: v.to(dev) for k, v in enc.items()}
                        o = ckpt(lambda ids, mask: model(input_ids=ids, attention_mask=mask,
                                 use_cache=False).last_hidden_state,
                                 enc["input_ids"], enc["attention_mask"], use_reentrant=False)
                        m = enc["attention_mask"].unsqueeze(-1)
                        pooled = (o * m).sum(1) / m.sum(1)
                        lg = heads[tid](pooled)
                        if prim == "score":
                            gold = torch.tensor([(r["gold"] - stats[tid]["mean"]) / stats[tid]["std"]
                                                 for r in rows_b], device=dev)
                            loss = F.mse_loss(lg.squeeze(-1).float(), gold)
                        else:
                            loss = F.cross_entropy(lg.float(),
                                                   torch.tensor([r["y"] for r in rows_b], device=dev))
                    try:
                        (loss / a.accum).backward()
                        loss_acc += loss.item()
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        break
                if oom:
                    step += 1
                    trace.append({"step": step, "loss": None, "oom_skip": True,
                                  "sec": round(time.monotonic() - t0, 1)})
                    continue
                torch.nn.utils.clip_grad_norm_(params, a.clip)
                wf = max(1, int(a.warmup_frac * total))
                lr_f = step / wf if step <= wf else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - wf) / max(1, total - wf)))
                for g in opt.param_groups: g["lr"] = (1e-4 if a.arch == "heads" else 5e-5) * lr_f
                opt.step()
                step += 1
                trace.append({"step": step, "loss": round(loss_acc, 4),
                              "sec": round(time.monotonic() - t0, 1)})
                if step % 40 == 0:
                    print(json.dumps(trace[-1]), flush=True)
                if a.dev_eval_every and step % a.dev_eval_every == 0:
                    res = eval_tasks(model, tok, heads, scorer,
                                     {t: (devrows[t], stats[t]) for t in devrows}, a.arch)
                    vals = [float(v.get("accuracy", v.get("spearman"))) for v in res.values()
                            if v.get("accuracy", v.get("spearman")) is not None]
                    ds = float(np.mean(vals)) if vals else 0.0
                    print(json.dumps({"dev_eval": step, "dev_score": round(ds, 4)}), flush=True)
                    if ds > best_dev:
                        best_dev = ds
                        save_file({k: v.detach().cpu() for k, v in model.state_dict().items()},
                                  str(out / "best.safetensors"))
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    (out / "train_trace.jsonl").write_text("\n".join(json.dumps(t) for t in trace) + "\n")
    (out / "run_config.json").write_text(json.dumps(vars(a) | {"tasks": list(train),
        "total_updates": step, "best_dev": best_dev}, indent=2) + "\n")
    res = eval_tasks(model, tok, heads, scorer,
                     {t: (load_task(UNI / t, 200, "test", random.Random(20261001)), stats[t]) for t in train}, a.arch)
    (out / "eval_results.json").write_text(json.dumps({"name": a.name, "results": res},
                                                      ensure_ascii=False, indent=2) + "\n")
    print(f"[DONE] {a.name} {step} updates in {round(time.monotonic()-t0,1)}s", flush=True)

if __name__ == "__main__":
    main()
