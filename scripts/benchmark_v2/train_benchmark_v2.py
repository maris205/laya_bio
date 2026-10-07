#!/usr/bin/env python3
"""Train a shared Laya-JEV decision model on benchmark v2 (task-agnostic).

Generalizes the MVP1 frozen trainer to an arbitrary set of single-sequence
DNA/protein noul/choice/score tasks. Reuses the identical model path
(SharedDecision / build / render / pack / Representation) and optimizer/LR,
with a balanced random interleave over the selected tasks and per-task Score
anchors derived from each task's own train split.

Outputs a checkpoint consumable by eval_benchmark_v2.py.
"""
from __future__ import annotations
import argparse, json, math, random, hashlib, sys, time
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
from torch.nn import functional as F
from safetensors.torch import save_file
from laya_jev_multitask_data import Representation, interpolate, TYPE_IDS
import laya_jev_multitask_train as t
from transformers import AutoTokenizer

MAXLEN = 512
SEED = 20261001
BATCH = 64
LR_ENC, LR_HEAD = 2e-5, 1e-4


PREFIX_FAMILY = [("pg_","ProteinGym"),("rnac_","RNAcompete"),("gue_","GUE"),("gb_","GenomicBenchmarks"),
                 ("gl_","gene_lan"),("dna_","dnagpt_pools"),("deepstarr","DeepSTARR"),("tape_","TAPE"),
                 ("lg_","local_snapshots"),("protein_homology","local_snapshots"),("deeploc","DeepLoc")]

def family_of(tid):
    for pre,f in PREFIX_FAMILY:
        if tid.startswith(pre): return f
    return "other"

def discover_supported():
    """single-sequence dna/protein noul/choice/score task dirs under OUT."""
    out=[]
    for d in sorted(OUT.iterdir()):
        if not d.is_dir(): continue
        tf=d/"train.jsonl"
        if not tf.exists(): continue
        try: first=json.loads(open(tf).readline())
        except Exception: continue
        if first["primitive"] not in ("noul","choice","score"): continue
        seqs=first.get("sequences",[])
        if len(seqs)!=1 or seqs[0]["modality"] not in ("dna","protein"): continue
        out.append(d.name)
    return out

def schedule_family(bytask, epochs, batch, bpf, rng):
    """Each family gets `bpf` batches per epoch (equal family budget) so no
    single family (e.g. ProteinGym's 217 assays) floods the diverse families.
    Within a family, batches are pooled across its tasks and cycled."""
    pool = {}  # family -> list[(task, rows)]
    for fam in {family_of(t) for t in bytask}:
        slices = []
        for tk in [x for x in bytask if family_of(x) == fam]:
            rows = bytask[tk]
            for s in range(0, len(rows) - batch + 1, batch):
                slices.append((tk, rows[s:s+batch]))
        pool[fam] = slices
    for epoch in range(1, epochs + 1):
        for fam, slices in pool.items():
            if not slices:
                continue
            order = list(range(len(slices)))
            rng.shuffle(order)
            for i in range(bpf):
                tk, rr = slices[order[i % len(order)]]
                yield epoch, tk, rr

def derive_anchors(cuts):
    c = list(cuts)
    if len(c) >= 2:
        step = (c[-1] - c[0]) / (len(c) - 1)
        return [c[0] - step/2, (c[0]+c[1])/2, (c[1]+c[2])/2, (c[2]+c[3])/2, c[-1] + step/2]
    return [c[0]]*5 if c else [0.0]*5


def load_task_rows(rep, task_id, maxn, split="train"):
    """benchmark v2 <split>.jsonl -> prepared MVP1 rows + per-task score spec."""
    f = OUT / task_id / f"{split}.jsonl"
    if not f.exists():
        return [], None
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not recs:
        return [], None
    prim = recs[0]["primitive"]
    rng = np.random.default_rng(hash(task_id) & 0xffffffff)
    if len(recs) > maxn:
        recs = [recs[i] for i in sorted(rng.choice(len(recs), size=maxn, replace=False))]
    spec = None
    rows = []
    # for balanced single-label tasks, class-balance the cap; for score keep order
    if prim in ("noul", "choice"):
        byclass = defaultdict(list)
        for r in recs: byclass[str(r.get("answer"))].append(r)
        k = max(1, maxn // max(1, len(byclass)))
        recs = [r for b in byclass.values() for r in b[:k]]
    anchors = recs[0].get("anchors") if prim == "score" else None
    if prim == "score" and anchors:
        a5 = derive_anchors(anchors); spec = {"anchors": a5, "values": []}
    for r in recs:
        seqs = r["sequences"]
        if len(seqs) != 1 or seqs[0]["modality"] not in ("dna", "protein"):
            continue
        m = seqs[0]["modality"]; s = seqs[0]["sequence"]
        if prim == "noul":
            # use the benchmark's raw candidates (["no","yes"]); the "false:/true:"
            # prefix was redundant and hurt noul discrimination (see noul_ablation).
            choices = list(r["candidates"])
            label = 1 if r["answer"] == "yes" else 0
        elif prim == "choice":
            choices = list(r["candidates"])
            if r["answer"] not in choices: continue
            label = choices.index(r["answer"])
        elif prim == "score":
            choices = [f"value = {a:.6g}" for a in a5]
            label = int(r["gold_level"])
            val = float(r["gold_value"])
            spec["values"].append(val)
        else:
            continue
        row = {"id": r.get("group","") + f":{len(rows)}", "task": task_id, "primitive": prim,
               "modality": m, "question": r["question"], "sequence": s, "choices": choices, "label": label}
        if prim == "score":
            row["value"] = val
            row["anchors"] = a5
            row["target_probs"] = interpolate(val, a5)
        try:
            row = rep.prepare(row)
        except Exception:
            continue  # sequence has chars the DNA/protein BPE cannot encode (N/X/*/lowercase)
        if row["length"] > MAXLEN:
            continue
        rows.append(row)
    if spec is not None and spec["values"]:
        v = np.array(spec["values"]); spec["train_mean"] = float(v.mean()); spec["train_std"] = float(v.std() or 1)
    return rows, spec


def lr_factor(step, total, warmup_frac=0.05):
    warmup = max(1, int(warmup_frac*total))
    if step <= warmup: return step/warmup
    return .1 + .9*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))


def make_optimizer(model):
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        rate = LR_ENC if name.startswith("encoder.") else LR_HEAD
        decay = 0. if p.ndim < 2 or "norm" in name.lower() or "embeddings" in name or "type_emb" in name else .01
        groups[(rate, decay)].append(p)
    return torch.optim.AdamW([{"params": ps, "lr": rt, "base_lr": rt, "weight_decay": dc} for (rt, dc), ps in groups.items()])


def losses(logits, rows, spec):
    labels = torch.tensor([r["presented_label"] for r in rows], device=logits.device)
    if rows[0]["primitive"] != "score":
        return F.cross_entropy(logits, labels, reduction="none").sum()
    targets = torch.tensor([r["target_probs"] for r in rows], device=logits.device)
    ce = -(targets * F.log_softmax(logits, dim=-1)).sum(-1)
    anchors = torch.tensor(spec["anchors"], device=logits.device)
    pred = logits.softmax(-1) @ anchors
    true = torch.tensor([r["value"] for r in rows], device=logits.device)
    std = spec.get("train_std", float(true.std())) or 1.0
    return (.5*ce + .5*((pred-true)/std).square()).sum()


def schedule(bytask, rng):
    # balanced: each task contributes equal-size batches; interleave batch order randomly
    task_batches = min(len(v)//BATCH for v in bytask.values())
    assert task_batches >= 1, "each selected task needs >= BATCH rows after capping"
    for epoch in range(1, EPOCHS+1):
        for tk in bytask: rng.shuffle(bytask[tk])
        steps = []
        for b in range(task_batches):
            tasks = list(bytask); rng.shuffle(tasks)
            for tk in tasks:
                steps.append((epoch, tk, bytask[tk][b*BATCH:(b+1)*BATCH]))
        yield from steps


@torch.no_grad()
def dev_eval(model, devrows, specs, pad, micro=32):
    """Return {task_id: metric} where metric in [0,1]: choice/noul=accuracy,
    score=(spearman+1)/2. Caller aggregates globally or per family."""
    from scipy.stats import spearmanr
    model.eval()
    per_task = {}
    for tk, rows in devrows.items():
        if not rows: continue
        prim = rows[0]["primitive"]
        ys, preds, goldv, estv = [], [], [], []
        for st in range(0, len(rows), micro):
            rr = [t.render(r) for r in rows[st:st+micro]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**t.pack(rr, pad))
            for row, logit in zip(rr, logits):
                k = len(row["choices"])
                p = logit[:k].float().softmax(-1).cpu().numpy()
                canon = np.zeros(k)
                for j, idx in enumerate(row["order"]): canon[idx] = p[j]
                ys.append(row["label"]); preds.append(int(canon.argmax()))
                if prim == "score":
                    a5 = (specs.get(tk) or {}).get("anchors")
                    if a5: estv.append(float(np.dot(canon, a5))); goldv.append(row.get("value", row["label"]))
        if prim == "score" and len(goldv) > 2:
            rho = spearmanr(np.array(goldv), np.array(estv)).statistic
            per_task[tk] = (float(rho)+1)/2 if np.isfinite(rho) else 0.5
        else:
            yv = np.array(ys); pv = np.array(preds)
            per_task[tk] = float((pv == yv).mean()) if len(yv) else 0.0
    model.train()
    return per_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="auto-discover all supported single-seq dna/protein tasks")
    ap.add_argument("--max-per-task", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--family-balance", action="store_true", help="equal batches-per-epoch per family instead of per task")
    ap.add_argument("--batches-per-family", type=int, default=8)
    ap.add_argument("--min-valid", type=int, default=64, help="drop tasks with fewer prepared valid rows")
    ap.add_argument("--cap-family", nargs="*", default=[], help="FAMILY=N limit tasks in FAMILY to top-N by size")
    ap.add_argument("--exclude-family", nargs="*", default=[], help="hold out entire families from training (cross-family zero-shot test)")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--head-lr-scale", type=float, default=1.0, help="scale head/scorer LR (encoder LR unchanged)")
    ap.add_argument("--dev-eval-every", type=int, default=0, help=">0: eval dev every N updates for early stopping")
    ap.add_argument("--dev-max", type=int, default=64, help="dev samples per task for the early-stop metric")
    ap.add_argument("--per-family-early-stop", action="store_true", help="also save a best-dev checkpoint per family")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--name", default="bv2_pilot")
    ap.add_argument("--save-init", action="store_true", help="dump the pre-training checkpoint for a clean zero-shot baseline")
    ap.add_argument("--arm", default="no_cpt", choices=["no_cpt", "cpt"], help="encoder init: base tokenizer-expanded model (no_cpt) or biology-CPT checkpoint (cpt)")
    ap.add_argument("--out", type=Path, default=JEV/"artifacts/benchmark_v2_train")
    a = ap.parse_args()
    global EPOCHS, SEED; EPOCHS = a.epochs; SEED = a.seed
    tasks = list(a.tasks or [])
    if a.all: tasks = discover_supported()
    caps = {}
    for spec in a.cap_family:
        f, _, n = spec.partition("="); caps[f] = int(n)
    # cap family task count (top-N by train.jsonl line count) to bound prep cost
    for f, n in caps.items():
        fam_tasks = [x for x in tasks if family_of(x) == f]
        if len(fam_tasks) <= n: continue
        fam_tasks.sort(key=lambda x: sum(1 for _ in open(OUT/x/"train.jsonl")), reverse=True)
        keep = set(fam_tasks[:n])
        tasks = [x for x in tasks if family_of(x) != f or x in keep]
    # hold out entire families for cross-family zero-shot testing
    excluded = set(a.exclude_family or [])
    if excluded:
        held = [x for x in tasks if family_of(x) in excluded]
        tasks = [x for x in tasks if family_of(x) not in excluded]
        print(f"[holdout] families {sorted(excluded)}: {len(held)} tasks held out from training", flush=True)
    torch.set_num_threads(8)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    out = a.out / a.name; out.mkdir(parents=True, exist_ok=True)
    rep = Representation(CPT/"data")
    pad = AutoTokenizer.from_pretrained(CPT/"data/representation/base_tokenizer").pad_token_id
    model = t.build(MODEL, CPT, a.arm)
    if a.save_init:
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out/"init.safetensors"))
        print(f"[init checkpoint saved] {out/'init.safetensors'}", flush=True)
    bytask = {}; specs = {}; prep_fail = Counter()
    for tk in tasks:
        rows, spec = load_task_rows(rep, tk, a.max_per_task)
        if len(rows) < max(BATCH, a.min_valid):
            prep_fail[tk] = len(rows); continue
        bytask[tk] = rows; specs[tk] = spec
    nfam = len({family_of(x) for x in bytask})
    total = EPOCHS * a.batches_per_family * nfam if a.family_balance else EPOCHS * min(len(v)//BATCH for v in bytask.values()) * len(bytask)
    print(f"trained tasks={len(bytask)} families={nfam} balance={'family' if a.family_balance else 'task'} updates≈{total} | gated_out={len(prep_fail)}", flush=True)
    if bytask: print("  per-family task counts:", dict(Counter(family_of(x) for x in bytask)), flush=True)
    if not bytask: print("no tasks to train"); return 1
    # dev rows for early stopping (same tasks, dev split, small cap)
    devrows = {}
    if a.dev_eval_every > 0:
        for tk in bytask:
            drows, _ = load_task_rows(rep, tk, a.dev_max, split="dev")
            devrows[tk] = drows
        print(f"[dev-eval] prepared dev rows for {sum(1 for v in devrows.values() if v)}/{len(bytask)} tasks", flush=True)
    rng = random.Random(SEED+41)
    opt = make_optimizer(model)
    if a.head_lr_scale != 1.0:
        for g in opt.param_groups:
            if abs(g["base_lr"] - LR_HEAD) < 1e-12:
                g["base_lr"] = LR_HEAD * a.head_lr_scale
    step = 0; trace = []; best_dev = -1.0; best_step = 0; dev_curve = []
    fam_best = {}; fam_best_step = {}
    t0 = time.monotonic()
    sched = schedule_family(bytask, EPOCHS, BATCH, a.batches_per_family, rng) if a.family_balance else schedule(bytask, rng)
    for epoch, tk, rr in sched:
        step += 1
        for g in opt.param_groups: g["lr"] = g["base_lr"]*lr_factor(step, total, a.warmup_frac)
        model.train(); opt.zero_grad(set_to_none=True)
        rendered = [t.render(r, epoch, True) for r in rr]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**t.pack(rendered, pad))
            loss = losses(logits, rendered, specs[tk] or {"anchors":[0]*5,"train_std":1})
        (loss/len(rr)).backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip, error_if_nonfinite=False))
        opt.step()
        rec = {"step": step, "epoch": epoch, "task": tk, "loss": float(loss)/len(rr), "grad": grad}
        trace.append(rec)
        if step % 20 == 0 or step == 1:
            print(json.dumps({**rec, "sec": round(time.monotonic()-t0,1)}), flush=True)
        if a.dev_eval_every > 0 and step % a.dev_eval_every == 0:
            per_task = dev_eval(model, devrows, specs, pad)
            ds = float(np.mean(list(per_task.values()))) if per_task else 0.0
            # per-family dev score
            fam_scores = defaultdict(list)
            for tk, mv in per_task.items(): fam_scores[family_of(tk)].append(mv)
            fam_scores = {f: float(np.mean(v)) for f, v in fam_scores.items()}
            dev_curve.append({"step": step, "dev_score": ds, "per_family": fam_scores})
            improved = ds > best_dev
            if improved:
                best_dev = ds; best_step = step
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out/"best.safetensors"))
            # per-family best-dev checkpoints
            if a.per_family_early_stop:
                for f, sc in fam_scores.items():
                    if sc > fam_best.get(f, -1.0):
                        fam_best[f] = sc; fam_best_step[f] = step
                        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out/f"best_{f}.safetensors"))
            print(json.dumps({"dev_eval": step, "dev_score": round(ds,4), "best": round(best_dev,4), "best_step": best_step,
                              "fam_best": {f: round(s,3) for f,s in fam_best.items()} if a.per_family_early_stop else None}), flush=True)
    # final checkpoint always saved; if early-stop found a better dev point, it is in best.safetensors
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out/"model.safetensors"))
    if a.dev_eval_every > 0 and best_step > 0 and (out/"best.safetensors").exists():
        # promote best-dev as the primary model.safetensors; keep final for comparison
        import shutil
        shutil.copy2(out/"model.safetensors", out/"model_final.safetensors")
        shutil.copy2(out/"best.safetensors", out/"model.safetensors")
        print(f"[early-stop] promoted best-dev step {best_step} (dev {best_dev:.4f}) as model.safetensors; final kept as model_final.safetensors", flush=True)
    if dev_curve:
        (out/"dev_curve.jsonl").write_text("\n".join(json.dumps(r) for r in dev_curve)+"\n")
    (out/"train_trace.jsonl").write_text("\n".join(json.dumps(r) for r in trace)+"\n")
    config = {"tasks": list(bytask), "max_per_task": a.max_per_task, "epochs": EPOCHS, "batch": BATCH,
              "seed": SEED, "total_updates": step, "encoder_lr": LR_ENC, "head_lr": LR_HEAD,
              "family_balance": a.family_balance, "batches_per_family": a.batches_per_family,
              "min_valid": a.min_valid, "cap_family": caps, "gated_out": dict(prep_fail),
              "warmup_frac": a.warmup_frac, "clip": a.clip, "head_lr_scale": a.head_lr_scale,
              "dev_eval_every": a.dev_eval_every, "best_dev_score": best_dev, "best_step": best_step,
              "per_family_early_stop": a.per_family_early_stop, "fam_best_dev": fam_best, "fam_best_step": fam_best_step,
              "per_family_tasks": dict(Counter(family_of(x) for x in bytask)),
              "score_specs": {k: (v["anchors"] if v else None) for k, v in specs.items()},
              "elapsed_sec": round(time.monotonic()-t0,1), "test_inference": False}
    (out/"run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    print(f"[DONE] saved {out/'model.safetensors'} steps={step} elapsed={config['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
