"""Probe task interference in JEV joint checkpoints: gradient cosines and a classification-only forgetting test.

Train split only (fixed train-diagnostic panels); no dev or test access.
"""

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

PROJECT = Path(__file__).resolve().parents[1]
FROZEN = PROJECT / "artifacts/laya_jev_multitask_v1/round/frozen_code/laya_jev_multitask_train.py"
MODEL = PROJECT / "artifacts/laya_model"
CPT = PROJECT / "artifacts/laya_biocpt_v2"
DATA = PROJECT / "artifacts/laya_jev_multitask_v1/data"
JP = PROJECT / "artifacts/laya_jev_joint_path_diagnostics"
SEED = 20260928

CHECKPOINTS = {
    "initial": None,
    "interleaved": PROJECT / "artifacts/laya_jev_multitask_seed_confirm/run/round/seed_20260928_no_cpt_joint/model.safetensors",
    "gfp_last": JP / "run/round/seed_20260928_no_cpt_joint_task_block_gfp_last/model.safetensors",
    "gfp_last_rep1": JP / "run_replicate/round/seed_20260928_no_cpt_joint_task_block_gfp_last_rep1/model.safetensors",
    "gfp_first": JP / "run_followup/round/seed_20260928_no_cpt_joint_task_block_gfp_first/model.safetensors",
}
# Forgetting continuations: (start checkpoint, trainable parameter scope)
CONTINUATIONS = (
    ("gfp_last", "all"),
    ("gfp_last_rep1", "all"),
    ("gfp_last", "encoder_only"),
    ("gfp_last", "head_scorer_only"),
)
GRAD_BATCHES = 4
EVAL_EVERY = 32


def load_frozen():
    sys.path.insert(0, str(FROZEN.parent))
    spec = importlib.util.spec_from_file_location("frozen_jev_train", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEED = SEED
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_jev_joint_path_diagnostics import patch_task_block

    patch_task_block(module, ("fluorescence", "promoter", "structural_class"))
    return module


def group_of(name):
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("scorer."):
        return "scorer"
    return "head"


def load_model(m, path):
    model = m.build(MODEL, CPT, "no_cpt")
    if path is not None:
        model.load_state_dict(load_file(str(path)), strict=True)
    return model


def task_gradients(m, model, batches, pad, spec):
    """Mean gradient per task over fixed batches, dropout off, flattened per parameter group."""
    model.eval()
    grads = {}
    for task, task_batches in batches.items():
        model.zero_grad(set_to_none=True)
        for rows in task_batches:
            for start in range(0, len(rows), 8):
                rr = rows[start : start + 8]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = m.losses(model(**m.pack(rr, pad)), rr, spec).sum() / (len(rows) * len(task_batches))
                loss.backward()
        flat = {}
        for group in ("encoder", "head", "scorer"):
            parts = [p.grad.detach().float().flatten() for n, p in model.named_parameters() if group_of(n) == group and p.grad is not None]
            flat[group] = torch.cat(parts)
        flat["all"] = torch.cat([flat[g] for g in ("encoder", "head", "scorer")])
        grads[task] = flat
    model.zero_grad(set_to_none=True)
    result = {}
    pairs = (("fluorescence", "promoter"), ("fluorescence", "structural_class"), ("promoter", "structural_class"))
    for group in ("encoder", "head", "scorer", "all"):
        result[group] = {
            "norm": {task: float(grads[task][group].norm()) for task in grads},
            "cosine": {f"{a}|{b}": float(torch.nn.functional.cosine_similarity(grads[a][group], grads[b][group], dim=0)) for a, b in pairs},
        }
    return result


def score_panel(m, model, rows, pad, spec):
    records = m.evaluate(model, rows, pad, 16, spec)
    metrics = m.summarize(records, spec)
    out = {}
    for task, value in metrics.items():
        if task == "fluorescence":
            out[task] = {k: value[k] for k in ("rmse", "mae", "spearman", "prediction_std")}
        else:
            out[task] = {k: value[k] for k in ("accuracy", "macro_f1")}
    return out


def train_step(m, model, opt, rows, pad, spec):
    """Frozen train_step without its fixed-module gradient-norm probes (they fail when a scope is frozen)."""
    model.train()
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, len(rows), 8):
        rr = rows[start : start + 8]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m.losses(model(**m.pack(rr, pad)), rr, spec).sum() / len(rows)
        loss.backward()
        total += float(loss.detach())
    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    opt.step()
    return {"loss": total}


def continuation(m, start, scope, steps, panel, pad, spec, log):
    """Replay gfp_first's post-GFP epoch-3 classification updates from a GFP-last checkpoint."""
    model = load_model(m, CHECKPOINTS[start])
    for name, p in model.named_parameters():
        g = group_of(name)
        p.requires_grad_(scope == "all" or (scope == "encoder_only" and g == "encoder") or (scope == "head_scorer_only" and g != "encoder"))
    opt = m.make_optimizer(model)
    total = 1152
    curve = [{"step": 0, **score_panel(m, model, panel, pad, spec)}]
    log(json.dumps({"start": start, "scope": scope, **curve[0]}))
    for index, (global_step, epoch, task, rows) in enumerate(steps, 1):
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * m.lr_factor(global_step, total)
        rendered = [m.render(r, epoch, True) for r in rows]
        trace = train_step(m, model, opt, rendered, pad, spec)
        if index % EVAL_EVERY == 0:
            point = {"step": index, "task": task, "loss": trace["loss"], **score_panel(m, model, panel, pad, spec)}
            curve.append(point)
            log(json.dumps({"start": start, "scope": scope, **point}))
    del model, opt
    torch.cuda.empty_cache()
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=JP / "interference_probe")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    log_path = args.output / "probe.log"

    def log(line):
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    m = load_frozen()
    torch.set_num_threads(8)
    m.bio.seed(SEED)
    manifest = json.loads((DATA / "manifest.json").read_text())
    train_path = DATA / manifest["outputs"]["train"]["file"]
    assert m.sha(train_path) == manifest["outputs"]["train"]["sha256"]
    rows = m.read_lines(train_path)
    spec = manifest["score_spec"]
    from transformers import AutoTokenizer

    pad = AutoTokenizer.from_pretrained(CPT / "data/representation/base_tokenizer").pad_token_id

    # Fixed train-diagnostic panel (same selection rule as the trainer).
    panel = []
    for task in m.TASKS:
        pool = sorted([r for r in rows if r["task"] == task], key=lambda r: m.digest("train-eval:" + r["id"]))
        panel.extend(pool[: 64 if args.smoke else 1024])
    # Fixed gradient batches: first batches of each task in epoch 1 of the frozen shuffle, rendered as training panels.
    schedule = list(m.schedule(rows))
    batches = {task: [] for task in m.TASKS}
    for step, epoch, task, rr in schedule:
        if epoch == 1 and len(batches[task]) < (1 if args.smoke else GRAD_BATCHES):
            batches[task].append([m.render(r, epoch, True) for r in rr])
    # gfp_first epoch-3 classification updates that follow its GFP block.
    post_gfp = [(s, e, t, rr) for s, e, t, rr in schedule if e == 3 and t != "fluorescence"]
    assert len(post_gfp) == 256 and post_gfp[0][2] == "promoter" and post_gfp[-1][2] == "structural_class"
    if args.smoke:
        post_gfp = post_gfp[: EVAL_EVERY]

    started = time.monotonic()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "arm": "no_cpt", "splits_used": ["train"],
              "test_inference": False, "dev_inference": False, "panel_rows_per_task": len(panel) // 3,
              "gradient_batches_per_task": len(batches["promoter"]), "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()}}
    result["gradients"] = {}
    for name, path in CHECKPOINTS.items():
        if args.smoke and name not in ("initial", "gfp_last"):
            continue
        model = load_model(m, path)
        result["gradients"][name] = task_gradients(m, model, batches, pad, spec)
        log(json.dumps({"gradients": name, **{g: v["cosine"] for g, v in result["gradients"][name].items()}}))
        del model
        torch.cuda.empty_cache()
    result["forgetting"] = {}
    for start, scope in CONTINUATIONS[:1] if args.smoke else CONTINUATIONS:
        result["forgetting"][f"{start}:{scope}"] = continuation(m, start, scope, post_gfp, panel, pad, spec, log)
    result["elapsed_seconds"] = time.monotonic() - started
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    log(json.dumps({"status": "complete", "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
