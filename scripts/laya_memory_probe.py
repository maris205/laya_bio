#!/usr/bin/env python3
"""Measure full-parameter Laya supervised-CE memory on a single GPU.

This is an engineering probe only.  It uses synthetic token ids (every position is
attended, length 1024), so it does not produce a biological result.  The probe loads
the fixed local typed-decisions checkpoint, keeps parameters in FP32, uses BF16 CUDA
autocast, enables gradient checkpointing on the ModernBERT encoder, and runs AdamW
updates for micro-batches 4 and 8.  The decision head's ``act_head`` is frozen because
this CE probe has no action target.  Laya's ``head_checkpointing`` attribute is only a
configuration flag; it is deliberately not counted as checkpointing here.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="artifacts/laya_model")
    p.add_argument("--laya-repo", default="vendor/laya")
    p.add_argument("--output", default="artifacts/laya_memory_probe.json")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--micro-batches", type=int, nargs="+", default=[4, 8])
    return p.parse_args()


def driver_info() -> dict[str, Any]:
    """Read driver details without making nvidia-smi a hard dependency."""
    info: dict[str, Any] = {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        rows = []
        for line in out.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) >= 4:
                rows.append(
                    {
                        "index": fields[0],
                        "name": fields[1],
                        "memory_total_mib": fields[2],
                        "driver_version": fields[3],
                    }
                )
        info["nvidia_smi"] = rows
    except (OSError, subprocess.CalledProcessError) as exc:
        info["nvidia_smi_error"] = str(exc)
    return info


def import_laya(repo: str):
    repo_path = str(Path(repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from laya.common import build_model

    return build_model


def load_fixed_model(model_dir: Path, build_model, device: torch.device):
    """Build and load on CPU first, then move one copy to the GPU."""
    with (model_dir / "rl_agent_config.json").open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    encoder_dir = model_dir / "encoder"
    model = build_model(cfg, encoder_dir=str(encoder_dir))
    # safetensors is explicitly read on CPU.  Loading into the CPU model first
    # avoids a transient second full model allocation on the GPU.
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()

    # This is the real activation checkpointing switch for ModernBERT.  The
    # DecisionModel.head_checkpointing attribute is not used by its forward pass.
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to(device)
    model.train()
    # There is no action supervision in this CE objective.  Keep act_head out of
    # autograd and report that it was frozen rather than pretending it was trained.
    for parameter in model.act_head.parameters():
        parameter.requires_grad_(False)
    model.act_head.eval()
    return model, cfg


def memory_snapshot(device: torch.device) -> dict[str, float]:
    return {
        "allocated_gib": float(torch.cuda.memory_allocated(device) / 2**30),
        "reserved_gib": float(torch.cuda.memory_reserved(device) / 2**30),
        "max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 2**30),
    }


def assert_finite_gradients(model: torch.nn.Module) -> tuple[float, int]:
    finite_count = 0
    squared = None
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all().item(), "non-finite gradient"
        finite_count += 1
        value = parameter.grad.detach().float().norm(2)
        squared = value * value if squared is None else squared + value * value
    assert finite_count > 0, "no trainable parameter received a gradient"
    grad_norm = float(torch.sqrt(squared).detach().cpu())
    assert torch.isfinite(torch.tensor(grad_norm)), "non-finite aggregate gradient norm"
    return grad_norm, finite_count


def run_trial(
    model: torch.nn.Module,
    device: torch.device,
    micro_batch: int,
    seq_len: int,
    updates: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    # Parameters remain FP32; BF16 is used only by CUDA autocast.
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    assert parameter_dtypes == ["torch.float32"], parameter_dtypes
    assert torch.cuda.is_bf16_supported(), "the requested BF16 CUDA autocast is unavailable"

    model.train()
    model.act_head.eval()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    optimizer.zero_grad(set_to_none=True)

    vocab_size = int(model.encoder.config.vocab_size)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    input_ids_cpu = torch.randint(
        0, vocab_size, (micro_batch, seq_len), generator=rng, dtype=torch.long
    )
    attention_cpu = torch.ones((micro_batch, seq_len), dtype=torch.long)
    # Two valid choice markers.  Their token values are irrelevant to the memory probe.
    marker_pos_cpu = torch.tensor([[0, 1]] * micro_batch, dtype=torch.long)
    marker_mask_cpu = torch.ones((micro_batch, 2), dtype=torch.bool)
    qtype_cpu = torch.zeros(micro_batch, dtype=torch.long)
    labels_cpu = torch.arange(micro_batch, dtype=torch.long) % 2

    torch.cuda.reset_peak_memory_stats(device)
    update_records: list[dict[str, Any]] = []
    sync_times: list[float] = []
    t_total_start = time.perf_counter()

    for update in range(updates):
        # Include host-to-device copies outside the measured synchronized model step.
        batch = {
            "input_ids": input_ids_cpu.to(device, non_blocking=True),
            "attention_mask": attention_cpu.to(device, non_blocking=True),
            "marker_pos": marker_pos_cpu.to(device, non_blocking=True),
            "marker_mask": marker_mask_cpu.to(device, non_blocking=True),
            "qtype": qtype_cpu.to(device, non_blocking=True),
            "labels": labels_cpu.to(device, non_blocking=True),
        }
        pre_t = time.perf_counter()
        torch.cuda.synchronize(device)
        sync_times.append(time.perf_counter() - pre_t)
        compute_t = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )
            logits = logits.masked_fill(~batch["marker_mask"], -1e4)
            loss = F.cross_entropy(logits, batch["labels"])
        assert torch.isfinite(loss).item(), "non-finite loss"
        loss.backward()
        grad_norm, finite_grad_tensors = assert_finite_gradients(model)
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        assert torch.isfinite(clipped_norm).item(), "non-finite clipped gradient norm"
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        post_t = time.perf_counter()
        torch.cuda.synchronize(device)
        sync_times.append(time.perf_counter() - post_t)
        wall_seconds = time.perf_counter() - compute_t

        state_tensors = sum(
            1 for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq") if key in state
        )
        snap = memory_snapshot(device)
        update_records.append(
            {
                "optimizer_update": update + 1,
                "loss": float(loss.detach().cpu()),
                "grad_norm_before_clip": grad_norm,
                "clipped_grad_norm": float(clipped_norm.detach().cpu()),
                "finite_gradient_tensors": finite_grad_tensors,
                "adam_state_tensors": state_tensors,
                "wall_seconds_with_synchronize": wall_seconds,
                "memory": snap,
            }
        )
        print(
            f"[memory-probe] microbatch={micro_batch} update={update + 1}/{updates} "
            f"loss={float(loss.detach().cpu()):.5f} grad={grad_norm:.3f} "
            f"max_alloc={snap['max_allocated_gib']:.2f}GiB "
            f"max_reserved={snap['max_reserved_gib']:.2f}GiB",
            flush=True,
        )
        del batch, logits, loss

    total_seconds = time.perf_counter() - t_total_start
    final = memory_snapshot(device)
    return {
        "micro_batch": micro_batch,
        "seq_len": seq_len,
        "attention_all_ones": True,
        "updates": updates,
        "parameter_dtypes": parameter_dtypes,
        "autocast_dtype": "torch.bfloat16",
        "encoder_gradient_checkpointing": bool(
            getattr(model.encoder, "is_gradient_checkpointing", False)
        ),
        "head_checkpointing_attribute": bool(getattr(model, "head_checkpointing", False)),
        "head_checkpointing_used": False,
        "act_head_frozen": all(not p.requires_grad for p in model.act_head.parameters()),
        "total_seconds": total_seconds,
        "synchronize_seconds": {
            "count": len(sync_times),
            "sum": float(sum(sync_times)),
            "max": float(max(sync_times) if sync_times else 0.0),
        },
        "updates_detail": update_records,
        "final_memory": final,
        "peak_memory": {
            "max_allocated_gib": max(x["memory"]["max_allocated_gib"] for x in update_records),
            "max_reserved_gib": max(x["memory"]["max_reserved_gib"] for x in update_records),
        },
        "assertions": {
            "finite_loss_each_update": True,
            "finite_gradients_each_update": True,
            "adam_states_present_after_update_1": update_records[0]["adam_state_tensors"] > 0,
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the GPU should be enabled before this probe")
    assert args.seq_len == 1024, "the requested probe is defined for 1024-token inputs"
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_dir = Path(args.model_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_model = import_laya(args.laya_repo)
    props = torch.cuda.get_device_properties(device)
    result: dict[str, Any] = {
        "probe": "laya_full_supervised_ce_memory",
        "synthetic_inputs_only": True,
        "model_dir": str(model_dir),
        "laya_repo": str(Path(args.laya_repo).resolve()),
        "gpu": {
            "index": 0,
            "name": props.name,
            "total_memory_gib": float(props.total_memory / 2**30),
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            **driver_info(),
        },
        "config": {
            "seq_len": args.seq_len,
            "updates": args.updates,
            "micro_batches": args.micro_batches,
            "lr": args.lr,
            "seed": args.seed,
            "optimizer": "AdamW",
            "parameters": "FP32",
            "autocast": "BF16",
            "encoder_gradient_checkpointing": True,
            "no_cpt": True,
        },
        "trials": [],
    }

    for micro_batch in args.micro_batches:
        # Each trial gets a fresh model and optimizer, ensuring the second trial's
        # peak is not contaminated by the first trial's Adam state.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = None
        try:
            model, cfg = load_fixed_model(model_dir, build_model, device)
            result["checkpoint"] = {
                "encoder": cfg.get("encoder"),
                "max_len": cfg.get("max_len"),
                "head_max_len": cfg.get("head_max_len"),
                "vocab_size": int(model.encoder.config.vocab_size),
            }
            loaded = memory_snapshot(device)
            trial = run_trial(
                model,
                device,
                micro_batch,
                args.seq_len,
                args.updates,
                args.lr,
                args.seed + micro_batch,
            )
            trial["model_loaded_memory"] = loaded
            result["trials"].append(trial)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            result["trials"].append(
                {
                    "micro_batch": micro_batch,
                    "status": "oom",
                    "error": str(exc),
                    "memory_at_oom": memory_snapshot(device),
                }
            )
            print(f"[memory-probe] microbatch={micro_batch} OOM", flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    result["status"] = "ok" if all(t.get("status", "ok") == "ok" for t in result["trials"]) else "partial"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("[memory-probe] summary", flush=True)
    for trial in result["trials"]:
        if trial.get("status") == "oom":
            print(f"  microbatch={trial['micro_batch']}: OOM", flush=True)
        else:
            peak = trial["peak_memory"]
            print(
                f"  microbatch={trial['micro_batch']}: "
                f"max_alloc={peak['max_allocated_gib']:.2f} GiB, "
                f"max_reserved={peak['max_reserved_gib']:.2f} GiB, "
                f"updates={trial['updates']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
