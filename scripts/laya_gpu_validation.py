#!/usr/bin/env python3
"""Small GPU validation for the independent Laya-Bio experiment.

This is deliberately a smoke test, not a production trainer.  It checks:

* the real Laya typed-decisions checkpoint loads on CUDA;
* BioPAWS-2 samples can be formatted with natural language + sequence input;
* supervised choice CE produces finite loss and gradients;
* a compact tokenizer expansion can be resized and its new embeddings receive
  gradients, without any unsupervised CPT step.

The script uses only a deterministic, small train subset and writes a JSON
summary plus optional expanded tokenizer under the requested output directory.
It does not modify the source dataset or publish a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tokenizers import AddedToken
from transformers import AutoTokenizer

# The repository is intentionally kept outside this project while the upstream
# Laya package is being evaluated.  Set PYTHONPATH to the checked-out repo or
# pass --laya-repo.


TASKS = {
    "lg_promoter_detection": {
        "file": "data/03_sft_biopaws2/jsonl/lg_promoter_detection.jsonl",
        "modality": "DNA",
    },
    "lg_fold_class": {
        "file": "data/03_sft_biopaws2/jsonl/lg_fold_class.jsonl",
        "modality": "protein",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="artifacts/laya_model")
    p.add_argument("--output-dir", default="artifacts/laya_validation")
    p.add_argument("--laya-repo", default="/tmp/laya_repo_inspect_2")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--head-max-len", type=int, default=256)
    p.add_argument("--expanded-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    return p.parse_args()


def import_laya(repo: str):
    import sys

    sys.path.insert(0, repo)
    from laya.common import QTYPES, build_model, build_sequence

    return QTYPES, build_model, build_sequence


def read_balanced(path: Path, per_class: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("split") == "train":
                rows.append(rec)

    by_label: dict[str, list[dict[str, Any]]] = {}
    for rec in rows:
        by_label.setdefault(str(rec["answer_short"]), []).append(rec)
    rng = random.Random(seed)
    out = []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        out.extend(group[:per_class])
    rng.shuffle(out)
    return out


def extract_context_and_sequence(rec: dict[str, Any]) -> tuple[str, str]:
    user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
    # BioPAWS-2 rows put the sequence after the final newline.  Keep the task
    # wording as natural-language context, but avoid duplicating the candidate
    # list in both state and choice criteria.
    if "\n" in user:
        context, sequence = user.rsplit("\n", 1)
    else:
        context, sequence = "Biological sequence classification", user
    # Keep the task wording, but strip the source row's repeated candidate list
    # from state.  Candidates are supplied once through the typed choice head.
    context = re.split(
        r",?\s*The result will be one of the following\s*:",
        context,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,")
    return context, sequence.strip()


def extract_state(rec: dict[str, Any]) -> str:
    context, sequence = extract_context_and_sequence(rec)
    return f"Task context: {context}\nSequence: {sequence}"


def extract_sequence(rec: dict[str, Any]) -> str:
    return extract_context_and_sequence(rec)[1]


def make_items(rows, tokenizer, build_sequence, qtypes, max_len, head_max_len, state_fn=extract_state):
    items = []
    lengths = []
    for rec in rows:
        choices = [str(x) for x in rec["choices"]]
        criteria = {x: None for x in choices}
        q = {
            "t": "choice",
            "ins": "Choose the correct biological label for the sequence.",
            "crit": criteria,
        }
        ids, markers = build_sequence(
            tokenizer,
            state_fn(rec),
            q,
            max_len=max_len,
            head_max_len=head_max_len,
        )
        label = choices.index(str(rec["answer_short"]))
        items.append(
            {
                "ids": ids,
                "markers": markers,
                "qtype": qtypes["choice"],
                "label": label,
                "task": rec["task_id"],
                "id": rec["id"],
                "n_tokens": len(ids),
            }
        )
        lengths.append(len(ids))
    return items, lengths


def collate(items, pad_id: int):
    n = len(items)
    length = max(len(x["ids"]) for x in items)
    n_markers = max(len(x["markers"]) for x in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, n_markers), dtype=torch.long)
    marker_mask = torch.zeros((n, n_markers), dtype=torch.bool)
    labels = torch.zeros(n, dtype=torch.long)
    for i, item in enumerate(items):
        ids[i, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        attention[i, : len(item["ids"])] = 1
        marker_pos[i, : len(item["markers"])] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[i, : len(item["markers"])] = True
        labels[i] = item["label"]
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": torch.zeros(n, dtype=torch.long),
        "label": labels,
    }


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_steps(model, items, tokenizer, device, steps, micro_batch, grad_accum, tag):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-5, weight_decay=0.01)
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    losses = []
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    peak_before = torch.cuda.max_memory_allocated(device)
    for step in range(steps):
        start = (step * micro_batch) % len(items)
        chunk = [items[(start + j) % len(items)] for j in range(micro_batch)]
        batch = move_batch(collate(chunk, tokenizer.pad_token_id), device)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )
            logits = logits.masked_fill(~batch["marker_mask"], -1e4)
            loss = torch.nn.functional.cross_entropy(logits, batch["label"])
        scaler.scale(loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0 or step + 1 == steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % max(1, steps // 4) == 0:
            print(
                f"[{tag}] step={step + 1}/{steps} loss={losses[-1]:.4f} "
                f"alloc={torch.cuda.memory_allocated(device) / 2**30:.2f}GiB "
                f"peak={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB",
                flush=True,
            )
    return {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "losses": losses,
        "seconds": time.time() - t0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_before_gib": peak_before / 2**30,
        "amp_dtype": str(amp_dtype),
    }


def expand_tokenizer_and_model(tokenizer, model, candidates, n_new):
    """Add a small candidate list and initialize rows from base-token means."""
    # Capture decompositions before mutating the tokenizer.  AddedToken entries
    # take precedence after insertion; encoding after mutation could therefore
    # return the very new ID we are trying to initialize.
    base_vocab = set(tokenizer.get_vocab())
    chosen = []
    for tok in candidates:
        if tok in base_vocab or tok in chosen or len(tok) < 4:
            continue
        chosen.append(tok)
        if len(chosen) >= n_new:
            break
    old_size = len(tokenizer)
    special_ids = {
        x for x in (
            tokenizer.unk_token_id,
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id,
        ) if x is not None
    }
    decompositions = {
        tok: [
            x for x in tokenizer(tok, add_special_tokens=False)["input_ids"]
            if x < old_size and x not in special_ids
        ]
        for tok in chosen
    }
    added = tokenizer.add_tokens([AddedToken(x, normalized=False) for x in chosen])
    if added != len(chosen):
        raise RuntimeError(f"tokenizer added {added}, expected {len(chosen)}")
    model.encoder.resize_token_embeddings(len(tokenizer))
    emb = model.encoder.get_input_embeddings().weight
    with torch.no_grad():
        for i, text in enumerate(chosen):
            new_id = old_size + i
            pieces = decompositions[text]
            if pieces:
                emb[new_id].copy_(emb[torch.tensor(pieces, device=emb.device)].mean(dim=0))
            else:
                emb[new_id].copy_(emb[:old_size].mean(dim=0))
    return chosen, old_size


def check_new_embedding_gradient(model, items, tokenizer, device, new_ids):
    """Run one backward pass and return the gradient norm on newly added rows."""
    model.train()
    model.zero_grad(set_to_none=True)
    batch = move_batch(collate(items[: min(4, len(items))], tokenizer.pad_token_id), device)
    with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        logits, _ = model(
            batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
            batch["marker_mask"], batch["qtype"],
        )
        loss = torch.nn.functional.cross_entropy(
            logits.masked_fill(~batch["marker_mask"], -1e4), batch["label"]
        )
    loss.backward()
    grad = model.encoder.get_input_embeddings().weight.grad
    norm = None if grad is None else float(grad[new_ids].norm().detach().cpu())
    model.zero_grad(set_to_none=True)
    return norm


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this validation is intended for the opened GPU.")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 2**30:.2f} GiB")

    qtypes, build_model, build_sequence = import_laya(args.laya_repo)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    with (model_dir / "rl_agent_config.json").open() as fh:
        cfg = json.load(fh)
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = args.head_max_len
    cfg["gradient_checkpointing"] = True
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    weights = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to(device)
    print(f"Loaded checkpoint with vocab={len(tokenizer)}", flush=True)

    all_rows = []
    for task, spec in TASKS.items():
        rows = read_balanced(Path(spec["file"]), args.per_class, args.seed)
        all_rows.extend(rows)
    items, lengths = make_items(
        all_rows, tokenizer, build_sequence, qtypes, args.max_len, args.head_max_len
    )
    print(
        f"Prepared {len(items)} items; token length p50={sorted(lengths)[len(lengths)//2]}, "
        f"p95={sorted(lengths)[max(0, int(len(lengths)*.95)-1)]}, max={max(lengths)}",
        flush=True,
    )

    base_result = run_steps(
        model,
        items,
        tokenizer,
        device,
        args.steps,
        args.micro_batch,
        args.grad_accum,
        "original-vocab",
    )

    # Use the existing BPE artifacts only as candidates.  The first compact
    # tokens are long/high-frequency entries in each source vocabulary; this is
    # a smoke test, not a test-set-driven vocabulary search.
    # Rank candidate BPE strings by occurrence in the sampled *training* rows.
    # This avoids the first smoke test's failure mode, where selecting very long
    # vocabulary entries by length produced no actual input coverage.  This is
    # only a train-subset validation; the final run will compute this ranking on
    # the complete training split and freeze it before looking at dev/test.
    sequences_by_kind = {
        "dna": [extract_sequence(r) for r in all_rows if "dna" in r.get("modality", [])],
        "protein": [extract_sequence(r) for r in all_rows if "protein" in r.get("modality", [])],
    }
    natural_texts = []
    for rec in all_rows:
        context, _ = extract_context_and_sequence(rec)
        natural_texts.append(context)
        natural_texts.extend(str(x) for x in rec["choices"])
    natural_texts.append("Choose the correct biological label for the sequence.")
    ranked_by_kind = {}
    for vocab_file in ("data/02_vocab/dna_bpe_20k.json", "data/02_vocab/protein_bpe_8k.json"):
        data = json.loads(Path(vocab_file).read_text())
        vocab = data["model"]["vocab"]
        kind = "dna" if "/dna_" in vocab_file else "protein"
        ranked = []
        for tok, vocab_id in vocab.items():
            if len(tok) < 4 or tok in {"[UNK]", "[PAD]"}:
                continue
            # AddedToken matching is global.  Do not admit a candidate that is
            # already a substring of the natural-language prompt/candidate
            # text, otherwise a protein token such as ALA could alter prose.
            if any(tok in text for text in natural_texts):
                continue
            count = sum(seq.count(tok) for seq in sequences_by_kind[kind])
            if count:
                ranked.append((count, len(tok), -int(vocab_id), tok))
        ranked_by_kind[kind] = [tok for _, _, _, tok in sorted(ranked, reverse=True)]
    # Reserve half of the compact budget for each sequence modality.  Without
    # this cap, DNA has more repeated short motifs and can consume the entire
    # budget before protein candidates are considered.
    n_dna = args.expanded_tokens // 2
    n_protein = args.expanded_tokens - n_dna
    dna_candidates = ranked_by_kind.get("dna", [])[: max(n_dna * 4, n_dna)]
    protein_candidates = ranked_by_kind.get("protein", [])[: max(n_protein * 4, n_protein)]
    candidates = []
    for i in range(max(len(dna_candidates), len(protein_candidates))):
        if i < len(dna_candidates):
            candidates.append(dna_candidates[i])
        if i < len(protein_candidates):
            candidates.append(protein_candidates[i])
    expanded, old_size = expand_tokenizer_and_model(tokenizer, model, candidates, args.expanded_tokens)
    expanded_items, expanded_lengths = make_items(
        all_rows, tokenizer, build_sequence, qtypes, args.max_len, args.head_max_len
    )
    torch.cuda.reset_peak_memory_stats()
    expanded_result = run_steps(
        model,
        expanded_items,
        tokenizer,
        device,
        args.steps,
        args.micro_batch,
        args.grad_accum,
        "expanded-vocab",
    )
    new_ids = list(range(old_size, len(tokenizer)))
    new_grad_norm = check_new_embedding_gradient(
        model, expanded_items, tokenizer, device, new_ids
    )
    tokenizer.save_pretrained(output_dir / "expanded_tokenizer")
    checkpoint_dir = output_dir / "expanded_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(cpu_state, str(checkpoint_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(checkpoint_dir / "encoder")
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    (checkpoint_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    # Reload from the just-written files and run one finite forward pass.  This
    # catches mismatches between the resized embedding, tokenizer, and encoder
    # config before a longer experiment is launched.
    reloaded = build_model(cfg, encoder_dir=str(checkpoint_dir / "encoder"))
    reloaded.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")), strict=True)
    reloaded.to(device).eval()
    reload_batch = move_batch(collate(expanded_items[:4], tokenizer.pad_token_id), device)
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ):
        reload_logits, _ = reloaded(
            reload_batch["input_ids"], reload_batch["attention_mask"],
            reload_batch["marker_pos"], reload_batch["marker_mask"], reload_batch["qtype"],
        )
    save_reload_ok = bool(torch.isfinite(reload_logits).all().item())
    del reloaded
    new_token_occurrences = sum(
        sum(i >= old_size for i in item["ids"]) for item in expanded_items
    )
    new_token_items = sum(any(i >= old_size for i in item["ids"]) for item in expanded_items)
    summary = {
        "gpu": torch.cuda.get_device_name(device),
        "vram_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
        "cuda_version": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "model_dir": str(model_dir),
        "n_items": len(items),
        "tasks": sorted({x["task"] for x in items}),
        "original_vocab_size": old_size,
        "expanded_vocab_size": len(tokenizer),
        "new_tokens": expanded,
        "new_token_count": len(expanded),
        "original_token_length": {
            "p50": sorted(lengths)[len(lengths) // 2],
            "p95": sorted(lengths)[max(0, int(len(lengths) * 0.95) - 1)],
            "max": max(lengths),
        },
        "expanded_token_length": {
            "p50": sorted(expanded_lengths)[len(expanded_lengths) // 2],
            "p95": sorted(expanded_lengths)[max(0, int(len(expanded_lengths) * 0.95) - 1)],
            "max": max(expanded_lengths),
        },
        "token_length_delta": {
            "p50": sorted(expanded_lengths)[len(expanded_lengths) // 2]
            - sorted(lengths)[len(lengths) // 2],
            "max": max(expanded_lengths) - max(lengths),
        },
        "expanded_token_coverage": {
            "items_with_new_token": new_token_items,
            "new_token_occurrences": new_token_occurrences,
        },
        "original_vocab_training": base_result,
        "expanded_vocab_training": expanded_result,
        "new_embedding_grad_norm_after_step": new_grad_norm,
        "save_reload_finite_forward": save_reload_ok,
        "no_cpt": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
