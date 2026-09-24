#!/usr/bin/env python3
"""Fair, held-out pilot for Laya-Bio original-vocabulary vs M2.

This pilot is still too small to be a paper result.  It exists to validate the
comparison protocol after the GPU smoke tests:

* train and validation are cleaned from the two-task audit policy;
* M1 and M2 each start from the same untouched Laya checkpoint;
* the same natural-language/sequence serialization is used in both conditions;
* the M2 vocabulary uses modality-scoped tokens (`▶` DNA, `◆` protein) built
  from the existing source BPE tokenizers, preventing bare biological tokens
  from matching English prompt text;
* a held-out validation accuracy/NLL is computed only as a pilot diagnostic;
* finite-loss/gradient checks and checkpoint reload are explicit.

No CPT data is read.  The raw JSONL files remain unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tokenizers import AddedToken, Tokenizer
from transformers import AutoTokenizer

from laya_gpu_validation import (
    TASKS,
    collate,
    extract_context_and_sequence,
    expand_tokenizer_and_model,
    make_items,
    move_batch,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="artifacts/laya_model")
    p.add_argument("--output-dir", default="artifacts/laya_fair_pilot")
    p.add_argument("--laya-repo", default="vendor/laya")
    p.add_argument("--train-per-class", type=int, default=100)
    p.add_argument("--val-per-class", type=int, default=50)
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--expanded-tokens", type=int, default=64)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--head-max-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260922)
    return p.parse_args()


def load_laya(repo: str):
    import sys

    sys.path.insert(0, repo)
    from laya.common import QTYPES, build_model, build_sequence

    return QTYPES, build_model, build_sequence


def raw_rows(path: Path, split: str) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("split") == split:
                out.append(rec)
    return out


def seq_key(rec: dict[str, Any]) -> str:
    return extract_context_and_sequence(rec)[1].strip().upper()


def clean_train_val(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_raw, val_raw = raw_rows(path, "train"), raw_rows(path, "val")
    grouped: dict[str, set[str]] = defaultdict(set)
    for rec in train_raw + val_raw:
        grouped[seq_key(rec)].add(str(rec["answer_short"]))
    conflict = {k for k, labels in grouped.items() if len(labels) > 1}
    train, train_seen = [], set()
    for rec in train_raw:
        key = seq_key(rec)
        if key in conflict or key in train_seen:
            continue
        train_seen.add(key)
        train.append(rec)
    val, val_seen = [], set()
    for rec in val_raw:
        key = seq_key(rec)
        if key in conflict or key in train_seen or key in val_seen:
            continue
        val_seen.add(key)
        val.append(rec)
    return train, val, {
        "raw_train": len(train_raw),
        "raw_val": len(val_raw),
        "clean_train": len(train),
        "clean_val": len(val),
        "conflict_sequences_removed": len(conflict),
        "train_val_exact_overlap_after_cleanup": len(train_seen & val_seen),
    }


def balanced(rows: list[dict[str, Any]], per_class: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in rows:
        groups[str(rec["answer_short"])].append(rec)
    rng = random.Random(seed)
    out = []
    for label in sorted(groups):
        group = list(groups[label])
        rng.shuffle(group)
        out.extend(group[:per_class])
    rng.shuffle(out)
    return out


def modality(rec: dict[str, Any]) -> str:
    return "dna" if "dna" in rec.get("modality", []) else "protein"


def source_serializers() -> dict[str, Tokenizer]:
    return {
        "dna": Tokenizer.from_file("data/02_vocab/dna_bpe_20k.json"),
        "protein": Tokenizer.from_file("data/02_vocab/protein_bpe_8k.json"),
    }


def source_state_fn(source: dict[str, Tokenizer]):
    def state_fn(rec: dict[str, Any]) -> str:
        context, sequence = extract_context_and_sequence(rec)
        kind = modality(rec)
        pieces = source[kind].encode(sequence).tokens
        prefix = "▶" if kind == "dna" else "◆"
        # Every source-BPE piece gets the modality prefix.  This reproduces the
        # isolation convention in the saved expansion reference: no bare token
        # can match a word in the natural-language context.
        serialized = prefix + prefix.join(pieces)
        return f"Task context: {context}\nSequence: {serialized}"

    return state_fn


def build_candidates(rows, source: dict[str, Tokenizer], base_tok, n_new: int):
    # Count source-BPE pieces on train rows only.  Candidate strings are already
    # modality-scoped, so this count is not a raw-substring approximation.
    counts = {"dna": Counter(), "protein": Counter()}
    prompt_texts = []
    for rec in rows:
        context, _ = extract_context_and_sequence(rec)
        prompt_texts.append(context)
        prompt_texts.extend(str(x) for x in rec["choices"])
        kind = modality(rec)
        prefix = "▶" if kind == "dna" else "◆"
        for piece in source[kind].encode(extract_context_and_sequence(rec)[1]).tokens:
            counts[kind][prefix + piece] += 1
    ranked = {}
    for kind, counter in counts.items():
        ranked[kind] = [
            token for token, count in sorted(
                counter.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0])
            )
            if len(token) >= 4 and not any(token in text for text in prompt_texts)
        ]
    n_dna = n_new // 2
    n_protein = n_new - n_dna
    candidates = []
    dna, protein = ranked["dna"][: n_dna * 4], ranked["protein"][: n_protein * 4]
    for i in range(max(len(dna), len(protein))):
        if i < len(dna):
            candidates.append(dna[i])
        if i < len(protein):
            candidates.append(protein[i])
    # Return ranked metadata as well; final selection is done by the shared
    # embedding initializer after checking the base vocabulary.
    return candidates, ranked


def load_model(model_dir, cfg, build_model, device):
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for name, parameter in model.named_parameters():
        if name.startswith("act_head."):
            parameter.requires_grad_(False)
    model.to(device)
    return model


def train(model, items, tokenizer, device, steps, micro_batch, grad_accum, seed, tag):
    if steps % grad_accum:
        raise ValueError("--steps must be divisible by --grad-accum for this pilot")
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01
    )
    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    order = list(range(len(items)))
    rng = random.Random(seed)
    cursor = 0
    losses, updates, examples = [], 0, 0
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for step in range(steps):
        if cursor + micro_batch > len(order):
            rng.shuffle(order)
            cursor = 0
        chunk = [items[i] for i in order[cursor: cursor + micro_batch]]
        cursor += micro_batch
        batch = move_batch(collate(chunk, tokenizer.pad_token_id), device)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits, _ = model(
                batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                batch["marker_mask"], batch["qtype"],
            )
            loss = torch.nn.functional.cross_entropy(
                logits.masked_fill(~batch["marker_mask"], -1e4), batch["label"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{tag}: non-finite loss at step {step + 1}")
        scaler.scale(loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
                error_if_nonfinite=True,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"{tag}: non-finite gradient at step {step + 1}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        losses.append(float(loss.detach().cpu()))
        examples += len(chunk)
    return {
        "tag": tag,
        "steps": steps,
        "optimizer_updates": updates,
        "examples_consumed": examples,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "seconds": time.time() - start_time,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "amp_dtype": str(amp_dtype),
        "finite": True,
    }


def evaluate(model, items, tokenizer, device, batch_size=8):
    model.eval()
    records = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start: start + batch_size]
            batch = move_batch(collate(chunk, tokenizer.pad_token_id), device)
            with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits, _ = model(
                    batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                    batch["marker_mask"], batch["qtype"],
                )
            probs = torch.softmax(logits.masked_fill(~batch["marker_mask"], -1e4).float(), -1)
            for i, item in enumerate(chunk):
                k = len(item["markers"])
                p = probs[i, :k]
                gold = int(item["label"])
                records.append({
                    "task": item["task"],
                    "correct": int(int(p.argmax()) == gold),
                    "nll": float(-torch.log(p[gold].clamp_min(1e-9))),
                })
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    result = {}
    for task, values in sorted(by_task.items()):
        result[task] = {
            "n": len(values),
            "accuracy": sum(x["correct"] for x in values) / len(values),
            "nll": sum(x["nll"] for x in values) / len(values),
        }
    result["all"] = {
        "n": len(records),
        "accuracy": sum(x["correct"] for x in records) / len(records),
        "nll": sum(x["nll"] for x in records) / len(records),
    }
    return result


def grad_by_task(model, items, tokenizer, device, new_ids):
    out = {}
    model.train()
    for task in sorted({x["task"] for x in items}):
        covered = [x for x in items if x["task"] == task and any(i in new_ids for i in x["ids"])]
        if not covered:
            out[task] = None
            continue
        model.zero_grad(set_to_none=True)
        batch = move_batch(collate(covered[:4], tokenizer.pad_token_id), device)
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
        out[task] = None if grad is None else float(grad[new_ids].norm().detach().cpu())
    model.zero_grad(set_to_none=True)
    return out


def candidate_text_ids(tokenizer, rows):
    texts = []
    for rec in rows:
        context, _ = extract_context_and_sequence(rec)
        texts.extend([context, *map(str, rec["choices"]), "Choose the correct biological label for the sequence."])
    return [tokenizer(x, add_special_tokens=False)["input_ids"] for x in texts]


def main():
    a = args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda:0")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = Path(a.model_dir)
    qtypes, build_model, build_sequence = load_laya(a.laya_repo)
    source = source_serializers()
    state_fn = source_state_fn(source)

    train_rows, val_rows, cleanup = {}, {}, {}
    for task, spec in TASKS.items():
        tr, va, stats = clean_train_val(Path(spec["file"]))
        train_rows[task], val_rows[task], cleanup[task] = tr, va, stats
    train_selected = sum(
        (balanced(train_rows[t], a.train_per_class, a.seed + i) for i, t in enumerate(TASKS)), []
    )
    val_selected = sum(
        (balanced(val_rows[t], a.val_per_class, a.seed + 100 + i) for i, t in enumerate(TASKS)), []
    )

    base_tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    with (model_dir / "rl_agent_config.json").open() as fh:
        cfg = json.load(fh)
    cfg.update({"max_len": a.max_len, "head_max_len": a.head_max_len, "gradient_checkpointing": True})
    base_weights = load_file(str(model_dir / "model.safetensors"))

    def fresh_model():
        model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
        model.load_state_dict(base_weights, strict=True)
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        for name, parameter in model.named_parameters():
            if name.startswith("act_head."):
                parameter.requires_grad_(False)
        model.to(device)
        return model

    # M1: untouched tokenizer and untouched checkpoint.
    m1_tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    m1_train, _ = make_items(train_selected, m1_tok, build_sequence, qtypes, a.max_len, a.head_max_len, state_fn)
    m1_val, _ = make_items(val_selected, m1_tok, build_sequence, qtypes, a.max_len, a.head_max_len, state_fn)
    m1 = fresh_model()
    m1_train_result = train(m1, m1_train, m1_tok, device, a.steps, a.micro_batch, a.grad_accum, a.seed, "M1-original")
    m1_eval = evaluate(m1, m1_val, m1_tok, device)
    del m1
    torch.cuda.empty_cache()
    gc.collect()

    # M2: same untouched checkpoint, then a modality-scoped compact expansion.
    m2_tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    candidates, ranked = build_candidates(train_selected, source, m2_tok, a.expanded_tokens)
    m2 = fresh_model()
    chosen, old_size = expand_tokenizer_and_model(m2_tok, m2, candidates, a.expanded_tokens)
    m2_train, m2_train_lengths = make_items(train_selected, m2_tok, build_sequence, qtypes, a.max_len, a.head_max_len, state_fn)
    m2_val, m2_val_lengths = make_items(val_selected, m2_tok, build_sequence, qtypes, a.max_len, a.head_max_len, state_fn)
    m1_train_lengths = [x["n_tokens"] for x in m1_train]
    m1_val_lengths = [x["n_tokens"] for x in m1_val]
    m1_prompt_ids = candidate_text_ids(base_tok, val_selected)
    m2_prompt_ids = candidate_text_ids(m2_tok, val_selected)
    prompt_changed = sum(x != y for x, y in zip(m1_prompt_ids, m2_prompt_ids))
    m2_train_result = train(m2, m2_train, m2_tok, device, a.steps, a.micro_batch, a.grad_accum, a.seed, "M2-expanded")
    m2_eval = evaluate(m2, m2_val, m2_tok, device)
    new_ids = list(range(old_size, len(m2_tok)))
    modality_grad = grad_by_task(m2, m2_train, m2_tok, device, new_ids)

    # Save/reload M2 using the expanded tokenizer from disk and compare the
    # re-tokenized held-out inputs before running another finite evaluation.
    m2.eval()
    checkpoint = output / "m2_checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in m2.state_dict().items()}, str(checkpoint / "model.safetensors"))
    m2.encoder.config.save_pretrained(checkpoint / "encoder")
    m2_tok.save_pretrained(checkpoint / "tokenizer")
    (checkpoint / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    disk_tok = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    disk_val, _ = make_items(val_selected, disk_tok, build_sequence, qtypes, a.max_len, a.head_max_len, state_fn)
    reload_ids_match = all(x["ids"] == y["ids"] for x, y in zip(m2_val, disk_val))
    reloaded = load_laya(a.laya_repo)[1](cfg, encoder_dir=str(checkpoint / "encoder"))
    reloaded.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    reloaded.to(device).eval()
    reload_eval = evaluate(reloaded, disk_val, disk_tok, device)
    del reloaded

    summary = {
        "pilot_only": True,
        "no_cpt": True,
        "gpu": torch.cuda.get_device_name(device),
        "vram_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
        "cuda_version": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "laya_source_revision": "573e5b62696ba441230cd6be71d593331b5d23af",
        "model_revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
        "cleanup": cleanup,
        "n_train": len(train_selected),
        "n_val": len(val_selected),
        "train_tasks": sorted({x["task_id"] for x in train_selected}),
        "conditions_share_initial_checkpoint": True,
        "steps": a.steps,
        "micro_batch": a.micro_batch,
        "grad_accum": a.grad_accum,
        "m1_training": m1_train_result,
        "m1_val": m1_eval,
        "m2_training": m2_train_result,
        "m2_val": m2_eval,
        "m2_reload_val": reload_eval,
        "reload_tokenizer_ids_match": reload_ids_match,
        "base_vocab_size": len(base_tok),
        "expanded_vocab_size": len(m2_tok),
        "new_tokens": chosen,
        "candidate_counts_by_source": {k: len(v) for k, v in ranked.items()},
        "prompt_only_tokenization_changed_examples": prompt_changed,
        "token_lengths": {
            "m1_train_p50": sorted(m1_train_lengths)[len(m1_train_lengths) // 2],
            "m2_train_p50": sorted(m2_train_lengths)[len(m2_train_lengths) // 2],
            "m1_val_p50": sorted(m1_val_lengths)[len(m1_val_lengths) // 2],
            "m2_val_p50": sorted(m2_val_lengths)[len(m2_val_lengths) // 2],
            "m1_train_max": max(m1_train_lengths),
            "m2_train_max": max(m2_train_lengths),
            "m1_val_max": max(m1_val_lengths),
            "m2_val_max": max(m2_val_lengths),
        },
        "new_embedding_grad_norm_by_task": modality_grad,
        "source_tokenizer_isolation": "modality-prefixed source BPE pieces serialized with ▶/◆",
        "formal_paper_result": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
