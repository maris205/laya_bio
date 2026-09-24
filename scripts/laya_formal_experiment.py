#!/usr/bin/env python3
"""Formal Laya-Bio train/evaluate runner.

This runner deliberately contains no continual-pretraining path.  It loads the
fixed Laya typed-decision checkpoint, builds the biological choice prompt, and
trains one requested condition with supervised choice cross-entropy.  M2's
compact vocabulary is selected from the training split only.  The three
evaluation files are read after training and are never used for optimization
or vocabulary selection.

The formal data builder writes rows with ``context``, ``sequence``,
``choices``, ``label``, ``task``/``task_id`` and ``modality`` fields.  A small
compatibility parser also accepts the original BioPAWS message rows, which is
useful when checking an artifact before the formal split builder finishes.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file
from tokenizers import AddedToken, Tokenizer
from transformers import AutoTokenizer

try:
    from laya_metrics import classification_metrics, fit_temperature as fit_metric_temperature
except ImportError:  # pragma: no cover - direct module execution fallback
    from scripts.laya_metrics import classification_metrics, fit_temperature as fit_metric_temperature


ROOT = Path(__file__).resolve().parents[1]
TASK_FILES = {
    "promoter_detection": "promoter_detection",
    "fold_class": "fold_class",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=("b0", "m1", "m2"), required=True)
    p.add_argument("--task", choices=("promoter_detection", "fold_class", "both"), default="both")
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--model-dir", default="artifacts/laya_model")
    p.add_argument("--data-dir", default="artifacts/laya_formal_data")
    p.add_argument("--representation-dir", default="artifacts/laya_formal_representation",
                   help="frozen train-only Representation directory")
    p.add_argument("--eligible-ids", default=None,
                   help="optional JSON list or JSON object of IDs retained by the representation audit")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--laya-repo", default="vendor/laya")
    p.add_argument("--device", default="cuda")
    p.add_argument("--updates", type=int, default=0,
                   help="optimizer updates; 0 derives a fixed three-epoch budget")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--head-max-len", type=int, default=256)
    p.add_argument("--expanded-tokens", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-splits", default="selection_dev,calibration",
                   help="comma-separated split names; add test explicitly after model selection")
    p.add_argument("--allow-test", action="store_true",
                   help="permit reading/evaluating the locked test split")
    p.add_argument("--allow-truncation", action="store_true",
                   help="allow max_len truncation; default fails preflight")
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def import_laya(repo: str):
    repo_path = str(resolve(repo))
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from laya.common import QTYPES, build_model, build_sequence
    return QTYPES, build_model, build_sequence


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _message_fields(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages") or []
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    if "\n" in user:
        context, sequence = user.rsplit("\n", 1)
    else:
        context, sequence = "Biological sequence classification", user
    context = re.split(
        r",?\s*The result will be one of the following\s*:",
        context,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,")
    return context, sequence.strip()


def normalize_row(row: dict[str, Any], file_task: str) -> dict[str, Any]:
    context = row.get("context")
    sequence = row.get("sequence")
    if context is None or sequence is None:
        context, sequence = _message_fields(row)
    choices = [str(x) for x in row.get("choices", [])]
    if not choices:
        raise ValueError(f"row {row.get('id')} has no choices")
    label = row.get("label", row.get("answer_short"))
    if isinstance(label, bool):
        label = int(label)
    if isinstance(label, int):
        label_idx = label
    else:
        label_s = str(label)
        if label_s not in choices:
            raise ValueError(f"row {row.get('id')} label {label_s!r} absent from choices")
        label_idx = choices.index(label_s)
    if not 0 <= label_idx < len(choices):
        raise ValueError(f"row {row.get('id')} label index {label_idx} out of range")
    task = str(row.get("task_id", row.get("task", file_task)))
    modality = row.get("modality", [])
    if isinstance(modality, str):
        modality = [modality]
    modality = [str(x).lower() for x in modality]
    if not modality:
        modality = ["dna" if "promoter" in file_task else "protein"]
    kind = "dna" if any(x in {"dna", "nucleotide"} for x in modality) else "protein"
    rid = str(row.get("id", f"{task}:{hashlib.sha1(str(sequence).encode()).hexdigest()[:12]}"))
    return {
        "id": rid,
        "task": task,
        "context": str(context),
        "sequence": str(sequence).strip(),
        "choices": choices,
        "label": label_idx,
        "label_text": choices[label_idx],
        "kind": kind,
        "modality": modality,
    }


def load_split(data_dir: Path, task: str, split: str) -> list[dict[str, Any]]:
    tasks = list(TASK_FILES) if task == "both" else [task]
    out: list[dict[str, Any]] = []
    for task_name in tasks:
        stem = TASK_FILES[task_name]
        path = data_dir / f"{stem}_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing formal split: {path}")
        out.extend(normalize_row(x, task_name) for x in jsonl(path))
    if not out:
        raise ValueError(f"empty formal split {split!r} for task {task!r}")
    return out


def source_tokenizers() -> dict[str, Tokenizer]:
    return {
        "dna": Tokenizer.from_file(str(ROOT / "data/02_vocab/dna_bpe_20k.json")),
        "protein": Tokenizer.from_file(str(ROOT / "data/02_vocab/protein_bpe_8k.json")),
    }


def load_eligible_ids(path: str | None) -> set[str] | None:
    """Read the optional no-truncation eligibility list produced by the audit."""
    if not path:
        return None
    value = json.loads(resolve(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("tasks"), dict):
        # The formal audit stores IDs by task and split so counts and hashes
        # remain inspectable.  Flatten that frozen view for the trainer.
        flattened: list[str] = []
        for task in value["tasks"].values():
            for split in task.get("splits", {}).values():
                flattened.extend(str(x) for x in split.get("ids", []))
        value = flattened
    elif isinstance(value, dict):
        for key in ("eligible_ids", "ids", "kept_ids"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError("--eligible-ids must point to a JSON list or object containing eligible_ids")
    return {str(x) for x in value}


def load_representation(rep_dir: Path, condition: str):
    """Load the frozen representation module without importing it at startup."""
    module_path = ROOT / "scripts" / "laya_representation.py"
    if not rep_dir.exists() or not module_path.exists():
        return None
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from laya_representation import Representation
    rep = Representation.load(rep_dir, expanded=condition == "m2")
    base_rep = Representation.load(rep_dir, expanded=False)
    return rep, base_rep


def expansion_spec(rep_dir: Path, base_tokenizer, expanded_tokenizer):
    """Return added token IDs and base decompositions from flexible metadata."""
    token_file = rep_dir / "new_tokens.json"
    if not token_file.exists():
        raise FileNotFoundError(f"M2 representation is missing {token_file}")
    raw = json.loads(token_file.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        tokens = [str(x.get("token", x.get("content"))) if isinstance(x, dict) else str(x)
                  for x in raw]
    elif isinstance(raw, dict):
        tokens = raw.get("tokens") or raw.get("new_tokens") or raw.get("added_tokens")
        if tokens is None:
            tokens = list(raw)
        tokens = [str(x.get("token", x.get("content"))) if isinstance(x, dict) else str(x) for x in tokens]
    else:
        raise ValueError("new_tokens.json must be a list or object")
    metadata_path = rep_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    decompositions = (metadata.get("base_decompositions")
                      or metadata.get("decompositions")
                      or metadata.get("decompositions_before_mutation")
                      or {})
    if not decompositions and isinstance(raw, dict):
        decompositions = raw.get("base_decompositions") or raw.get("decompositions") or {}
    out: dict[str, list[int]] = {}
    for token in tokens:
        value = decompositions.get(token) if isinstance(decompositions, dict) else None
        if isinstance(value, dict):
            value = value.get("ids") or value.get("base_ids")
        if value is None and isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict) and str(entry.get("token")) == token:
                    value = entry.get("base_ids")
                    break
        if value is None:
            value = base_tokenizer(token, add_special_tokens=False)["input_ids"]
        if value and isinstance(value[0], str):
            # A metadata producer may store source pieces instead of IDs.
            value = [base_tokenizer(x, add_special_tokens=False)["input_ids"][0] for x in value]
        out[token] = [int(x) for x in value if int(x) < len(base_tokenizer)]
    missing = [x for x in tokens if x not in expanded_tokenizer.get_vocab()]
    if missing:
        raise ValueError(f"expanded tokenizer is missing {len(missing)} new tokens, e.g. {missing[:3]}")
    return tokens, out


def resize_from_representation(model, base_tokenizer, expanded_tokenizer, rep_dir: Path):
    tokens, decompositions = expansion_spec(rep_dir, base_tokenizer, expanded_tokenizer)
    old_size = len(base_tokenizer)
    if len(expanded_tokenizer) < old_size + len(tokens):
        raise ValueError("expanded tokenizer size is smaller than the base plus declared tokens")
    # We overwrite every added row with deterministic base-token means below;
    # disable Transformers' random mean/covariance initialization so resizing
    # cannot consume RNG state or leave an uninitialized row unnoticed.
    model.encoder.resize_token_embeddings(len(expanded_tokenizer), mean_resizing=False)
    emb = model.encoder.get_input_embeddings().weight
    fallback = 0
    with torch.no_grad():
        for token in tokens:
            new_id = int(expanded_tokenizer.get_vocab()[token])
            pieces = decompositions[token]
            pieces = [x for x in pieces if x < old_size]
            if pieces:
                emb[new_id].copy_(emb[torch.tensor(pieces, device=emb.device)].mean(dim=0))
            else:
                emb[new_id].copy_(emb[:old_size].mean(dim=0))
                fallback += 1
    return {"requested": len(tokens), "chosen": len(tokens), "chosen_tokens": tokens,
            "old_vocab_size": old_size, "new_vocab_size": len(expanded_tokenizer),
            "new_token_ids": [int(expanded_tokenizer.get_vocab()[x]) for x in tokens],
            "fallback_initializations": fallback,
            "metadata_decomposition_count": sum(bool(x) for x in decompositions.values())}


def raw_state(row: dict[str, Any]) -> str:
    return f"Task context: {row['context']}\nSequence: {row['sequence']}"


def expanded_state(row: dict[str, Any], source: dict[str, Tokenizer]) -> str:
    prefix = "▶" if row["kind"] == "dna" else "◆"
    pieces = source[row["kind"]].encode(row["sequence"]).tokens
    sequence = prefix + prefix.join(pieces)
    return f"Task context: {row['context']}\nSequence: {sequence}"


def candidate_tokens(rows: Iterable[dict[str, Any]], source: dict[str, Tokenizer], n_new: int,
                     base_tokenizer) -> tuple[list[str], dict[str, Any]]:
    counts: dict[str, Counter[str]] = {"dna": Counter(), "protein": Counter()}
    prompt_texts: list[str] = []
    for row in rows:
        prompt_texts.extend((row["context"], *row["choices"],
                             "Choose the correct biological label for the sequence."))
        pieces = source[row["kind"]].encode(row["sequence"]).tokens
        prefix = "▶" if row["kind"] == "dna" else "◆"
        for piece in pieces:
            token = prefix + piece
            if len(token) >= 4:
                counts[row["kind"]][token] += 1
    base_vocab = set(base_tokenizer.get_vocab())
    ranked: dict[str, list[str]] = {}
    for kind, counter in counts.items():
        ranked[kind] = [
            token for token, count in sorted(counter.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
            if token not in base_vocab and not any(token in text for text in prompt_texts)
        ]
    n_dna = n_new // 2
    n_protein = n_new - n_dna
    chosen: list[str] = []
    for i in range(max(n_dna, n_protein)):
        if i < n_dna and i < len(ranked["dna"]):
            chosen.append(ranked["dna"][i])
        if i < n_protein and i < len(ranked["protein"]):
            chosen.append(ranked["protein"][i])
    # The chosen list may be short when a source tokenizer has no usable pieces.
    # That is valid, but must be visible in the summary rather than silently
    # claiming the requested expansion was made.
    return chosen, {
        "requested": n_new,
        "chosen": len(chosen),
        "candidate_counts": {k: len(v) for k, v in ranked.items()},
        "occurrences": {k: sum(counts[k][x] for x in ranked[k][:len(chosen)]) for k in counts},
        "ranking": ranked,
    }


def expand_tokenizer_and_model(tokenizer, model, candidates: list[str]):
    old_size = len(tokenizer)
    base_vocab = set(tokenizer.get_vocab())
    chosen: list[str] = []
    for token in candidates:
        if token in base_vocab or token in chosen or len(token) < 4:
            continue
        chosen.append(token)
    special_ids = {x for x in (tokenizer.unk_token_id, tokenizer.pad_token_id,
                               tokenizer.cls_token_id, tokenizer.sep_token_id,
                               tokenizer.mask_token_id) if x is not None}
    decompositions = {
        token: [i for i in tokenizer(token, add_special_tokens=False)["input_ids"]
                if i < old_size and i not in special_ids]
        for token in chosen
    }
    added = tokenizer.add_tokens([AddedToken(x, normalized=False) for x in chosen])
    if added != len(chosen):
        raise RuntimeError(f"tokenizer added {added} tokens, expected {len(chosen)}")
    model.encoder.resize_token_embeddings(len(tokenizer))
    emb = model.encoder.get_input_embeddings().weight
    with torch.no_grad():
        for i, token in enumerate(chosen):
            ids = decompositions[token]
            if ids:
                emb[old_size + i].copy_(emb[torch.tensor(ids, device=emb.device)].mean(dim=0))
            else:
                emb[old_size + i].copy_(emb[:old_size].mean(dim=0))
    return chosen, old_size, decompositions


def make_items(rows, tokenizer, build_sequence, qtypes, max_len: int, head_max_len: int,
               state_fn, shuffle_choices: bool = False, shuffle_seed: int = 0):
    items: list[dict[str, Any]] = []
    lengths: list[int] = []
    full_lengths: list[int] = []
    truncated = 0
    marker_dropped = 0
    for row in rows:
        choices = row["choices"]
        q = {
            "t": "choice",
            "ins": "Choose the correct biological label for the sequence.",
            "crit": {x: None for x in choices},
        }
        state = state_fn(row)
        option_order = list(range(len(choices)))
        if shuffle_choices:
            # Stable per-row permutation: unlike Python's hash(), SHA-1 gives
            # identical train ordering across processes and machines.  M1/M2
            # receive the same permutation; the target index is remapped.
            digest = hashlib.sha256(f"{shuffle_seed}:{row['id']}".encode()).digest()
            local_rng = random.Random(int.from_bytes(digest[:8], "big"))
            local_rng.shuffle(option_order)
        ids, markers = build_sequence(tokenizer, state, q, max_len=max_len,
                                       head_max_len=head_max_len, option_order=option_order)
        # A very large limit gives the untruncated reference length while
        # retaining exactly the same head/choice formatting.
        full_ids, full_markers = build_sequence(tokenizer, state, q, max_len=1_000_000,
                                                head_max_len=head_max_len, option_order=option_order)
        if len(markers) != len(choices) or len(full_markers) != len(choices):
            raise ValueError(f"choice markers truncated for row {row['id']}: {len(markers)}/{len(choices)}")
        was_truncated = len(ids) < len(full_ids)
        truncated += int(was_truncated)
        marker_dropped += int(len(markers) != len(full_markers))
        items.append({
            "ids": ids,
            "markers": markers,
            "qtype": qtypes["choice"],
            "label": option_order.index(row["label"]),
            "task": row["task"],
            "id": row["id"],
            "n_tokens": len(ids),
            "full_tokens": len(full_ids),
            "truncated": was_truncated,
        })
        lengths.append(len(ids))
        full_lengths.append(len(full_ids))
    lengths_sorted = sorted(lengths)
    return items, {
        "n": len(items),
        "length_p50": lengths_sorted[len(lengths_sorted) // 2],
        "length_p95": lengths_sorted[max(0, int(len(lengths_sorted) * 0.95) - 1)],
        "length_max": max(lengths),
        "full_length_max": max(full_lengths),
        "truncated_count": truncated,
        "truncated_rate": truncated / len(items),
        "marker_dropped_count": marker_dropped,
    }


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
        ids[i, :len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        attention[i, :len(item["ids"])] = 1
        marker_pos[i, :len(item["markers"])] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[i, :len(item["markers"])] = True
        labels[i] = item["label"]
    return {"input_ids": ids, "attention_mask": attention, "marker_pos": marker_pos,
            "marker_mask": marker_mask, "qtype": torch.zeros(n, dtype=torch.long),
            "label": labels}


def move(batch, device):
    return {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}


def autocast_context(device, dtype):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def load_model(model_dir: Path, cfg: dict[str, Any], build_model, device, trainable: bool):
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # The RL action classifier is not part of the biological choice loss.  It
    # is frozen in every condition so the trainable parameter set is explicit.
    for name, parameter in model.named_parameters():
        if name.startswith("act_head."):
            parameter.requires_grad_(False)
        elif not trainable:
            parameter.requires_grad_(False)
    model.to(device)
    return model


def train_model(model, items, tokenizer, device, updates, micro_batch, grad_accum,
                lr, weight_decay, seed, log_every, epochs=3, warmup_fraction=0.05):
    if updates == 0:
        updates = math.ceil(len(items) * epochs / (micro_batch * grad_accum))
    if updates < 1:
        raise ValueError("--updates must be >= 1 for trainable conditions")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warmup_steps = int(math.ceil(updates * warmup_fraction))
    def schedule(step: int) -> float:
        if warmup_steps and step <= warmup_steps:
            return max(1e-8, step / warmup_steps)
        remain = max(1, updates - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remain))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not bf16)
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    cursor = 0
    losses: list[float] = []
    grad_norms: list[float] = []
    examples = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    model.train()
    for update in range(1, updates + 1):
        update_loss = 0.0
        for _ in range(grad_accum):
            if cursor + micro_batch > len(order):
                rng.shuffle(order)
                cursor = 0
            chunk = [items[i] for i in order[cursor:cursor + micro_batch]]
            cursor += micro_batch
            batch = move(collate(chunk, tokenizer.pad_token_id), device)
            with autocast_context(device, amp_dtype):
                logits, _ = model(batch["input_ids"], batch["attention_mask"],
                                  batch["marker_pos"], batch["marker_mask"], batch["qtype"])
                loss = torch.nn.functional.cross_entropy(
                    logits.masked_fill(~batch["marker_mask"], -1e4), batch["label"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer update {update}")
            scaled_loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            update_loss += float(loss.detach().cpu())
            examples += len(chunk)
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at optimizer update {update}")
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(update_loss / grad_accum)
        grad_norms.append(float(grad_norm.detach().cpu()))
        if log_every and (update == 1 or update % log_every == 0 or update == updates):
            print(f"update={update}/{updates} loss={losses[-1]:.5f} grad={grad_norms[-1]:.4f}", flush=True)
    return {
        "updates": updates,
        "epochs_requested": epochs,
        "warmup_fraction": warmup_fraction,
        "warmup_steps": warmup_steps,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "examples_consumed": examples,
        "effective_batch": micro_batch * grad_accum,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "grad_norm_max": max(grad_norms),
        "seconds": time.time() - started,
        "finite": True,
        "amp_dtype": str(amp_dtype) if device.type == "cuda" else "none",
        "lr_final": float(optimizer.param_groups[0]["lr"]),
    }


def evaluate(model, items, tokenizer, device, batch_size):
    model.eval()
    records: list[dict[str, Any]] = []
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            batch = move(collate(chunk, tokenizer.pad_token_id), device)
            with autocast_context(device, torch.bfloat16 if bf16 else torch.float16):
                logits, _ = model(batch["input_ids"], batch["attention_mask"],
                                  batch["marker_pos"], batch["marker_mask"], batch["qtype"])
            logits = logits.float()
            for i, item in enumerate(chunk):
                k = len(item["markers"])
                row_logits = logits[i, :k].detach().cpu()
                probs = torch.softmax(row_logits, dim=-1)
                records.append({"id": item["id"], "task": item["task"], "label": item["label"],
                                "n_classes": k, "logits": row_logits.tolist(), "probs": probs.tolist()})
    return records


def metric_from_records(records: list[dict[str, Any]], temperatures: dict[str, float] | None = None) -> dict[str, Any]:
    """Use the shared fixed-class metric implementation for every task."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_task[rec["task"]].append(rec)
    result: dict[str, Any] = {}
    for task, values in sorted(by_task.items()):
        k = int(values[0]["n_classes"])
        if any(int(x["n_classes"]) != k for x in values):
            raise ValueError(f"task {task} has inconsistent class counts")
        result[task] = classification_metrics(
            [x["logits"] for x in values], [x["label"] for x in values],
            n_classes=k, temperature=(temperatures or {}).get(task, 1.0)
        )
    # Mixed-task aggregate is reported only for accuracy/NLL-like quantities;
    # the fixed-K metrics above remain the primary task-level results.
    if len(result) == 1:
        result["all"] = next(iter(result.values()))
    else:
        total = sum(x["n"] for x in result.values())
        result["all"] = {"n": total, "accuracy": sum(x["n"] * x["accuracy"] for x in result.values()) / total,
                          "balanced_accuracy": sum(x["n"] * x["balanced_accuracy"] for x in result.values()) / total,
                          "macro_f1": sum(x["n"] * x["macro_f1"] for x in result.values()) / total,
                          "mcc": None,
                          "nll": sum(x["n"] * x["nll"] for x in result.values()) / total,
                          "brier": sum(x["n"] * x["brier"] for x in result.values()) / total,
                          "ece15": sum(x["n"] * x["ece15"] for x in result.values()) / total}
    return result


def save_checkpoint(model, tokenizer, cfg, checkpoint: Path) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, str(checkpoint / "model.safetensors"))
    model.encoder.config.save_pretrained(checkpoint / "encoder")
    tokenizer.save_pretrained(checkpoint / "tokenizer")
    (checkpoint / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def fresh_reload(checkpoint: Path, cfg: dict[str, Any], build_model, device):
    model = build_model(cfg, encoder_dir=str(checkpoint / "encoder"))
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    model.to(device).eval()
    return model


def main() -> None:
    a = parse_args()
    set_seed(a.seed)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu for a non-GPU smoke test")
    data_dir, model_dir = resolve(a.data_dir), resolve(a.model_dir)
    output = resolve(a.output_dir) if a.output_dir else ROOT / "artifacts" / "laya_formal" / f"{a.condition}_seed{a.seed}"
    output.mkdir(parents=True, exist_ok=True)
    qtypes, build_model, build_sequence = import_laya(a.laya_repo)

    requested_eval_splits = [x.strip() for x in a.eval_splits.split(",") if x.strip()]
    if "test" in requested_eval_splits and not a.allow_test:
        raise ValueError("test evaluation requires explicit --allow-test")
    load_splits = ["train", "selection_dev", "calibration"]
    if "test" in requested_eval_splits:
        load_splits.append("test")
    rows = {split: load_split(data_dir, a.task, split) for split in load_splits}
    eligible = load_eligible_ids(a.eligible_ids)
    if eligible is not None:
        rows = {split: [row for row in values if row["id"] in eligible]
                for split, values in rows.items()}
        if not rows["train"]:
            raise ValueError("eligible-id filter removed every training row")
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    cfg.update({"max_len": a.max_len, "head_max_len": a.head_max_len, "gradient_checkpointing": True})
    rep_bundle = load_representation(resolve(a.representation_dir), a.condition)
    if rep_bundle is None:
        raise FileNotFoundError(
            "formal runs require the frozen train-only representation directory; "
            f"expected {resolve(a.representation_dir)}"
        )
    source = source_tokenizers() if rep_bundle is None else None
    state_fn = raw_state
    if rep_bundle is not None:
        representation, base_representation = rep_bundle
        base_tok = base_representation.tokenizer
        tokenizer = representation.tokenizer
        state_fn = representation.state
        representation_source = str(resolve(a.representation_dir))
    else:
        base_tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        representation_source = "legacy-inline-source-bpe"
    model = load_model(model_dir, cfg, build_model, device, trainable=a.condition != "b0")
    expansion = None
    if a.condition == "m2":
        if rep_bundle is not None:
            expansion = resize_from_representation(model, base_tok, tokenizer, resolve(a.representation_dir))
        else:
            candidates, expansion = candidate_tokens(rows["train"], source, a.expanded_tokens, tokenizer)
            chosen, old_size, decompositions = expand_tokenizer_and_model(tokenizer, model, candidates)
            expansion.update({"chosen_tokens": chosen, "old_vocab_size": old_size,
                              "new_vocab_size": len(tokenizer),
                              "new_token_ids": list(range(old_size, len(tokenizer))),
                              "fallback_initializations": sum(not bool(x) for x in decompositions.values())})
            state_fn = lambda row: expanded_state(row, source)

    item_sets: dict[str, list[dict[str, Any]]] = {}
    item_stats: dict[str, Any] = {}
    for split in rows:
        item_sets[split], item_stats[split] = make_items(rows[split], tokenizer, build_sequence,
                                                          qtypes, a.max_len, a.head_max_len, state_fn,
                                                          shuffle_choices=(split == "train"),
                                                          shuffle_seed=a.seed)
    if not a.allow_truncation:
        truncated = {split: stats["truncated_count"] for split, stats in item_stats.items()
                     if stats["truncated_count"]}
        if truncated:
            raise RuntimeError(
                "formal preflight found max_len truncation; provide the audited eligible IDs "
                f"or pass --allow-truncation explicitly: {truncated}"
            )
    print(json.dumps({"condition": a.condition, "task": a.task, "item_stats": item_stats}, indent=2), flush=True)

    if a.condition == "b0":
        train_result = {"frozen": True, "updates": 0, "examples_consumed": 0, "finite": True}
    else:
        train_result = train_model(model, item_sets["train"], tokenizer, device, a.updates,
                                   a.micro_batch, a.grad_accum, a.lr, a.weight_decay,
                                   a.seed, a.log_every, a.epochs, a.warmup_fraction)

    eval_splits = requested_eval_splits
    valid_splits = {"selection_dev", "calibration", "test"}
    unknown = sorted(set(eval_splits) - valid_splits)
    if unknown:
        raise ValueError(f"unknown --eval-splits: {unknown}")
    if not eval_splits:
        raise ValueError("--eval-splits cannot be empty")
    raw_eval = {split: evaluate(model, item_sets[split], tokenizer, device, a.eval_batch)
                for split in eval_splits}
    if "calibration" not in raw_eval:
        raise ValueError("calibration split is required for temperature fitting")
    calibration_by_task = defaultdict(list)
    for rec in raw_eval["calibration"]:
        calibration_by_task[rec["task"]].append(rec)
    temperature_info = {}
    temperatures = {}
    for task_name, values in calibration_by_task.items():
        temperature_info[task_name] = fit_metric_temperature(
            [x["logits"] for x in values], [x["label"] for x in values],
            n_classes=int(values[0]["n_classes"])
        )
        temperatures[task_name] = float(temperature_info[task_name]["temperature"])
    metrics = {split: {"raw": metric_from_records(raw_eval[split]),
                       "calibrated": metric_from_records(raw_eval[split], temperatures)}
               for split in raw_eval}

    checkpoint = output / "checkpoint"
    save_checkpoint(model, tokenizer, cfg, checkpoint)
    # Strict reload catches tokenizer/encoder vocabulary mismatches before the
    # result is considered usable.  Compare one split's logits at a tight but
    # realistic tolerance; both models use the same saved float weights.
    disk_tok = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    reload_split = eval_splits[0]
    disk_items, _ = make_items(rows[reload_split], disk_tok, build_sequence, qtypes,
                               a.max_len, a.head_max_len, state_fn,
                               shuffle_choices=False, shuffle_seed=a.seed)
    reloaded = fresh_reload(checkpoint, cfg, build_model, device)
    reload_records = evaluate(reloaded, disk_items, disk_tok, device, a.eval_batch)
    reload_match = len(reload_records) == len(raw_eval[reload_split]) and all(
        max(abs(x - y) for x, y in zip(a_rec["logits"], b_rec["logits"])) < 1e-5
        for a_rec, b_rec in zip(raw_eval[reload_split], reload_records)
    )
    del reloaded
    gc.collect()

    new_grad_norm_by_task = {} if a.condition == "m2" else None
    if a.condition == "m2":
        model.train(); model.zero_grad(set_to_none=True)
        new_ids = list(expansion.get("new_token_ids", range(expansion["old_vocab_size"], expansion["new_vocab_size"])))
        for task in sorted({x["task"] for x in item_sets["train"]}):
            covered = [x for x in item_sets["train"] if x["task"] == task and any(i in new_ids for i in x["ids"])]
            if not covered:
                continue
            batch = move(collate(covered[:min(4, len(covered))], tokenizer.pad_token_id), device)
            bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
            with autocast_context(device, torch.bfloat16 if bf16 else torch.float16):
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
                loss = torch.nn.functional.cross_entropy(logits.masked_fill(~batch["marker_mask"], -1e4), batch["label"])
            loss.backward()
            grad = model.encoder.get_input_embeddings().weight.grad
            new_grad_norm_by_task[task] = None if grad is None else float(grad[new_ids].norm().detach().cpu())
        model.zero_grad(set_to_none=True)

    summary = {
        "formal": True, "no_cpt": True, "condition": a.condition, "task": a.task, "seed": a.seed,
        "model_dir": str(model_dir), "data_dir": str(data_dir), "output_dir": str(output),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "vram_gib": torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
        "condition_uses_only_train_split": True,
        "representation_source": representation_source,
        "eligible_id_filter": str(resolve(a.eligible_ids)) if a.eligible_ids else None,
        "evaluated_splits": eval_splits,
        "n_rows": {k: len(v) for k, v in rows.items()},
        "item_stats": item_stats,
        "training": train_result,
        "evaluation": metrics,
        "calibration_temperature_fit_on": "calibration",
        "calibration_temperature": temperature_info,
        "test_access": bool(a.allow_test and "test" in eval_splits),
        "checkpoint": str(checkpoint),
        "checkpoint_reload_logits_match": bool(reload_match),
        "expansion": expansion,
        "new_embedding_grad_norm_by_task": new_grad_norm_by_task,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
