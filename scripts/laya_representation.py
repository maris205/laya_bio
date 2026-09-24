#!/usr/bin/env python3
"""Train-only sequence representation used by the formal Laya experiment.

The representation intentionally keeps the *text* passed to both M1 and M2
identical.  It runs one fitted source BPE tokenizer over the sequence and
serializes every source piece with a modality-specific opening/closing pair:

    DNA     ``▶{piece}◀``
    protein ``◆{piece}◇``

The closing delimiter is part of every M2 AddedToken.  Consequently an added
token cannot match the prefix of a longer source piece (the failure mode of a
bare ``▶GGC`` token matching ``▶GGCC``).  Natural-language context is returned
unchanged apart from the same task/sequence labels used by the pilot.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer


DNA_OPEN, DNA_CLOSE = "▶", "◀"
PROTEIN_OPEN, PROTEIN_CLOSE = "◆", "◇"


def _user_text(record: Mapping[str, Any]) -> str:
    if "messages" in record:
        for message in record["messages"]:
            if message.get("role") == "user":
                return str(message.get("content", ""))
    for key in ("user", "prompt", "text", "input"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def extract_context_and_sequence(record: Mapping[str, Any]) -> tuple[str, str]:
    """Read either a BioPAWS row or a formal-data normalized row."""
    sequence = record.get("sequence")
    context = record.get("context")
    if sequence is not None:
        seq = str(sequence).strip()
        return (str(context).strip() if context is not None else "Biological sequence classification", seq)
    user = _user_text(record)
    if "\n" in user:
        context, sequence = user.rsplit("\n", 1)
    else:
        context, sequence = "Biological sequence classification", user
    # The source rows repeat the answer choices in the user message.  Choices
    # are supplied separately through the decision head; removing this suffix
    # avoids accidentally changing natural-language state between conditions.
    context = re.split(
        r",?\s*The result will be one of the following\s*:",
        context,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,")
    return context, sequence.strip()


def modality(record: Mapping[str, Any], *, source_name: str | None = None) -> str:
    value = record.get("modality", [])
    if isinstance(value, str):
        value = [value]
    value = {str(x).lower() for x in value}
    if "dna" in value:
        return "dna"
    if "protein" in value or "aa" in value:
        return "protein"
    name = (source_name or record.get("task_id") or record.get("id") or "").lower()
    if any(x in name for x in ("dna", "promoter", "splice", "npp")):
        return "dna"
    if any(x in name for x in ("protein", "fold", "signal", "homology")):
        return "protein"
    raise ValueError(f"Cannot infer sequence modality for record {record.get('id', '<unknown>')!r}")


class Representation:
    """Load and apply a frozen formal representation directory.

    ``expanded=False`` loads the base tokenizer (M1); ``expanded=True`` loads
    the AddedToken tokenizer (M2).  Both modes produce the same state string.
    """

    def __init__(self, root: Path, tokenizer, dna_source, protein_source, metadata: dict[str, Any], expanded: bool):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self._source = {"dna": dna_source, "protein": protein_source}
        self.metadata = metadata
        self.expanded = bool(expanded)

    @classmethod
    def load(cls, directory: str | Path, expanded: bool = False) -> "Representation":
        root = Path(directory)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"representation metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tok_dir = root / ("expanded_tokenizer" if expanded else "base_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        from tokenizers import Tokenizer

        source_dir = root / "source_tokenizers"
        dna_source = Tokenizer.from_file(str(source_dir / "dna_bpe_20k.json"))
        protein_source = Tokenizer.from_file(str(source_dir / "protein_bpe_8k.json"))
        return cls(root, tokenizer, dna_source, protein_source, metadata, expanded)

    def source_pieces(self, record: Mapping[str, Any]) -> tuple[str, list[str]]:
        context, sequence = extract_context_and_sequence(record)
        kind = modality(record)
        return kind, self._source[kind].encode(sequence).tokens

    @staticmethod
    def wrap_piece(kind: str, piece: str) -> str:
        if kind == "dna":
            return f"{DNA_OPEN}{piece}{DNA_CLOSE}"
        if kind == "protein":
            return f"{PROTEIN_OPEN}{piece}{PROTEIN_CLOSE}"
        raise ValueError(f"unknown modality: {kind}")

    def state(self, record: Mapping[str, Any]) -> str:
        context, _ = extract_context_and_sequence(record)
        kind, pieces = self.source_pieces(record)
        sequence = "".join(self.wrap_piece(kind, piece) for piece in pieces)
        return f"Task context: {context}\nSequence: {sequence}"

    # Alias used by a few training/audit scripts.
    serialize = state


__all__ = [
    "DNA_OPEN",
    "DNA_CLOSE",
    "PROTEIN_OPEN",
    "PROTEIN_CLOSE",
    "Representation",
    "extract_context_and_sequence",
    "modality",
]
