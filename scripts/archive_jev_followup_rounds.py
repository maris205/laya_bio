#!/usr/bin/env python3
"""Archive compact evidence for JEV follow-up rounds, keeping model states and resume files local."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

KEEP_SUFFIXES = {".json", ".jsonl", ".md", ".log"}
SKIP_DIRS = {".ipynb_checkpoints", "__pycache__"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive(root, output, exclude):
    files, skipped = [], []
    for src in sorted(root.rglob("*")):
        rel = src.relative_to(root)
        if not src.is_file() or SKIP_DIRS & set(rel.parts):
            continue
        if any(rel.parts[0] == name for name in exclude):
            continue
        if src.suffix not in KEEP_SUFFIXES:
            skipped.append({"path": str(rel), "bytes": src.stat().st_size})
            continue
        if src.name.endswith("_predictions.jsonl"):
            content = src.read_bytes()
            dest = output / (str(rel) + ".gz")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
            assert gzip.decompress(dest.read_bytes()) == content
            files.append({"source_relative": str(rel), "archive_relative": str(dest.relative_to(output)),
                          "source_sha256": hashlib.sha256(content).hexdigest(), "compressed_sha256": sha(dest)})
        else:
            dest = output / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            assert sha(src) == sha(dest)
            files.append({"source_relative": str(rel), "archive_relative": str(rel), "sha256": sha(src)})
    manifest = {"source_root": str(root), "excluded_subdirs": sorted(exclude), "files": files,
                "not_archived": skipped, "raw_sequences_included": False,
                "model_states_included": False, "test_predictions": False}
    (output / "archive_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(files), len(skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[], help="top-level subdirectories to leave out (e.g. runs in progress)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archived, skipped = archive(args.root, args.output, set(args.exclude))
    print(json.dumps({"output": str(args.output), "archived_files": archived, "not_archived": skipped}))


if __name__ == "__main__":
    main()
