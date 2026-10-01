#!/usr/bin/env python3
"""Download HF dataset snapshots with pinned revision + receipt (sha256, sizes)."""
import os, sys, json, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import snapshot_download
from huggingface_hub.hf_api import HfApi

ROOT = Path("/root/autodl-tmp/jev_gene/data/05_benchmark_v2_raw")
RECEIPT = ROOT / "download_receipt.json"

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(2**20), b""):
            d.update(c)
    return d.hexdigest()

def load_receipt():
    if RECEIPT.exists():
        return json.loads(RECEIPT.read_text())
    return {"session": "2026-10-01", "description": "MVP2 Benchmark v2 public dataset collection", "datasets": []}

def save_receipt(r):
    RECEIPT.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n")

def fetch(repo_id, subdir):
    api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=True)
    revision = info.sha
    license_ = "?"
    try:
        license_ = (info.card_data or {}).get("license", "?") if info.card_data else "?"
    except Exception:
        pass
    dest = ROOT / subdir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] downloading {repo_id} @ {revision[:10]} -> {subdir}", flush=True)
    local = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                              local_dir=str(dest), local_dir_use_symlinks=False)
    # inventory files
    files = []
    total = 0
    for f in sorted(dest.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            files.append({"path": str(f.relative_to(dest)), "bytes": sz, "sha256": sha256_file(f)})
    entry = {
        "repo_id": repo_id, "subdir": subdir, "revision": revision, "license": license_,
        "num_files": len(files), "total_bytes": total, "total_mb": round(total/(1024**2), 1),
        "download_time": datetime.now(timezone.utc).isoformat(), "files": files,
    }
    r = load_receipt()
    r["datasets"] = [d for d in r["datasets"] if d.get("repo_id") != repo_id or d.get("subdir") != subdir]
    r["datasets"].append(entry)
    r["last_updated"] = datetime.now(timezone.utc).isoformat()
    save_receipt(r)
    print(f"  ✓ {len(files)} files, {total/(1024**2):.1f} MB recorded", flush=True)
    return entry

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--subdir", required=True)
    a = ap.parse_args()
    fetch(a.repo, a.subdir)
