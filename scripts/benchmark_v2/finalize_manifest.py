#!/usr/bin/env python3
"""Refresh manifest split counts from files after isolation, add quarantine stats."""
import json
from pathlib import Path
OUT = Path("/root/autodl-tmp/jev_gene/data/06_benchmark_v2_unified")
manifest = json.load(open(OUT / "manifest.json"))
iso = json.load(open(OUT / "isolation_report.json"))
for tid, info in manifest.items():
    for split in ("train", "dev", "test"):
        f = OUT / tid / f"{split}.jsonl"
        info[split] = sum(1 for l in open(f) if l.strip()) if f.exists() else 0
    if tid in iso:
        info["quarantined_groups"] = iso[tid]["quarantined_groups"]
        info["dup_dropped"] = iso[tid]["dropped_from_lower_splits"]
json.dump(manifest, open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=2)
print(f"refreshed {len(manifest)} tasks")
