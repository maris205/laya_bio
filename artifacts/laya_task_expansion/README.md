# Task expansion source audits

These are read-only provenance/schema/membership checks, not model results or a frozen training release. Raw source data and model weights are not stored here.

- `admission_audit.json`: the initial eight local single-sequence files. Its proposed roles describe the local-only stage before upstream training pools were discovered; the current 12-task plan supersedes those proposed roles. Test target values were not examined by that script.
- `upstream_datasets.json`, `source_cards/`, `download_receipt.json`: pinned public Hugging Face metadata, source cards and downloaded file hashes. The cards contain limited ontology information.
- `upstream_audit.json`: public annotation mapping, duplicates, old split membership, paired sequence extraction, endpoint overlap, and an explicit standard-code translation diagnostic. This script does inspect public annotation values for source auditing; it does not compute model scores.

Run from the project root with Python, PyArrow and Biopython installed. Substitute your downloaded source directories; their original local locations are recorded in the receipt, not required paths:

```sh
python scripts/audit_laya_task_expansion.py --jsonl-root /path/to/local/jsonl --output artifacts/laya_task_expansion/admission_audit.json
python scripts/audit_laya_upstream_tasks.py --source-root /path/to/05_task_extension_sources --legacy-jsonl /path/to/local/jsonl --output artifacts/laya_task_expansion/upstream_audit.json
```

A single Hugging Face split named train is not a validated training split: all four DNA pools include old test members. Endpoint exact deduplication is not protein homology clustering. Full-input model-tokenizer checks, conflict resolution, source/annotation verification, and a global split registry are still admission gates. See the [task catalog](../../research/biological_decision_task_catalog.md) and [experiment plan](../../refine-logs/EXPERIMENT_PLAN.md).
