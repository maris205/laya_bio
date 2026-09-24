#!/usr/bin/env python3
"""Prepare deterministic Laya train, selection-dev, calibration, and test views.

Standard library only. Input JSONL files are read-only. Only train+val labels
participate in conflict removal and stratification. Test labels are serialized
as canonical indices but never counted, compared, scored, or used for decisions.
DNA reverse-complement grouping is partition hygiene, not semantic relabeling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re

from audit_laya_tasks import TASKS

SOURCE_SPLITS = ("train", "val", "test")
FINAL_SPLITS = ("train", "selection_dev", "calibration", "test")
COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN-", "TGCAYRSWMKVHDBN-")
SCHEMA = ["id", "task", "modality", "context", "sequence", "choices", "label",
          "source_split", "split", "sequence_sha256", "group_id"]
POLICY = [
    "Normalize the final nonempty user-message line with strip and uppercase; preserve sequence orientation.",
    "Validate all rows and stop on a schema/alphabet error rather than silently dropping malformed rows.",
    "Primary identity is SHA-256 of the normalized exact sequence.",
    "For DNA only, a partition group is SHA-256 of min(sequence, reverse_complement(sequence)); this does not imply label equivalence.",
    "Using train+val only, exclude every exact sequence with conflicting labels and every DNA RC group with conflicting labels; never majority-vote or relabel.",
    "Within each source split retain the first exact-sequence occurrence in source-line order; distinct RC sequences with consistent train+val labels can coexist in one group.",
    "Retain train first, then exclude validation records whose exact/RC partition group is in retained train.",
    "Split retained val at group level, stratified by canonical label, into approximately 50% selection_dev and 50% calibration using a fixed seed; each class appears in both.",
    "Test is processed only after training and validation membership is fixed. Deduplicate test by exact sequence and exclude test group overlap with retained train/val; no test labels affect any exclusion or partition.",
    "Test labels are converted to canonical indices only for an eventual locked evaluation; no test label histogram, conflict analysis, model evaluation, or performance access is performed here.",
    "Raw files are never modified. Source provenance and historical clustering summaries are documented; no homology-independence claim is made.",
]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_record(raw: dict, task: dict, line: int) -> dict:
    split = raw.get("split")
    if split not in SOURCE_SPLITS:
        raise ValueError(f"line {line}: unknown split {split!r}")
    if raw.get("task_id") != task["task_id"]:
        raise ValueError(f"line {line}: unexpected task_id")
    if task["modality"] not in raw.get("modality", []):
        raise ValueError(f"line {line}: wrong modality")
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError(f"line {line}: missing id")
    if raw.get("choices") != task["choices"]:
        raise ValueError(f"line {line}: unexpected choices or candidate order")
    if raw.get("answer_short") not in task["choices"]:
        raise ValueError(f"line {line}: unknown canonical label")
    users = [m.get("content") for m in raw.get("messages", [])
             if isinstance(m, dict) and m.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0], str):
        raise ValueError(f"line {line}: expected one string user message")
    lines = [part.strip() for part in users[0].splitlines() if part.strip()]
    if len(lines) < 2:
        raise ValueError(f"line {line}: cannot identify context and sequence")
    sequence = lines[-1].upper()
    if not sequence or not set(sequence) <= set(task["allowed"]):
        raise ValueError(f"line {line}: invalid sequence alphabet")
    context = re.split(r",?\s*The result will be one of the following\s*:",
                       "\n".join(lines[:-1]), maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,")
    if not context:
        raise ValueError(f"line {line}: empty context")
    if split != "test":
        assistants = [m.get("content") for m in raw.get("messages", [])
                      if isinstance(m, dict) and m.get("role") == "assistant"]
        if assistants != [raw["answer_short"]]:
            raise ValueError(f"line {line}: training/validation assistant target mismatch")
    seq_hash = digest(sequence)
    group_sequence = min(sequence, sequence.translate(COMPLEMENT)[::-1]) if task["modality"] == "dna" else sequence
    return {"id": raw["id"], "task": task["task_id"], "modality": task["modality"],
            "context": context, "sequence": sequence, "choices": list(task["choices"]),
            "label": task["choices"].index(raw["answer_short"]), "source_split": split,
            "split": split, "sequence_sha256": seq_hash,
            "group_id": f'{task["task_id"]}:{digest(group_sequence)}', "_line": line}


def label_counts(rows: list[dict]) -> dict[str, int]:
    if any(row["source_split"] == "test" for row in rows):
        raise ValueError("Test label aggregation is forbidden in split preparation")
    return {str(k): v for k, v in sorted(Counter(row["label"] for row in rows).items())}


def group_membership(rows: list[dict], key: str) -> dict[str, set[str]]:
    return {split: {row[key] for row in rows if row["source_split"] == split} for split in SOURCE_SPLITS}


def overlaps(rows: list[dict], key: str) -> dict[str, int]:
    members = group_membership(rows, key)
    return {f"{a}__{b}": len(members[a] & members[b]) for a, b in itertools.combinations(SOURCE_SPLITS, 2)}


def split_validation(rows: list[dict], task_id: str, n_labels: int, seed: int):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    by_label = defaultdict(list)
    for gid, members in groups.items():
        labels = {row["label"] for row in members}
        if len(labels) != 1:
            raise ValueError("Conflicting validation group survived cleaning")
        by_label[next(iter(labels))].append((gid, members))
    dev_ids, details = set(), {}
    for label in range(n_labels):
        items = sorted(by_label[label], key=lambda x: digest(f"{seed}:{task_id}:{label}:{x[0]}"))
        if len(items) < 2:
            raise ValueError(f"{task_id} class {label}: fewer than two validation groups")
        total = sum(len(members) for _, members in items)
        target = total // 2
        # Exact subset-sum up to floor(total/2). Record each reachable sum once;
        # deterministic hashed group order determines equally good solutions.
        reachable, mask, previous = 1, (1 << (target + 1)) - 1, {}
        for gid, members in items:
            weight = len(members)
            added = ((reachable << weight) & mask) & ~reachable
            while added:
                low = added & -added
                amount = low.bit_length() - 1
                previous[amount] = (amount - weight, gid)
                added ^= low
            reachable |= (reachable << weight) & mask
        amount = reachable.bit_length() - 1
        if amount == 0:
            raise ValueError(f"{task_id} class {label}: cannot preserve coverage in both halves")
        selected_count = amount
        while amount:
            amount, gid = previous[amount]
            dev_ids.add(gid)
        details[str(label)] = {"source_val_rows": total, "group_count": len(items),
                               "target_dev_rows": target, "selection_dev_rows": selected_count,
                               "calibration_rows": total - selected_count}
    dev, calibration = [], []
    for row in rows:
        split = "selection_dev" if row["group_id"] in dev_ids else "calibration"
        copy = dict(row, split=split)
        (dev if split == "selection_dev" else calibration).append(copy)
    assert set(label_counts(dev)) == set(label_counts(calibration)) == {str(i) for i in range(n_labels)}
    assert {r["group_id"] for r in dev}.isdisjoint(r["group_id"] for r in calibration)
    return dev, calibration, details


def write_jsonl(path: Path, rows: list[dict], *, compact_records: bool = True):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            obj = {key: row[key] for key in SCHEMA} if compact_records else row
            handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def historical_metadata(root: Path, name: str, current_rows: int):
    path = root / "data/03_sft_biopaws2/docs/leakage_stats.json"
    out = {"row_level_group_ids_found": False, "homology_independence_verified": False}
    if path.exists():
        historical = json.loads(path.read_text(encoding="utf-8")).get(name)
        out.update({"historical_summary_path": str(path.relative_to(root)),
                    "historical_summary_sha256": file_digest(path),
                    "historical_task_summary": historical,
                    "historical_row_count_matches": bool(historical and historical.get("n_rows") == current_rows),
                    "historical_summary_current_sequence_hashes_verified": False,
                    "scope": "Historical aggregate only; no row-level membership or per-input hash binding. Not used for partitioning, model choices, or claims of homology independence."})
    return out


def prepare_task(root: Path, outdir: Path, name: str, task: dict, seed: int):
    source_path = root / "data/03_sft_biopaws2/jsonl" / f"{name}.jsonl"
    input_hash = file_digest(source_path)
    rows, field_counts, source_counts, source_by_split = [], Counter(), Counter(), defaultdict(Counter)
    seen_ids = set()
    with source_path.open(encoding="utf-8") as handle:
        for line, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            raw = json.loads(raw_line)
            row = parse_record(raw, task, line)
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate source ID: {row['id']}")
            seen_ids.add(row["id"])
            rows.append(row)
            # Count field presence without concatenating string values into a
            # giant Counter payload (Counter.update(mapping) treats values as
            # counts).  This is provenance metadata only.
            field_counts.update(raw.keys())
            source_counts[str(raw.get("source"))] += 1
            source_by_split[row["source_split"]][str(raw.get("source"))] += 1
    trainval = [row for row in rows if row["source_split"] != "test"]
    exact_labels, group_labels, group_sequences = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in trainval:
        exact_labels[row["sequence_sha256"]].add(row["label"])
        group_labels[row["group_id"]].add(row["label"])
    for row in rows:
        group_sequences[row["group_id"]].add(row["sequence_sha256"])
    exact_conflicts = {key for key, labels in exact_labels.items() if len(labels) > 1}
    group_conflicts = {key for key, labels in group_labels.items() if len(labels) > 1}
    retained, exclusions = {}, []
    for split in SOURCE_SPLITS:
        selected, seen_sequences = [], set()
        earlier_groups = set()
        if split == "val":
            earlier_groups = {row["group_id"] for row in retained["train"]}
        elif split == "test":
            earlier_groups = {row["group_id"] for earlier in ("train", "val") for row in retained[earlier]}
        for row in rows:
            if row["source_split"] != split:
                continue
            reason = None
            if split != "test" and row["sequence_sha256"] in exact_conflicts:
                reason = "train_val_exact_label_conflict"
            elif split != "test" and row["group_id"] in group_conflicts:
                reason = "train_val_rc_group_label_conflict"
            elif row["group_id"] in earlier_groups:
                reason = "group_overlap_with_retained_train" if split == "val" else "group_overlap_with_retained_train_val"
            elif row["sequence_sha256"] in seen_sequences:
                reason = "same_split_exact_duplicate"
            if reason:
                exclusions.append({"id": row["id"], "source_split": split,
                                   "source_line": row["_line"], "sequence_sha256": row["sequence_sha256"],
                                   "group_id": row["group_id"], "reason": reason})
            else:
                selected.append(row)
                seen_sequences.add(row["sequence_sha256"])
        retained[split] = selected
    n_labels = len(task["choices"])
    assert set(label_counts(retained["train"])) == {str(i) for i in range(n_labels)}
    dev, calibration, stratification = split_validation(retained["val"], task["task_id"], n_labels, seed)
    outputs = {"train": retained["train"], "selection_dev": dev, "calibration": calibration, "test": retained["test"]}
    files = {}
    for split, records in outputs.items():
        path = outdir / f"{task['task_id']}_{split}.jsonl"
        write_jsonl(path, records)
        files[split] = {"path": str(path.relative_to(root)), "sha256": file_digest(path), "rows": len(records),
                        "unique_exact_sequences": len({row["sequence_sha256"] for row in records}),
                        "partition_groups": len({row["group_id"] for row in records}),
                        "ordered_ids_sha256": digest("\n".join(row["id"] for row in records)),
                        "sorted_sequence_hashes_sha256": digest("\n".join(sorted(row["sequence_sha256"] for row in records)))}
        if split != "test":
            files[split]["label_counts"] = label_counts(records)
    output_groups = {split: {row["group_id"] for row in records} for split, records in outputs.items()}
    final_overlaps = {f"{a}__{b}": len(output_groups[a] & output_groups[b])
                      for a, b in itertools.combinations(FINAL_SPLITS, 2)}
    assert not any(final_overlaps.values())
    exclusion_path = outdir / f"{task['task_id']}_exclusions.jsonl"
    write_jsonl(exclusion_path, exclusions, compact_records=False)
    assert file_digest(source_path) == input_hash, "Raw data unexpectedly changed"
    return {"task": task["task_id"], "modality": task["modality"], "choices": task["choices"],
            "input": {"path": str(source_path.relative_to(root)), "sha256": input_hash, "rows": len(rows),
                      "split_counts": dict(sorted(Counter(r["source_split"] for r in rows).items()))},
            "source_metadata": {"field_counts": dict(sorted(field_counts.items())),
                                "source_counts": dict(sorted(source_counts.items())),
                                "source_by_split": {s: dict(sorted(v.items())) for s, v in source_by_split.items()},
                                **historical_metadata(root, name, len(rows))},
            "audit": {"raw_cross_split_exact_overlap": overlaps(rows, "sequence_sha256"),
                      "raw_cross_split_partition_group_overlap": overlaps(rows, "group_id"),
                      "raw_groups_with_distinct_rc_sequences": sum(len(v) > 1 for v in group_sequences.values()) if task["modality"] == "dna" else None,
                      "train_val_exact_label_conflict_hashes": sorted(exact_conflicts),
                      "train_val_partition_label_conflict_group_ids": sorted(group_conflicts),
                      "train_val_additional_conflicting_rc_group_ids": sorted(gid for gid in group_conflicts if not (group_sequences[gid] & exact_conflicts)),
                      "final_cross_split_partition_group_overlap": final_overlaps,
                      "test_label_statistics_computed": False, "test_used_to_filter_train_val": False},
            "validation_stratification": stratification, "files": files,
            "exclusions": {"path": str(exclusion_path.relative_to(root)), "sha256": file_digest(exclusion_path),
                           "counts_by_source_split": {split: dict(sorted(Counter(row["reason"] for row in exclusions if row["source_split"] == split).items())) for split in SOURCE_SPLITS}}}


def markdown_report(manifest: dict) -> str:
    lines = ["# Laya 正式实验数据视图", "", f"生成时间：{manifest['generated_at_utc']}；固定划分 seed：`{manifest['seed']}`。", "",
             "两个原始 JSONL 保持不变。脚本仅使用 Python 标准库；本阶段不运行模型、不计算 test 性能。", "",
             "| 任务 | 原始 train / val / test | train | selection_dev | calibration | test view |", "|---|---|---:|---:|---:|---:|"]
    for task in manifest["tasks"].values():
        inp, files = task["input"], task["files"]
        lines.append(f"| {task['task']} | " + " / ".join(str(inp['split_counts'][s]) for s in SOURCE_SPLITS) + " | " + " | ".join(str(files[s]['rows']) for s in FINAL_SPLITS) + " |")
    lines += ["", "## 冻结规则", ""] + [f"{i}. {text}" for i, text in enumerate(POLICY, 1)]
    lines += ["", "## 清洗、反向互补与分层", ""]
    for task in manifest["tasks"].values():
        au = task["audit"]
        lines += [f"### {task['task']}", "", f"- train+val exact 冲突组：{len(au['train_val_exact_label_conflict_hashes'])}；额外 DNA RC 冲突组：{len(au['train_val_additional_conflicting_rc_group_ids'])}。",
                  f"- 原始 exact 跨 split 重叠：`{json.dumps(au['raw_cross_split_exact_overlap'], ensure_ascii=False)}`。",
                  f"- 原始 partition group 跨 split 重叠：`{json.dumps(au['raw_cross_split_partition_group_overlap'], ensure_ascii=False)}`。",
                  f"- 删除计数：`{json.dumps(task['exclusions']['counts_by_source_split'], ensure_ascii=False)}`。",
                  "- 最终 train / selection_dev / calibration / test 之间 partition group 重叠全部为 0。",
                  "", "| canonical label | 类名 | train | selection_dev | calibration |", "|---:|---|---:|---:|---:|"]
        for i, choice in enumerate(task["choices"]):
            lines.append(f"| {i} | {choice} | " + " | ".join(str(task['files'][s]['label_counts'].get(str(i), 0)) for s in FINAL_SPLITS[:-1]) + " |")
        if task["modality"] == "dna":
            lines += ["", f"DNA 中包含两个不同方向序列的 RC group 为 {au['raw_groups_with_distinct_rc_sequences']}；当前 RC 检查没有引入额外删除。"]
    lines += ["", "selection_dev 和 calibration 按类别采用固定 seed 的组哈希顺序及确定性子集和划分，尽量达到每类 50/50；奇数类多出的一条归 calibration。每个训练类别在两个验证子集都保留覆盖。", "",
              "## 来源分组元数据与限制", "",
              "当前两个 JSONL 只有任务、source、id、split 等字段，没有可复用的 donor/species/homology cluster ID。`source` 的 dna_train/dna_eva、prot_train/prot_eva 是来源批次标签，不是独立生物实体分组。", "",
              "仓库带有历史 `data/03_sft_biopaws2/docs/leakage_stats.json` 和 `leakage_report.md`，记录了 MMseqs2 min-seq-id 0.5 / coverage 0.5 的汇总。该文件没有将每个当前输入哈希绑定到 cluster 的成员表，本次没有重跑或验证这些聚类。历史记录提示相似序列问题可能存在，不能把 exact/RC 无重叠写成同源独立。", "",
              "## 文件与复现", "", "`artifacts/laya_formal_data/manifest.json` 包含原始文件和输出 JSONL 的 SHA-256、行数、train/dev/calibration 类别计数、删除原因、来源统计和最终 overlap 断言。test 标签只被编码保存在 test view 中，没有类别直方图、冲突分析或性能访问。", "",
              "每个 compact JSONL 记录字段：`" + "`, `".join(SCHEMA) + "`。`label` 是该行 `choices` 的 0-based 索引；`context` 保留任务指令并移除重复候选列表。`group_id` 带 task 前缀，DNA 使用 RC canonical SHA-256，蛋白使用 exact SHA-256。", "",
              "```bash", "python scripts/prepare_laya_splits.py --seed 20260922", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/laya_formal_data"))
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed, "validation_fraction_selection_dev": 0.5,
                "schema": SCHEMA, "policy": POLICY,
                "script_sha256": file_digest(Path(__file__).resolve()),
                "tasks": {task["task_id"]: prepare_task(root, outdir, name, task, args.seed) for name, task in TASKS.items()}}
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = root / "research/laya_formal_data.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(manifest), encoding="utf-8")
    for name, task in manifest["tasks"].items():
        print(json.dumps({"task": name, "rows": {s: task["files"][s]["rows"] for s in FINAL_SPLITS},
                          "exclusions": task["exclusions"]["counts_by_source_split"],
                          "final_partition_overlap": task["audit"]["final_cross_split_partition_group_overlap"]}))
    print(f"Wrote {outdir / 'manifest.json'} and {report_path}")


if __name__ == "__main__":
    main()
