#!/usr/bin/env python3
"""Audit Laya's two local pilot tasks without emitting biological sequences.

Standard library only. Raw JSONL files are read only. The proposed pilot split
is based on train/val only; the test split is audited but never used to decide
which train/val examples to keep. Exact sequence separation is not homology
separation. Outputs contain aggregate counts and SHA-256 hashes, not sequences.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path


TASKS = {
    "lg_promoter_detection": {
        "task_id": "promoter_detection", "modality": "dna",
        "canonical": "ACGT", "allowed": "ACGTRYSWKMBDHVN-",
        "choices": ["Non-promoter", "promoter"],
    },
    "lg_fold_class": {
        "task_id": "fold_class", "modality": "protein",
        "canonical": "ACDEFGHIKLMNPQRSTVWY", "allowed": "ACDEFGHIKLMNPQRSTVWYBXZJUO*-",
        "choices": ["All Alpha", "All Beta", "Alpha and Beta", "Alpha plus Beta",
                    "Multi-domain Proteins", "Mixed Structures", "Small Proteins and Peptides"],
    },
}
SPLITS = ("train", "val", "test")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sorted_counts(values):
    return dict(sorted(values.items()))


def length_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": values[0], "max": values[-1],
        "median": values[(len(values) - 1) // 2],
        "p95": values[int((len(values) - 1) * .95)],
        "p99": values[int((len(values) - 1) * .99)],
    }


def sequence_from_user(record):
    users = [m.get("content") for m in record.get("messages", [])
             if isinstance(m, dict) and m.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0], str):
        return None, None
    lines = [line.strip() for line in users[0].splitlines() if line.strip()]
    if len(lines) < 2:
        return None, None
    sequence = lines[-1].upper()
    # This is deliberately stricter than searching for the longest alphabetic
    # substring: there may be no silent extraction of just part of a sequence.
    if not sequence or not all(c.isascii() and (c.isalpha() or c in "*-") for c in sequence):
        return None, None
    return sequence, sha256("\n".join(lines[:-1]))


def audit_task(root, name, expected):
    path = root / "data/03_sft_biopaws2/jsonl" / f"{name}.jsonl"
    file_hash = hashlib.sha256()
    counts, errors, splits, alphabets, instructions, choice_variants = (Counter() for _ in range(6))
    labels_by_split = defaultdict(Counter)
    lengths_by_split = defaultdict(list)
    groups = defaultdict(list)
    row_hashes, identifiers = set(), set()

    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            file_hash.update(raw)
            counts["physical_lines"] += 1
            if not raw.strip():
                counts["blank_lines"] += 1
                continue
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                errors["json_parse_errors"] += 1
                continue
            if not isinstance(row, dict):
                errors["non_object_rows"] += 1
                continue
            counts["rows"] += 1
            row_errors = []
            split = str(row.get("split"))
            splits[split] += 1
            if split not in SPLITS:
                row_errors.append("unexpected_split")
            if row.get("task_id") != expected["task_id"]:
                row_errors.append("unexpected_task_id")
            if expected["modality"] not in row.get("modality", []):
                row_errors.append("missing_sequence_modality")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier:
                row_errors.append("invalid_identifier")
            else:
                counts["duplicate_id_extra_rows"] += identifier in identifiers
                identifiers.add(identifier)
            row_hash = sha256(json.dumps(row, sort_keys=True, separators=(",", ":")))
            counts["duplicate_full_extra_rows"] += row_hash in row_hashes
            row_hashes.add(row_hash)
            choices, gold = row.get("choices"), row.get("answer_short")
            choice_variants[json.dumps(choices, ensure_ascii=False)] += 1
            if choices != expected["choices"]:
                row_errors.append("choices_not_expected_order_and_values")
            if not isinstance(choices, list) or len(choices) != len(set(map(str, choices))):
                row_errors.append("choices_invalid_or_duplicated")
            if not isinstance(choices, list) or gold not in choices:
                row_errors.append("gold_not_in_choices")
            labels_by_split[split][str(gold)] += 1
            assistants = [m.get("content") for m in row.get("messages", [])
                          if isinstance(m, dict) and m.get("role") == "assistant"]
            if len(assistants) != 1 or assistants[0] != gold:
                row_errors.append("assistant_not_exactly_gold")
            sequence, instruction_hash = sequence_from_user(row)
            if sequence is None:
                row_errors.append("user_last_line_extraction_failure")
            else:
                counts["extracted_sequences"] += 1
                counts[f"extracted_sequences_{split}"] += 1
                instructions[instruction_hash] += 1
                alphabets.update(sequence)
                lengths_by_split[split].append(len(sequence))
                if not set(sequence) <= set(expected["allowed"]):
                    row_errors.append("unexpected_alphabet")
                if not set(sequence) <= set(expected["canonical"]):
                    counts["rows_with_noncanonical_characters"] += 1
                groups[sha256(sequence)].append({
                    "split": split, "gold": str(gold), "line": line_number,
                    "valid": not row_errors,
                })
            errors.update(row_errors)
            counts["valid_rows_before_duplicate_conflict_filter"] += not row_errors

    seq_sets = {split: {h for h, rows in groups.items() if any(r["split"] == split for r in rows)}
                for split in SPLITS}
    exact_duplicates = {}
    for split in SPLITS:
        repeated = {h: sum(r["split"] == split for r in rows) for h, rows in groups.items()
                    if sum(r["split"] == split for r in rows) > 1}
        exact_duplicates[split] = {
            "unique_sequences": len(seq_sets[split]), "repeated_sequence_hashes": len(repeated),
            "extra_rows": sum(n - 1 for n in repeated.values()),
            "rows_in_repeated_groups": sum(repeated.values()),
        }
    cross_split = {}
    for left, right in itertools.combinations(SPLITS, 2):
        overlap = seq_sets[left] & seq_sets[right]
        cross_split[f"{left}__{right}"] = {
            "shared_sequence_hashes": len(overlap),
            "left_rows": sum(sum(r["split"] == left for r in groups[h]) for h in overlap),
            "right_rows": sum(sum(r["split"] == right for r in groups[h]) for h in overlap),
            "conflicting_label_hashes": sum(
                len({r["gold"] for r in groups[h] if r["split"] in (left, right)}) > 1
                for h in overlap),
        }
    all_conflicts = {h for h, rows in groups.items() if len({r["gold"] for r in rows}) > 1}
    trainval_conflicts = {h for h, rows in groups.items()
                         if len({r["gold"] for r in rows if r["split"] in ("train", "val")}) > 1}
    conflicts_by_split = {}
    for split in SPLITS:
        hashes = {h for h, rows in groups.items()
                  if len({r["gold"] for r in rows if r["split"] == split}) > 1}
        conflicts_by_split[split] = {
            "conflicting_sequence_hashes": len(hashes),
            "rows_in_conflicting_groups": sum(sum(r["split"] == split for r in groups[h]) for h in hashes),
        }

    # Deterministic pilot view only. All train/val conflicts are removed, then
    # first valid occurrence per sequence is retained in each split; train wins
    # any remaining train/val exact overlap. No test labels influence this view.
    retained, pilot_details = {}, {}
    for split in ("train", "val"):
        candidates = sorted(((r["line"], h, r) for h, rows in groups.items()
                             for r in rows if r["split"] == split), key=lambda x: x[0])
        selected, excluded, selected_labels = {}, Counter(), Counter()
        for _, h, row in candidates:
            if not row["valid"]:
                excluded["invalid_rows"] += 1
            elif h in trainval_conflicts:
                excluded["train_val_label_conflict_rows"] += 1
            elif split == "val" and h in retained["train"]:
                excluded["exact_overlap_with_clean_train_rows"] += 1
            elif h in selected:
                excluded["same_split_exact_duplicate_rows"] += 1
            else:
                selected[h] = row
                selected_labels[row["gold"]] += 1
        retained[split] = selected
        pilot_details[split] = {
            "eligible_rows": len(selected), "labels": sorted_counts(selected_labels),
            "excluded_rows": sorted_counts(excluded),
            "retained_sequence_hash_list_sha256": sha256("\n".join(sorted(selected))),
        }

    return {
        "relative_path": str(path.relative_to(root)), "file_sha256": file_hash.hexdigest(),
        "expected_task": expected["task_id"], "expected_modality": expected["modality"],
        "expected_choices": expected["choices"], "counts": sorted_counts(counts),
        "errors": sorted_counts(errors), "split_rows": sorted_counts(splits),
        "labels_by_split": {s: sorted_counts(c) for s, c in labels_by_split.items()},
        "choice_variants": sorted_counts(choice_variants),
        "instruction_template_sha256_counts": sorted_counts(instructions),
        "character_counts": sorted_counts(alphabets),
        "unexpected_characters": sorted(set(alphabets) - set(expected["allowed"])),
        "lengths": length_summary([v for values in lengths_by_split.values() for v in values]),
        "lengths_by_split": {s: length_summary(v) for s, v in lengths_by_split.items()},
        "exact_duplicates": exact_duplicates, "cross_split_exact_overlap": cross_split,
        "label_conflicts": {
            "all_splits_conflicting_sequence_hashes": len(all_conflicts),
            "all_splits_conflicting_rows": sum(len(groups[h]) for h in all_conflicts),
            "within_split": conflicts_by_split,
            "train_val_conflicting_sequence_hashes": len(trainval_conflicts),
            "train_val_conflicting_hashes": sorted(trainval_conflicts),
        },
        "pilot": {
            "decision": "eligible_for_small_supervised_train_dev_pilot" if
                        all(pilot_details[s]["eligible_rows"] > 0 for s in ("train", "val"))
                        and not errors else "inspect_errors_before_pilot",
            "policy": ["Normalize user last non-empty line with strip + uppercase; hash with SHA-256.",
                       "Reject invalid rows and remove every sequence with conflicting labels in train+val.",
                       "Retain the first valid source-line occurrence per sequence in each split.",
                       "Remove validation sequences also present in the retained training split.",
                       "Test labels and test membership do not decide train/val exclusions.",
                       "Do not select examples by current model loss or validation performance."],
            "splits": pilot_details,
            "post_cleanup_train_val_shared_hashes": len(set(retained["train"]) & set(retained["val"])),
            "test_used_for_pilot_filter": False,
            "homology_independence_verified": False,
        },
    }


def markdown_report(report):
    lines = ["# Laya 两任务数据审计", "", f"审计时间：{report['generated_at_utc']}", "",
             "只读扫描两个原始 JSONL；不修改数据、不安装依赖、不使用 GPU。结果文件只保存汇总与 SHA-256 哈希。",
             "", "## 输入与标签", "",
             "| 任务 | 总行数 | train / val / test | 末行序列覆盖 | 序列长度 min / median / p95 / max | 数据格式错误 |",
             "|---|---:|---|---:|---|---:|"]
    for name, task in report["tasks"].items():
        c, le = task["counts"], task["lengths"]
        lines.append(f"| {name} | {c['rows']:,} | " + " / ".join(f"{task['split_rows'].get(s, 0):,}" for s in SPLITS)
                     + f" | {c.get('extracted_sequences', 0):,}/{c['rows']:,} | "
                     + " / ".join(str(le[k]) for k in ("min", "median", "p95", "max"))
                     + f" | {sum(task['errors'].values())} |")
    lines += ["", "长度单位为 DNA 碱基或蛋白氨基酸字符，并非 tokenizer token。"
              "检查 choices 顺序和值、gold 属于 choices、assistant 与 gold 完全一致、task_id、modality、split 和字母表。", "",
              "## 重复、跨划分重叠与冲突", "",
              "| 任务 | train / val / test 内重复额外行 | train–val / train–test / val–test 共享序列 | 全部划分同序列多标签 | train+val 同序列多标签 |",
              "|---|---|---|---:|---:|"]
    for name, task in report["tasks"].items():
        dup, overlap, conflicts = task["exact_duplicates"], task["cross_split_exact_overlap"], task["label_conflicts"]
        lines.append(f"| {name} | " + " / ".join(str(dup[s]["extra_rows"]) for s in SPLITS)
                     + " | " + " / ".join(str(overlap[p]["shared_sequence_hashes"]) for p in ("train__val", "train__test", "val__test"))
                     + f" | {conflicts['all_splits_conflicting_sequence_hashes']} | {conflicts['train_val_conflicting_sequence_hashes']} |")
    lines += ["", "共享序列和标签冲突均按规范化序列 SHA-256 计数；同序列标签冲突不通过多数投票修复。", "",
              "## 少量监督 pilot 的建议视图", "",
              "先从 train+val 中完全删除标签冲突序列，再在各 split 内保留同序列第一条有效原始行，最后从 val 删除与清洗后 train 完全重复的序列。"
              "这个规则只使用 train/val 信息；test 不参与 pilot 过滤、调参或结果选择。原始数据不改动。", "",
              "| 任务 | 清洗后 train | 清洗后 val | train 删除行 | val 删除行 | 清洗后 train–val 共享序列 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, task in report["tasks"].items():
        pilot = task["pilot"]
        tr, va = pilot["splits"]["train"], pilot["splits"]["val"]
        lines.append(f"| {name} | {tr['eligible_rows']:,} | {va['eligible_rows']:,} | "
                     f"{sum(tr['excluded_rows'].values()):,} | {sum(va['excluded_rows'].values()):,} | "
                     f"{pilot['post_cleanup_train_val_shared_hashes']} |")
    lines += ["", "建议对上述 train/dev 视图固定 seed 后按类别抽取小批量，验证训练 loss、数值稳定性和词表扩展后的输入兼容性。"
              "小样本 pilot 不能提供论文最终效能结论。", "", "## 范围限制", "",
              "- 本次验证的是完全相同序列的重复；没有执行同源聚类、相似性搜索或 DNA 反向互补等价检查，不能声称同源独立。",
              "- 未验证来源许可、原始采样方式和生物标签是否正确；文件中的 license 字段不等于独立许可核验。",
              "- 未使用 tokenizer，本报告不保证 1024 token 内完整覆盖；需要另行验证候选、指令和序列共同占用的长度。",
              "- 全划分冲突仅作审计统计；pilot 清洗不使用 test 标签。正式论文评估仍需独立确定泄漏控制与标签冲突处理方案。",
              "", "复现命令：", "", "```bash", "python scripts/audit_laya_tasks.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    report = {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_hash_normalization": "single user message, last nonempty line, strip, uppercase, SHA-256 UTF-8",
        "tasks": {name: audit_task(root, name, expected) for name, expected in TASKS.items()},
    }
    json_path = root / "artifacts/laya_two_task_audit.json"
    markdown_path = root / "research/laya_two_task_audit.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown_path.write_text(markdown_report(report))
    for name, task in report["tasks"].items():
        pilot = task["pilot"]
        print(json.dumps({"task": name, "rows": task["counts"]["rows"], "errors": task["errors"],
                          "pilot": {split: pilot["splits"][split]["eligible_rows"] for split in ("train", "val")},
                          "train_val_conflict_hashes": task["label_conflicts"]["train_val_conflicting_sequence_hashes"],
                          "decision": pilot["decision"]}, ensure_ascii=False))
    print(f"Wrote {json_path.relative_to(root)} and {markdown_path.relative_to(root)}")


if __name__ == "__main__":
    main()
