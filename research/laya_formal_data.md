# Laya 正式实验数据视图

生成时间：2026-09-22T12:39:20.168787+00:00；固定划分 seed：`20260922`。

两个原始 JSONL 保持不变。脚本仅使用 Python 标准库；本阶段不运行模型、不计算 test 性能。

| 任务 | 原始 train / val / test | train | selection_dev | calibration | test view |
|---|---|---:|---:|---:|---:|
| promoter_detection | 16786 / 2109 / 2147 | 16766 | 1052 | 1053 | 2145 |
| fold_class | 15593 / 1881 / 1994 | 15593 | 939 | 942 | 1994 |

## 冻结规则

1. Normalize the final nonempty user-message line with strip and uppercase; preserve sequence orientation.
2. Validate all rows and stop on a schema/alphabet error rather than silently dropping malformed rows.
3. Primary identity is SHA-256 of the normalized exact sequence.
4. For DNA only, a partition group is SHA-256 of min(sequence, reverse_complement(sequence)); this does not imply label equivalence.
5. Using train+val only, exclude every exact sequence with conflicting labels and every DNA RC group with conflicting labels; never majority-vote or relabel.
6. Within each source split retain the first exact-sequence occurrence in source-line order; distinct RC sequences with consistent train+val labels can coexist in one group.
7. Retain train first, then exclude validation records whose exact/RC partition group is in retained train.
8. Split retained val at group level, stratified by canonical label, into approximately 50% selection_dev and 50% calibration using a fixed seed; each class appears in both.
9. Test is processed only after training and validation membership is fixed. Deduplicate test by exact sequence and exclude test group overlap with retained train/val; no test labels affect any exclusion or partition.
10. Test labels are converted to canonical indices only for an eventual locked evaluation; no test label histogram, conflict analysis, model evaluation, or performance access is performed here.
11. Raw files are never modified. Source provenance and historical clustering summaries are documented; no homology-independence claim is made.

## 清洗、反向互补与分层

### promoter_detection

- train+val exact 冲突组：2；额外 DNA RC 冲突组：0。
- 原始 exact 跨 split 重叠：`{"train__val": 0, "train__test": 0, "val__test": 0}`。
- 原始 partition group 跨 split 重叠：`{"train__val": 0, "train__test": 0, "val__test": 0}`。
- 删除计数：`{"train": {"same_split_exact_duplicate": 16, "train_val_exact_label_conflict": 4}, "val": {"same_split_exact_duplicate": 4}, "test": {"same_split_exact_duplicate": 2}}`。
- 最终 train / selection_dev / calibration / test 之间 partition group 重叠全部为 0。

| canonical label | 类名 | train | selection_dev | calibration |
|---:|---|---:|---:|---:|
| 0 | Non-promoter | 8395 | 521 | 521 |
| 1 | promoter | 8371 | 531 | 532 |

DNA 中包含两个不同方向序列的 RC group 为 0；当前 RC 检查没有引入额外删除。
### fold_class

- train+val exact 冲突组：0；额外 DNA RC 冲突组：0。
- 原始 exact 跨 split 重叠：`{"train__val": 0, "train__test": 0, "val__test": 0}`。
- 原始 partition group 跨 split 重叠：`{"train__val": 0, "train__test": 0, "val__test": 0}`。
- 删除计数：`{"train": {}, "val": {}, "test": {}}`。
- 最终 train / selection_dev / calibration / test 之间 partition group 重叠全部为 0。

| canonical label | 类名 | train | selection_dev | calibration |
|---:|---|---:|---:|---:|
| 0 | All Alpha | 2627 | 151 | 151 |
| 1 | All Beta | 3239 | 202 | 202 |
| 2 | Alpha and Beta | 4604 | 283 | 284 |
| 3 | Alpha plus Beta | 3754 | 217 | 218 |
| 4 | Multi-domain Proteins | 275 | 16 | 16 |
| 5 | Mixed Structures | 220 | 16 | 17 |
| 6 | Small Proteins and Peptides | 874 | 54 | 54 |

selection_dev 和 calibration 按类别采用固定 seed 的组哈希顺序及确定性子集和划分，尽量达到每类 50/50；奇数类多出的一条归 calibration。每个训练类别在两个验证子集都保留覆盖。

## 来源分组元数据与限制

当前两个 JSONL 只有任务、source、id、split 等字段，没有可复用的 donor/species/homology cluster ID。`source` 的 dna_train/dna_eva、prot_train/prot_eva 是来源批次标签，不是独立生物实体分组。

仓库带有历史 `data/03_sft_biopaws2/docs/leakage_stats.json` 和 `leakage_report.md`，记录了 MMseqs2 min-seq-id 0.5 / coverage 0.5 的汇总。该文件没有将每个当前输入哈希绑定到 cluster 的成员表，本次没有重跑或验证这些聚类。历史记录提示相似序列问题可能存在，不能把 exact/RC 无重叠写成同源独立。

## 文件与复现

`artifacts/laya_formal_data/manifest.json` 包含原始文件和输出 JSONL 的 SHA-256、行数、train/dev/calibration 类别计数、删除原因、来源统计和最终 overlap 断言。test 标签只被编码保存在 test view 中，没有类别直方图、冲突分析或性能访问。

每个 compact JSONL 记录字段：`id`, `task`, `modality`, `context`, `sequence`, `choices`, `label`, `source_split`, `split`, `sequence_sha256`, `group_id`。`label` 是该行 `choices` 的 0-based 索引；`context` 保留任务指令并移除重复候选列表。`group_id` 带 task 前缀，DNA 使用 RC canonical SHA-256，蛋白使用 exact SHA-256。

```bash
python scripts/prepare_laya_splits.py --seed 20260922
```
