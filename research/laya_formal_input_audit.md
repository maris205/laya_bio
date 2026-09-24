# Laya 正式输入长度与分词审计

本审计在 CPU 上运行，读取冻结的正式数据视图和 M1/M2 表示；不修改原始 JSONL，也不拟合词表或使用 test 标签决定任何表示。

- `max_len`: **1024**；`head_max_len`: **256**。
- 表示目录：`artifacts/laya_formal_representation`；正式数据清单：`artifacts/laya_formal_data`。
- `build_sequence` 使用仓库中的官方实现；M2 与 M1 的状态文本相同，序列由 train-only source BPE pieces 以 `▶piece◀` / `◆piece◇` 包裹。

## 共同完整覆盖

下表的 complete 是两种表示都未超过 1024、未触发 head/state/output 截断的共同集合。

| task | split | rows | common complete | rate |
|---|---:|---:|---:|---:|
| promoter_detection | train | 16766 | 16766 | 1.0000 |
| promoter_detection | selection_dev | 1052 | 1052 | 1.0000 |
| promoter_detection | calibration | 1053 | 1053 | 1.0000 |
| promoter_detection | test | 2145 | 2145 | 1.0000 |
| fold_class | train | 15593 | 15284 | 0.9802 |
| fold_class | selection_dev | 939 | 926 | 0.9862 |
| fold_class | calibration | 942 | 933 | 0.9904 |
| fold_class | test | 1994 | 1952 | 0.9789 |

## M1/M2 长度

p50/p95/p99/max 是按 full input length（head 已按 256 预算、state 未按 1024 截断）计算；actual 是官方 builder 输出长度。

### promoter_detection

| condition | split | full p50 | p95 | p99 | max | >1024 | head over budget | state trunc. | output trunc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1_base | train | 385 | 401 | 408 | 423 | 0 | 0 | 0 | 0 |
| M1_base | selection_dev | 391 | 407 | 413 | 423 | 0 | 0 | 0 | 0 |
| M1_base | calibration | 392 | 407 | 413 | 422 | 0 | 0 | 0 | 0 |
| M1_base | test | 392 | 408 | 415 | 424 | 0 | 0 | 0 | 0 |
| M2_expanded | train | 369 | 388 | 395 | 406 | 0 | 0 | 0 | 0 |
| M2_expanded | selection_dev | 373 | 392 | 400 | 404 | 0 | 0 | 0 | 0 |
| M2_expanded | calibration | 373 | 393 | 400 | 409 | 0 | 0 | 0 | 0 |
| M2_expanded | test | 373 | 392 | 399 | 414 | 0 | 0 | 0 | 0 |

Prompt-only tokenization changed **0 / 21016** rows (0.000000); exact changed-row list is empty when this is zero.

- `M1_base:over_1024` exact rows: **0**.
- `M1_base:head_over_budget` exact rows: **0**.
- `M1_base:head_truncated` exact rows: **0**.
- `M1_base:state_truncated` exact rows: **0**.
- `M1_base:output_truncated` exact rows: **0**.
- `M2_expanded:over_1024` exact rows: **0**.
- `M2_expanded:head_over_budget` exact rows: **0**.
- `M2_expanded:head_truncated` exact rows: **0**.
- `M2_expanded:state_truncated` exact rows: **0**.
- `M2_expanded:output_truncated` exact rows: **0**.
### fold_class

| condition | split | full p50 | p95 | p99 | max | >1024 | head over budget | state trunc. | output trunc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1_base | train | 394 | 884 | 1099 | 1228 | 309 | 0 | 309 | 309 |
| M1_base | selection_dev | 392 | 851 | 1054 | 1201 | 13 | 0 | 13 | 13 |
| M1_base | calibration | 386 | 849 | 1014 | 1202 | 9 | 0 | 9 | 9 |
| M1_base | test | 392 | 890 | 1098 | 1214 | 42 | 0 | 42 | 42 |
| M2_expanded | train | 368 | 823 | 1022 | 1163 | 151 | 0 | 151 | 151 |
| M2_expanded | selection_dev | 365 | 789 | 984 | 1154 | 6 | 0 | 6 | 6 |
| M2_expanded | calibration | 361 | 786 | 948 | 1127 | 3 | 0 | 3 | 3 |
| M2_expanded | test | 367 | 828 | 1023 | 1123 | 19 | 0 | 19 | 19 |

Prompt-only tokenization changed **0 / 19468** rows (0.000000); exact changed-row list is empty when this is zero.

- `M1_base:over_1024` exact rows: **373**.
- `M1_base:head_over_budget` exact rows: **0**.
- `M1_base:head_truncated` exact rows: **0**.
- `M1_base:state_truncated` exact rows: **373**.
- `M1_base:output_truncated` exact rows: **373**.
- `M2_expanded:over_1024` exact rows: **179**.
- `M2_expanded:head_over_budget` exact rows: **0**.
- `M2_expanded:head_truncated` exact rows: **0**.
- `M2_expanded:state_truncated` exact rows: **179**.
- `M2_expanded:output_truncated` exact rows: **179**.

## Raw alphabet checks

- **promoter_detection**: 21042 raw rows; unexpected characters `{}` across 0 rows.
  DNA `N`: 1 rows (0.00004752); exact row references are recorded in JSON without sequence text.
  Allowed noncanonical characters: `{'N': 1}`.
- **fold_class**: 19468 raw rows; unexpected characters `{}` across 0 rows.
  Allowed noncanonical characters: `{'B': 1, 'X': 329, 'Z': 1}`.

## Truncation and row references

Exact exception rows (IDs, source split, and sequence hashes) are in `artifacts/laya_formal_input_audit.json`; no biological sequence strings are emitted. Common eligible IDs are also saved to `artifacts/laya_formal_data/eligible_ids.json`.


## 独立复核

`python scripts/audit_laya_formal_inputs.py` 的 v2 复核重新计算所有正式输入，确认与原审计的共同完整 ID 列表逐条一致，检查完整候选/指令、source BPE 序列还原和自然语言分词隔离；额外输出 train/dev/calibration 的逐类保留率。复核文件：[laya_formal_input_audit_v2.json](../artifacts/laya_formal_input_audit_v2.json)。原始清单和训练正在使用的 ID 文件保持不变。
