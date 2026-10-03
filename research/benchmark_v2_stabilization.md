# Benchmark v2 稳定化 + dev 早停：修复大预算退化，但确认平台

日期：2026-10-03。回应"预算缩放负结果"的诊断（big1 失稳）。代码：`train_benchmark_v2.py` 加 `--warmup-frac/--clip/--head-lr-scale/--dev-eval-every/--dev-max` + best-dev 早停。结果 `artifacts/benchmark_v2_train/bv2_stab/`、`artifacts/benchmark_v2_eval/stab_{init,trained}/`、`stabilization_summary.json`。

## 改动（针对 big1 失稳诊断）

1. **更长 warmup** 0.05→0.15（big1 首段 grad_max 547 尖峰）。
2. **更紧 grad clip** 1.0→0.5。
3. **dev 早停**：每 200 更新在 dev（96/任务）算 mean 指标（choice/noul=acc，score=(spearman+1)/2），保存 best-dev checkpoint 为 `model.safetensors`，末 epoch 存为 `model_final.safetensors`。
4. 同 big1 预算（5600 updates、max 2048/任务、cap PG=12/GUE=12），37 任务（7 gate）。

## 结果：失稳被修复，但出现真平台

分类（21 共同任务，accuracy）：

| | init | #1-small(2520,无稳定) | big1(5600,无稳定) | **#1-stab(5600,稳定+早停)** | #3 frozen |
|---|---|---|---|---|---|
| ALL 分类 | 0.463 | 0.485 | **0.421**(退化) | **0.481**(修复) | 0.424 |
| GUE(12) | 0.478 | 0.457 | 0.444 | 0.466 | 0.436 |
| dnagpt(3) | 0.433 | 0.527 | 0.423 | 0.497 | 0.400 |
| local(5) | 0.479 | 0.545 | 0.381 | 0.542 | 0.432 |

score spearman：TAPE big1 −0.020 → stab **0.071**（修复，但仍 < small 0.237）；DeepSTARR small 0.048 → stab 0.079；ProteinGym small 0.038 → stab 0.022。

相对 init mean Δ：#1-small **+0.050**、#1-stab **+0.049**（持平）。

**早停轨迹**：best-dev 在 **step 4800（dev 0.4949）**；dev 曲线 5000→0.474、5200→0.479、5400→0.485、5600→0.477，末段平稳波动**未崩溃**（对比 big1 末段 test 退化到 0.421）。

## 判读

1. **big1 的退化确因优化失稳**：warmup0.15 + clip0.5 + dev 早停把 5600-更新的大预算从 0.421（低于 init）拉回 **0.481**（≈ small、高于 init 与 frozen #3）。诊断正确、修复有效。
2. **但算力不再是瓶颈，出现真平台**：稳定后 stab(0.481, +0.049) ≈ small(0.485, +0.050)，5600 更新并未超过 2520 更新。即 ~2500 更新后性能饱和，堆 epoch 无用——与"预算缩放"结论一致，且现在排除了失稳这一混淆。
3. **训练稳定优于 frozen #3**（0.48 vs 0.42），专门训练的价值在小预算即已体现。
4. **天花板 = joint 多任务容量/数据**，不是算力也不是稳定性。TAPE score 在 stab（0.071）仍低于 small（0.237）提示：早停按"全任务平均 dev"选点，可能牺牲个别族的最优点（小数据族 TAPE 的最优 step 与全局 best 不一致）。

## 局限与下一步
- 早停用**单一全局 dev 分数**选一个 checkpoint，对异质任务是折中；按族/按任务早停或每任务保留各自 best 可能更好（尤其 score 小数据族）。
- 平台期要突破需：更多数据/任务、任务分组或课程、per-task 专用化、或更大 backbone——非更多 epoch。
- 未做：#2 任务头的稳定化版对照；noul 校准修复；全 265 任务。
