# Benchmark v2 预算缩放实验：更多更新反而更差（联合优化失稳，非算力瓶颈）

日期：2026-10-03。回应用户"加大预算看训练能否显著超 frozen #3"。代码同 `train_benchmark_v2.py`；结果 `artifacts/benchmark_v2_train/bv2_big1/`、`artifacts/benchmark_v2_eval/big1_{init,trained}/`、`budget_scaling_summary.json`。

## 设置（与 #1-small/full2 同任务集，仅加大预算）

| | #1-small (full2) | #1-big (big1) |
|---|---|---|
| 任务 | 43（cap PG=12/GUE=12）| 36（同 cap，8 个低于 min-valid 128 被 gate）|
| max/任务 | 768 | 2048 |
| epochs × bpf × 族 | 12×30×7 = **2520 upd** | 20×40×7 = **5600 upd** |
| 耗时 | ~36 min | ~81 min |

## 结果：加大预算 → 变差

分类（21 共同任务，accuracy）：

| family | init | #1-small | #1-big | #3 frozen |
|---|---|---|---|---|
| GUE(12) | 0.478 | 0.457 | 0.444 | 0.436 |
| GenomicBenchmarks(1) | 0.290 | 0.400 | 0.345 | 0.320 |
| dnagpt_pools(3) | 0.433 | 0.527 | 0.423 | 0.400 |
| local_snapshots(5) | 0.479 | 0.545 | 0.381 | 0.432 |
| **ALL(21)** | **0.463** | **0.485** | **0.421** | **0.424** |

score spearman：TAPE #1-small **0.237** → #1-big **−0.020**（崩溃）；ProteinGym 0.038→0.042；DeepSTARR 0.048→0.003。

**相对 init 的 mean Δ**：#1-small **+0.050** → #1-big **−0.013**（跌破 init）。

## 诊断：联合优化早停+失稳，不是过拟合

- big1 的 train loss 在 **~900/5600 步后平台在 1.06 不动**，grad 中位数从首段 1.03 掉到 **0.25 并持续到结束**——后 4700 步几乎无进展。
- 首段 grad_max **547**（35 步 >50，2 步 >100）——初期梯度尖峰/失稳。
- **不是过拟合**：train loss 没有→0；lg_fold_class 的 train loss 全程停在 **1.90→1.95（≈ln7 随机水平）**，即更多更新下 fold **根本没学**（而 full2 用更少更新把它降到 1.607）。
- 解读：36 个异质任务在 family-balance 下共享一个 encoder+scorer，初期大梯度把模型推入一个差的联合 basin，随后梯度消失、卡死；额外算力只是让它更稳地停在差解上。

## 结论

1. **"预算是瓶颈"假设被推翻**：2520→5600 更新、768→2048 数据/任务，结果不升反降。瓶颈是**多任务联合优化的稳定性/收敛**，不是算力。
2. **训练确实有用，但只需小预算**：#1-small(0.485) 已经**赢 frozen #3(0.424)** 约 +0.06、赢 init(0.463)。即"专门训练 vs 现成小 LM"的差距在小预算下已存在，加大 naive 预算反而抹平（big1 0.421 ≈ #3 0.424）。
3. **诚实的负面结果**：当前 family-balance 联合训练在任务数增多/预算加大时失稳，不能靠堆 epoch 解决。

## 下一步（针对失稳，而非堆算力）
- 稳定化：更小/分任务 LR、更紧 grad clip、更长 warmup、score 与分类 loss 尺度归一。
- 课程/分组：按模态或接口分组训练，减少异质任务同时拉扯共享 encoder。
- 早停：按 dev 选 checkpoint（当前用最终 epoch，big1 末段已退化）。
- 数据：小数据族（TAPE 2 任务）单独控制 epoch，避免相对过度优化。
- 修好后重跑 #1/#2/#3 matched，再看是否随预算单调改善。
