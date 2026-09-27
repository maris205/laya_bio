# JEV 联合训练路径诊断结果

完成记录日期：2026-09-26 UTC。实验目录为 [`artifacts/laya_jev_joint_path_diagnostics`](../artifacts/laya_jev_joint_path_diagnostics)，冻结计划为 [`EXPERIMENT_PLAN.md`](../artifacts/laya_jev_joint_path_diagnostics/EXPERIMENT_PLAN.md)，机器可读汇总为 [`analysis/summary.json`](../artifacts/laya_jev_joint_path_diagnostics/analysis/summary.json)。

## 结论

**seed 20260928 NO-CPT 的 GFP 近常数失败可由联合训练路径触发，也可仅通过改变任务路径恢复。** 在不改数据、seed、NO-CPT 初始化、JEV 输出、GFP 曝光、optimizer、LR schedule、loss 和评估的前提下，把每个 epoch 内的随机 task interleave 改为 `promoter -> structural_class -> fluorescence` 任务块（`task_block_gfp_last`），GFP dev 从 RMSE/MAE/Spearman/预测 SD 0.8345/0.6241/0.1929/0.0127 恢复到 0.5953/0.3578/0.5588/0.5422，并通过两个门槛。seed 20260927 上同一路径也恢复（0.5791/0.3142/0.6059/0.5485）。

把 GFP 块放在最前（`task_block_gfp_first`）只部分恢复（0.8296/0.4935/0.3785/0.1552），支持"GFP 最后更新 / recency"是恢复因素之一，而不是任意任务块都充分。

在 GFP-last 下把 fluorescence loss 乘以 0.5 或 2.0，GFP 均恢复并过门槛，且对权重非单调（×1.0 最好）。在 0.5–2 倍范围内，loss 权重不是控制因素。

## 末轮结果（seed 20260928 NO-CPT，dev）

冻结常数基线：GFP 训练均值 RMSE 0.8357，训练中位数 MAE 0.5095。预测 SD < 0.05 预标记为近常数输出。

| Run | GFP RMSE ↓ | GFP MAE ↓ | Spearman ↑ | 预测 SD | train SD | Gate | Promoter Acc | Structure Acc / Macro-F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 原 interleaved joint | 0.8345 | 0.6241 | 0.1929 | 0.0127 | 0.0121 | fail | 88.59% | 56.55% / 0.4303 |
| GFP-only 单任务 | 0.6303 | 0.4781 | 0.5392 | 0.5637 | 0.5562 | pass | — | — |
| gfp_first ×1.0 | 0.8296 | 0.4935 | 0.3785 | 0.1552 | 0.1592 | pass（弱） | 88.02% | 58.15% / 0.4507 |
| gfp_last ×0.5 | 0.6610 | 0.4256 | 0.5117 | 0.4344 | 0.4090 | pass | 84.89% | 58.79% / 0.4499 |
| gfp_last ×1.0 | **0.5953** | **0.3578** | **0.5588** | 0.5422 | 0.5487 | pass | 86.69% | 54.21% / 0.4511 |
| gfp_last ×2.0 | 0.6298 | 0.4017 | 0.5292 | 0.4789 | 0.4805 | pass | 87.55% | 58.47% / 0.4420 |

seed 20260927 NO-CPT：原 interleaved joint 0.6910/0.5279（MAE fail）；gfp_last ×1.0 为 0.5791/0.3142/0.6059/0.5485（pass），promoter 88.78%，structure 57.29% / 0.4394。

## 方法学限制：GPU 非确定性

`task_block_gfp_last_fluo0p5` 首次运行（`run_weighting_v2`）在 step 992/1152 被外部终止（无 traceback，会话结束），在 `run_weighting_v3` 以同一 runner 完整重跑。两次运行的训练轨迹在前 137 步逐位一致（同任务顺序、同 loss），之后出现浮点级差异并放大；epoch-2 GFP dev 预测 SD 分别为 0.058 与 0.198。因此同配置 run-to-run 方差不可忽略，表中所有变体比较均为单 run。gfp_first 与 gfp_last 的差距是当前"recency"解释的主要证据，需要重复运行确认。`run_weighting_v2` 保留为不完整记录，不纳入汇总。

## Audit Notes

- 所有纳入汇总的 formal run：1,152 updates、每任务 384 updates、精确 checkpoint 重载、模型保留且哈希一致、无 test inference。
- 每个变体都先通过 3-update smoke（三任务均有有限非零梯度）。
- runner 在启动前核验训练数据 manifest、初始模型、CPT 模型和冻结 trainer 的哈希与原多任务轮一致。

## Replicate 轮结果（2026-09-27）

seed 20260928 NO-CPT，`run_replicate` 中每个顺序新增 2 次同配置重跑（4 个 formal run 均完成 1,152 updates、精确重载、模型保留、无 test inference），与原 run 合计 n = 3。

| 顺序 | Run | GFP RMSE ↓ | GFP MAE ↓ | Spearman | 预测 SD | Gate | Promoter Acc | Structure Acc / Macro-F1 |
|---|---|---:|---:|---:|---:|---|---:|---:|
| GFP-last | original | 0.5953 | 0.3578 | 0.5588 | 0.5422 | pass | 86.69% | 54.21% / 0.4511 |
| GFP-last | rep1 | 0.5317 | 0.3150 | 0.5811 | 0.6200 | pass | 87.83% | 54.21% / 0.4391 |
| GFP-last | rep2 | 0.5890 | 0.4393 | 0.5593 | 0.6314 | pass | 87.55% | 55.38% / 0.4339 |
| GFP-first | original | 0.8296 | 0.4935 | 0.3785 | 0.1552 | pass（弱） | 88.02% | 58.15% / 0.4507 |
| GFP-first | rep1 | 0.8617 | 0.5121 | 0.3252 | 0.0790 | fail | 87.93% | 58.79% / 0.4547 |
| GFP-first | rep2 | 0.8291 | 0.6441 | 0.3117 | **0.0216** | fail | 89.26% | 57.40% / 0.4576 |

| 汇总（n = 3） | RMSE 均值 [范围] | MAE 均值 [范围] | Gate 通过率 |
|---|---|---|---|
| GFP-last | 0.5720 [0.5317, 0.5953] | 0.3707 [0.3150, 0.4393] | 3/3 |
| GFP-first | 0.8401 [0.8291, 0.8617] | 0.5499 [0.4935, 0.6441] | 1/3 |

判读（按冻结规则）：所有 GFP-last run 在 GFP RMSE 与 MAE 上均优于所有 GFP-first run，且两指标的均值差（RMSE 0.268、MAE 0.179）都超过组内极差（RMSE ≤ 0.064、MAE ≤ 0.151），规则输出 `reproducible_gfp_last_effect`。GFP-first 的 3 次中 2 次不过门槛，rep2 近常数（SD 0.0216），即任务顺序在同 seed 下可复现地决定 GFP 是否 collapse。

附带观察：分类任务呈镜像模式。GFP-first 的块顺序为 fluorescence → promoter → structural_class，structure 位于最后，其 structure Accuracy（57.4–58.8%）稳定高于 GFP-last（54.2–55.4%），promoter 也略高。这更像"每个 epoch 最后训练的任务获益"的一般 recency／遗忘效应，而非 GFP 特有；GFP 只是对被后续任务覆盖最敏感。限制：单 seed（20260928）、复用 dev、GFP 共享亲本；n = 3 的极差不是统计显著性检验。

## 下一步

GFP-last recency 效应已按预设规则确认为可复现。loss weighting 方向暂停。下一轮应解释为什么"最后训练"关键，同时不牺牲其他任务：

1. **Optimizer-state 隔离**：GFP-last 路径下为每个任务使用独立 AdamW 状态（共享参数），检验 recency 依赖是否来自跨任务的动量／二阶矩污染。
2. **梯度冲突 probe**：在固定 checkpoint 上测 GFP 与分类任务梯度的余弦，定位竞争集中在 encoder 还是 scorer。
3. **实用 recipe 候选**：在每个 epoch 末尾追加短的混合 replay 或 round-robin 细粒度块，目标是三任务都不依赖处在最后位置。

每个候选均需同配置至少 n = 2 重复，才能与 run-to-run 噪声区分。

## 任务干扰探针（2026-09-27）

脚本 [`scripts/probe_jev_task_interference.py`](../scripts/probe_jev_task_interference.py)，结果 [`interference_probe/result.json`](../artifacts/laya_jev_joint_path_diagnostics/interference_probe/result.json)。只用训练集（固定 train 诊断面板，每任务 1,024 条），无 dev、无 test 推理。

**A. 梯度余弦**（每任务 4 个固定 epoch-1 batch，eval 模式，按参数组）：

| Checkpoint | GFP 训练偏差 | scorer GFP·promoter / GFP·structure | encoder GFP·promoter / GFP·structure |
|---|---:|---|---|
| 初始模型 | −0.283 | −0.999 / −0.995 | −0.10 / +0.04 |
| 原 interleaved（GFP 塌缩） | +0.026 | −0.998 / −0.999 | +0.02 / −0.09 |
| gfp_last | +0.041 | −0.82 / −0.72 | −0.15 / −0.18 |
| gfp_last rep1 | +0.023 | −0.98 / −0.96 | −0.07 / +0.00 |
| gfp_first | +0.252 | +0.998 / +0.949 | −0.02 / −0.05 |

scorer 梯度被单一方向主导（余弦总在 ±1 附近），符号随 GFP 当前状态变化，与 GFP 最终好坏没有稳定对应。encoder 上的任务梯度近似正交。静态余弦不足以解释塌缩。

**B. 分类-only 遗忘测试**：从 GFP 良好的 gfp_last checkpoint 出发，重放 gfp_first 在 epoch 3 中 GFP 块之后的真实 batch 与 LR 位置（promoter 128 步 + structure 128 步），全新 AdamW。

| 起点 / 可训练范围 | GFP RMSE/MAE/SD：0 → 128 → 256 步 | Promoter Acc | Structure Acc |
|---|---|---|---|
| gfp_last / 全部 | 0.581/0.346/0.549 → 0.586/0.349/0.542 → 0.675/0.352/0.502 | 90.6 → 94.8 → 94.0% | 66.8 → 66.2 → 76.4% |
| gfp_last rep1 / 全部 | 0.507/0.291/0.618 → 0.512/0.277/0.639 → 0.520/0.270/0.636 | 92.8 → 96.0 → 94.7% | 68.7 → 68.1 → 76.7% |
| gfp_last / 仅 encoder | 0.581/0.346/0.549 → 0.590/0.346/0.542 → 0.679/0.356/0.488 | 90.6 → 96.6 → 96.6% | 66.8 → 66.9 → 76.7% |
| gfp_last / 仅 head+scorer | 0.581/0.346/0.549 → 0.582/0.360/0.529 → 0.581/0.347/0.547 | 不变 | 不变 |

判读：从良好 GFP 状态出发，GFP 块之后的 256 步分类更新**不会**把 GFP 压成常数（SD 始终 ≥ 0.49）；一个起点 RMSE 升 0.09（来自 structure 阶段、经 encoder），另一个起点无退化。因此"GFP 训练好后被后续任务遗忘"不是 GFP-first 失败的主要机制，此前的 recency 表述需要修正：更可能是 GFP-first 的 GFP 块本身没有把 GFP 训练到良好状态（两种顺序中 GFP 块都紧接 structure 块，唯一结构差异是 epoch 1：GFP-first 让 GFP 从初始模型、在 LR warmup 中先训练）。但 GFP 单任务同样从初始模型起步且在该 seed 恢复，这一点仍只是待检验假设。

限制：epoch-3 LR 已接近下限，head/scorer 有效 LR 约 1e-5，仅 head+scorer 组连分类任务也不动，所以无法检验 scorer 冲突在高 LR 早期阶段的作用；使用全新 AdamW 而非真实 optimizer 状态；单 seed。

**下一步建议**：给两种顺序各跑一次带"每个任务块后 GFP 训练诊断"的 run（只增评估钩子，训练不变），直接看 GFP-first 的 GFP 块是否曾达到良好状态、塌缩发生在哪个 epoch/块；若 epoch-1 是关键，再做"仅 epoch 1 用 GFP-last、之后用 GFP-first"的交换对照。

## 逐块诊断（2026-09-27）

冻结计划见 `EXPERIMENT_PLAN.md` 末节；输出 `run_blockeval`。训练与原 gfp_first / gfp_last 完全相同，仅在每个 128 步任务块后于固定训练诊断面板（每任务 1,024 条）评估一次（eval 模式、无 RNG、无 optimizer 改动）。两个 run 均 1,152 updates、精确重载、无 test inference；末轮 dev GFP RMSE/MAE/SD：gfp_first 0.8151/0.5416/0.121（fail），gfp_last 0.5260/0.3537/0.647（pass），与此前 n = 3 结果一致，可视为各顺序第 4 次重复。

| 顺序 | epoch-块 | 刚训练的任务 | GFP RMSE | GFP MAE | GFP SD | Promoter Acc | Structure Acc |
|---|---|---|---:|---:|---:|---:|---:|
| GFP-first | 1-1 | **GFP** | 0.836 | 0.588 | 0.004 | 50.1% | 2.1% |
| GFP-first | 1-2 | promoter | 0.832 | 0.621 | 0.004 | 82.6% | 12.2% |
| GFP-first | 1-3 | structure | 0.840 | 0.575 | 0.004 | 51.9% | 56.4% |
| GFP-first | 2-4 | **GFP** | 0.846 | 0.726 | 0.011 | 53.7% | 47.9% |
| GFP-first | 2-5 | promoter | 0.840 | 0.706 | 0.007 | 91.7% | 53.2% |
| GFP-first | 2-6 | structure | 0.835 | 0.680 | 0.010 | 78.8% | 63.3% |
| GFP-first | 3-7 | **GFP** | 0.773 | 0.504 | 0.254 | 74.9% | 61.1% |
| GFP-first | 3-8 | promoter | 0.816 | 0.542 | 0.108 | 93.4% | 64.3% |
| GFP-first | 3-9 | structure | 0.815 | 0.534 | 0.121 | 94.1% | 70.7% |
| GFP-last | 1-1 | promoter | 1.138 | 1.118 | 0.035 | 85.9% | 17.2% |
| GFP-last | 1-2 | structure | 0.947 | 0.911 | 0.018 | 68.8% | 56.2% |
| GFP-last | 1-3 | **GFP** | 0.770 | 0.532 | 0.226 | 77.1% | 43.8% |
| GFP-last | 2-4 | promoter | 1.098 | 1.073 | 0.135 | 86.0% | 44.3% |
| GFP-last | 2-5 | structure | 0.809 | 0.596 | 0.140 | 82.3% | 62.0% |
| GFP-last | 2-6 | **GFP** | 0.617 | 0.414 | 0.520 | 81.3% | 57.8% |
| GFP-last | 3-7 | promoter | 0.789 | 0.417 | 0.355 | 93.5% | 60.9% |
| GFP-last | 3-8 | structure | 0.819 | 0.438 | 0.310 | 93.5% | 68.6% |
| GFP-last | 3-9 | **GFP** | 0.518 | 0.353 | 0.647 | 93.8% | 68.9% |

判读：

1. **GFP-first 的 GFP 块前两次根本没学起来**：epoch 1 与 epoch 2 的 GFP 块之后 SD 分别为 0.004 与 0.011（常数输出），直到 epoch 3 的 GFP 块才起步（SD 0.254），随后一个 promoter 块又把 SD 压回 0.108。失败主要是"学不动"，其次才是"刚学会就被覆盖"。
2. **GFP-last 的第一个 GFP 块就起步**（epoch 1，SD 0.226），epoch 2 达到 0.520。两种顺序的 GFP 块都紧接 structure 块（GFP-first 的 epoch 1 除外），所以差异集中在 epoch 1：GFP-first 让 GFP 从初始模型、在 LR warmup 与峰值期先训练，GFP-last 的 GFP 块开始前 encoder 已被 256 步分类训练塑形、LR 已过 warmup。冻结规则的严格条件（GFP-last 首块 SD > 0.3）未满足（0.226），但方向明确。
3. **分类更新对 GFP 的影响是均值平移为主**：GFP-last 中 promoter 块使 GFP MAE 升到 1.07–1.12（MAE ≈ RMSE，整体偏移），structure 块又部分拉回；这与 scorer 上 GFP 与分类梯度近乎反向的探针结果一致。GFP 区分度（SD）在已学好时能部分保留。
4. **分类任务之间在高 LR 阶段也强烈互相遗忘**：epoch 1 中 structure 块把 promoter 从 82.6% 打回 51.9%（GFP-first）/ 85.9% → 68.8%（GFP-last），epoch 3 降到几个点以内。这是 block 式调度的普遍代价，不只影响 GFP。

**修正后的机制图景**：GFP 能否学起来取决于它第一次被训练时的起点状态／LR 位置；一旦学起来，后续分类块主要造成均值偏移与部分区分度损失，最后一个 GFP 块再把它拉回。GFP-last 的优势 = 首个 GFP 块起步成功 + 最后位置的恢复。两个未分离的因素：encoder 已被分类塑形 vs. 避开 warmup/峰值 LR。

**下一步建议**：用两个单因素对照分离上述因素（均 seed 20260928 NO-CPT、GFP-first 顺序）：(a) epoch 1 改用 GFP-last 顺序、之后 GFP-first（encoder 预塑形 + 避开 warmup，一起改）；(b) 保持 GFP-first，但把 warmup 期挪到一个短的分类预热（或 GFP 块内用较低 LR）只改 LR 位置。实用 recipe 方向：细粒度 round-robin（如每 8–16 步轮换）以同时缓解 GFP 起步与分类间高 LR 遗忘。

## 轮换粒度实验（2026-09-27）

冻结计划见 `EXPERIMENT_PLAN.md` 末节；输出 `run_round_robin`。每个 epoch 内按 promoter → structure → GFP 循环，每任务每轮 K 个 batch；各任务每 epoch 的 batch 内容与按块训练完全相同（已离线核验），其余训练与评估不变。4 个 formal run 均 1,152 updates、精确重载、曝光 {3: 24576}、无 test inference。

| 调度 | Rep | GFP RMSE | GFP MAE | Spearman | 预测 SD | Promoter | Structure | 成功标准 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| K=8 | 1 | 0.8225 | 0.6462 | 0.281 | 0.045 | 83.8% | 56.4% | fail |
| K=8 | 2 | 0.8242 | 0.6480 | 0.308 | 0.039 | 87.7% | 54.5% | fail |
| K=32 | 1 | 0.7208 | 0.5830 | 0.422 | 0.382 | 78.6% | 58.7% | fail |
| K=32 | 2 | 0.7533 | 0.5039 | 0.413 | 0.258 | 87.0% | 57.0% | pass（MAE 余量 0.006） |
| K=128 按块 GFP-last（n = 4） | — | 0.526–0.595 | 0.315–0.439 | — | 0.54–0.65 | 86.7–87.8% | 54.2–55.4% | 4/4 pass |

判读（按冻结规则，每个 K 需两次都通过）：K=8 与 K=32 均未通过，**细粒度轮换不是可用 recipe**；按块 GFP-last 仍是唯一稳定路径。切换粒度本身是控制因素之一：GFP 通过率随 K 单调上升（K≈1 随机交错 0/1、K=8 0/2、K=32 1/2、K=128 4/4），K=8 两次都近常数（SD < 0.05），与原随机交错相同。

逐段诊断显示 K=8 与 K=32 在 epoch 1 末 GFP 都是常数（SD 0.003–0.005），K=32 到 epoch 3 才起步（SD 0.37），与 GFP-first 的"第三轮才起步"一致。也就是说，频繁切换同样让 GFP 在早期学不动；只有 epoch 1 内连续 128 步、且在分类块之后的 GFP 训练能让它起步。

轮换没有带来预期的分类稳定性收益：promoter 在 K=8/K=32 中有两次低于 85%（83.8%、78.6%），不优于按块训练。

**下一步建议**：recipe 方向暂以按块 GFP-last 为基线；机制上最值得做的是对照 (b)——保持 GFP-first，只改 GFP 首块的 LR 位置（例如 epoch 1 前加一段短的分类预热，或 GFP 块内降低 LR），以区分"encoder 已被分类塑形"与"避开 warmup/峰值 LR"。若 LR 位置是主因，则可尝试"GFP 专用较低 LR 或延迟 warmup"的实用 recipe，使 GFP 不依赖处在最后位置。
