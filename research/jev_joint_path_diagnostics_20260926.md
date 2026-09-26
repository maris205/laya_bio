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
