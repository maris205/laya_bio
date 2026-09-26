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

## 下一步

已冻结并启动 replicate 轮（`run_replicate`）：`task_block_gfp_last` 与 `task_block_gfp_first` 在 seed 20260928 各重复 2 次，与原 run 合计每变体 n = 3。若所有 GFP-last run 在 GFP RMSE 与 MAE 上都优于所有 GFP-first run，且均值差超过组内极差，则把 GFP-last recency 视为可复现效应，再进入 optimizer-state 隔离或梯度冲突 probe；若区间重叠，则记录为在 n = 3 下与 run-to-run 噪声不可分，不在其上构建机制主张。loss weighting 方向暂停。
