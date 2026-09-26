# Laya JEV 三任务联合模型：配对 seed 确认轮

冻结于新训练开始前，2026-09-25 UTC。本轮回应 [当前计划](../../plan.md) 中的确认 seed 条件。上一轮固定结果和证据保持原样；本轮只估计训练随机性下的稳定程度，不选择最优 seed 或修改配方。

## 固定问题和矩阵

在上一轮 8,192/任务的相同训练实体、相同完整 development 集和相同 Score 锚点上，分别以 seed 20260927、20260928 训练 NO-CPT-JOINT 与 CPT-JOINT，共四个新增模型。每对 seed 的两臂共享随机初始化的 typed head、任务批次顺序和 Choice 候选排列；两臂的 encoder 初始化只差原有生物 CPT。原 seed 20260926 与两对新增 seed 共同形成三 seed 描述。原单任务 CPT 参照不重跑，本轮不做新的联合退化因果比较。

每臂三 epoch、1,152 次更新、有效 batch 64、micro-batch 8；encoder LR 2e-5、共享 typed head/scorer LR 1e-4，原 5% warmup/cosine、AdamW、loss、BF16、模型结构、数据/词表及不截断约束全部沿用冻结代码。实际 seed 由只改变该代码全局 `SEED` 的薄包装器注入；底层训练实现不编辑。四臂依次执行 NO-CPT/CPT（20260927），NO-CPT/CPT（20260928），避免同 GPU 并发。先各运行 train-only 三更新 smoke，确认梯度、有限值和保存重载，再启动正式训练。

## 预先固定的读数

- 每个 seed 分别报告 promoter Accuracy/Macro-F1、structure Accuracy/Macro-F1 和七类召回、GFP 原生 RMSE/MAE/Spearman、两种候选顺序的语义 argmax 一致率及有效输出率；同时保留全部 epoch 指标与逐样本预测。
- 继续用上一轮冻结的训练多数类/均值/中位数常数基线和同一实用性门槛。报告三 seed 各自是否通过，另给均值和样本标准差；三 seed 不足以支撑稳定的统计显著性或总体方差估计。
- 按 [探索性诊断](../../research/jev_followup_diagnostics_20260925.md) 的锚点区间报告 GFP MAE 和偏差、类别 4/5 的正确数，并明确这些是后验切片，不用于新 checkpoint 选择。若任一结果改变当前推荐，只陈述配对 seed 证据，不再围绕复用 dev 调参。
- 不读取 test 标签，不做 test 推理；不拟合 dev 校准参数，也不评估 few-shot、未知任务或替代 backbone。

## 准入、留存和成本

输入数据、原始模型和 CPT 模型的 SHA-256 必须匹配上一轮冻结 manifest；本轮脚本/源代码哈希单独记录。需要一张空闲本地 RTX 4080 SUPER 和至少 20 GiB 可用磁盘。四个联合 checkpoint 与 CPT optimizer/RNG 状态保留，约需 13 GiB；不得覆盖旧输出。每个运行保存训练轨迹、末轮和各 epoch development 预测、固定训练诊断、顺序扰动、精确重载结果及日志。先前单臂约 34 分钟，四臂加评估与 smoke 约 2.5–3 小时，实际时间以日志为准。

如果 smoke、哈希、内存、梯度、有限值或重载失败，停止后续运行，保留故障证据；不将工程失败解释为模型失败。完成后独立复算并汇总，才决定是否需要完整训练池与真实 Qwen 成本对照。
