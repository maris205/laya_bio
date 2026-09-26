# JEV 三任务联合模型：配对 seed 确认结果

完成于 2026-09-25 14:36 UTC。**结论：分类任务在三组 seed 上都通过预设实用性门槛；GFP Score 没有稳定通过。** 原 NO-CPT-JOINT checkpoint（seed 20260926）仍是六个联合 checkpoint 中唯一同时通过三任务门槛的已观察模型，但不能再把它的三任务表现表述为稳定的训练结果。NO-CPT 与 CPT 在 GFP 上的优劣方向随 seed 反转，当前数据不支持普遍的 CPT 有益或有害结论。

## 冻结设置与核验

在[新轮冻结计划](../artifacts/laya_jev_multitask_seed_confirm/EXPERIMENT_PLAN.md)下，新增 seed 20260927/20260928 的 NO-CPT/CPT 联合训练各一臂，共四个模型、4,608 次正式更新。与 seed 20260926 的两臂合并描述。每臂沿用相同 8,192/任务训练实体、完整 development 集、三 epoch、共享 JEV scorer、Score 锚点、优化器和学习率日程；两臂按 seed 配对批次顺序与 Choice 候选排列。没有读取新的 test 标签、做 test 推理或用 dev 挑选 checkpoint。

四个 train-only smoke 与四个正式训练都完成。独立汇总复算了六个联合 checkpoint 的末轮预测，核对六份模型文件 SHA-256、各 1,152 次更新、每任务 24,576 次曝光、全部训练轨迹的有限且非零 encoder/scorer 梯度、有效输出、末轮指标与保存的 learning curve、精确重载状态、同 seed 共享头/批次/面板哈希，以及所有 seed 的样本 ID/标签/Score 真值一致性；审计为 **PASS**。四个新模型权重均保留。机器可读结果见 [summary.json](../artifacts/laya_jev_multitask_seed_confirm/run/analysis/summary.json)，运行状态与日志见[新轮目录](../artifacts/laya_jev_multitask_seed_confirm/run/status.json)，复算程序见 [summarize_jev_seed_confirmation.py](../scripts/summarize_jev_seed_confirmation.py)。

**冻结计划措辞勘误：** 计划写作“共享随机初始化的 typed head”不准确。六臂的初始共享头哈希全部相同（`15d87253bbef7cf7f1dc4ecff4009e3fb0eca211c3f1f2e58104f596304d5054`）；每种 encoder 初始化下，三个 seed 的 epoch-0 dev 预测文件逐字节相同。seed 实际改变训练批次顺序、Choice 排列和训练随机流，而没有提供新的头初始化。因此本轮估计的是**固定初始化下的训练随机性**，不涵盖头初始化方差。

## 六臂原始末轮结果

| Seed | 联合臂 | Promoter Acc | Structure Acc / Macro-F1 | GFP RMSE ↓ | GFP MAE ↓ | GFP Spearman ↑ | 三任务门槛 |
|---:|---|---:|---:|---:|---:|---:|---|
| 20260926 | NO-CPT | 88.12% | 55.48% / 0.4959 | **0.6993** | **0.4921** | 0.4589 | 全部通过 |
| 20260926 | CPT | 88.88% | 56.87% / 0.4390 | 0.7512 | 0.6229 | 0.3682 | GFP 未过 |
| 20260927 | NO-CPT | 89.26% | 56.98% / 0.4612 | **0.6910** | 0.5279 | 0.4313 | GFP 未过 |
| 20260927 | CPT | 88.50% | 57.08% / 0.4451 | 0.8160 | 0.6249 | 0.2757 | GFP 未过 |
| 20260928 | NO-CPT | 88.59% | 56.55% / 0.4303 | 0.8345 | 0.6241 | 0.1929 | GFP 未过 |
| 20260928 | CPT | 88.21% | 55.80% / 0.4333 | 0.7583 | 0.5200 | 0.4109 | GFP 未过 |

冻结常数基线：Promoter 训练多数类 Acc 49.52%、Macro-F1 0.3312；Structure 训练多数类 Acc 30.14%、Macro-F1 0.0662；GFP 训练均值 RMSE 0.8357、训练中位数 MAE 0.5095。所有六臂的分类 Accuracy/Macro-F1 均超过相应常数；GFP RMSE 均低于训练均值常数，但 20260928 NO-CPT 只低 0.00125。只有 20260926 NO-CPT 的 GFP MAE 低于训练中位数常数。

| 三 seed 描述统计 | NO-CPT 均值 ± 样本 SD | CPT 均值 ± 样本 SD |
|---|---:|---:|
| Promoter Accuracy | 88.66 ± 0.57% | 88.53 ± 0.33% |
| Structure Accuracy | 56.34 ± 0.77% | 56.59 ± 0.68% |
| Structure Macro-F1 | 0.4625 ± 0.0328 | 0.4391 ± 0.0059 |
| GFP RMSE ↓ | 0.7416 ± 0.0806 | 0.7752 ± 0.0355 |
| GFP MAE ↓ | 0.5480 ± 0.0683 | 0.5893 ± 0.0600 |
| GFP Spearman ↑ | 0.3610 ± 0.1462 | 0.3516 ± 0.0691 |

三个 seed 的样本标准差只描述本次训练随机性；development 集重复使用，不包含数据集抽样、亲本蛋白、预训练或 checkpoint 选择等不确定性。

## 关键诊断

| Seed | GFP RMSE：CPT − NO-CPT | GFP MAE：CPT − NO-CPT | NO-CPT 预测 SD | CPT 预测 SD |
|---:|---:|---:|---:|---:|
| 20260926 | +0.0519 | +0.1308 | 0.3689 | 0.3148 |
| 20260927 | +0.1250 | +0.0970 | 0.4212 | 0.1137 |
| 20260928 | **−0.0762** | **−0.1041** | **0.0127** | 0.2408 |

20260928 NO-CPT 的 GFP 连续预测几乎为常数：末轮 dev 预测标准差 0.0127，固定 1,024 条训练诊断上的标准差 0.0121；其 dev/train RMSE 为 0.8345/0.8322。故这是训练诊断中也出现的近常数输出，不能解释为单纯的 dev 过拟合。20260927 CPT 的 GFP 预测标准差也较小（0.1137）。这些是输出行为；具体原因可能涉及 Score 损失、任务交替或优化动力学，尚未被当前矩阵隔离验证。

结构任务的两个 16 条稀有类别在 NO-CPT 三 seed 中分别正确 4/0/0 条（类别 4）和 1/1/0 条（类别 5）；CPT 对两类均为 0。原轮 NO-CPT 的 macro-F1 优势因此也不稳定：20260928 CPT 的 macro-F1 反而高 0.0029。canonical/反序候选的语义 argmax 一致率在六臂为 88.29–91.37%，仍不能声称候选顺序不敏感。

## 决策与下一实验

1. **已观察 checkpoint：** 若现在必须部署同一个三任务模型，seed 20260926 NO-CPT 是唯一通过全部预设门槛的已保存选择；应附带 GFP seed 敏感和 Choice 顺序敏感的限制。不能从这六臂推断该配方重复训练会稳定达到同等性能。
2. **CPT 判断：** 三个配对 seed 的 GFP RMSE/MAE 差值方向不一致。尽管三 seed 均值略偏向 NO-CPT，不能把 CPT 对 GFP 的不利影响写成可重复的结论；同理也不能因 20260928 的反转而说 CPT 稳定有益。
3. **下一项可检验实验：** 在独立冻结的新轮中先分离 GFP 优化问题：给 NO-CPT 与 CPT 各做相同监督预算的 GFP 单任务参照，并与联合臂固定训练诊断比较；或者直接上完整训练池做配对 seed 剂量实验。两条路线不能混为一次机制证明。若研究目标是 backbone 选择，应另行实施相同任务、数据、JEV 接口、曝光和推理计时的 Qwen 对照。任何新配方都不应按这份复用的 dev 切片反复调参。

这些结论仍局限于三个训练随机 seed、固定初始化、8,192/任务标签、复用 dev，以及 GFP 变体共享亲本的原生 split。它们不构成独立 test、family-independent 泛化、统计显著性或架构优越性证据。
