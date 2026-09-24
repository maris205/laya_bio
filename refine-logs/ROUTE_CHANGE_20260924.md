# 路线调整与暂停记录：2026-09-24

用户决定暂停当前直接监督适配／GFP 拟合诊断主线，优先开展传统领域适配路线：生物词表扩展 → 新增 embedding/MLM 头适应 → 生物序列 CPT → 足量数据单任务 SFT → 累积多任务 SFT。始终保留同样扩词表、但不做 CPT 的配对对照。

## 已完成证据保留

- 最近一次规模／学习率诊断见 [报告](../research/laya_gfp_scale_probe.md)，提交 `f03f83b`。32 样本低学习率训练 RMSE 0.02420、Accuracy 100%；1,024 样本低学习率延长至 1,024 步后训练 RMSE 0.70786、Accuracy 32.13%，仍欠拟合。
- 这些是未做新增生物 CPT 的监督适配／训练诊断，不是完整传统路线的验证。最近的 32/1,024 样本结果也不能替代足量数据 SFT。
- 原先建议的 GFP 2,048 步精确重放延长 **暂停，不执行**。暂停时没有仍在运行的相关 GPU 进程；代码、权重和记录全部保留。
- 原接口比较计划归档于 [旧计划](EXPERIMENT_PLAN_interface_20260924_paused.md)。历史论文与结果不回写成 CPT 实验。

## 新路线的约束

1. 主对照只改变是否进行生物 CPT：共享扩展词表、初始化规则、分类器、SFT 样本、批次顺序、更新预算和种子。
2. 新 embedding 需验证旧 ID 不变、原始片段初始化、lossless 编码、可训练状态、输入/目标曝光、梯度与权重变化。只清零旧 embedding 梯度不足以保证 AdamW 不改旧行；适应阶段必须明确隔离旧行并校验其不变。
3. CPT 使用无标签 DNA/蛋白序列，单独留出固定 MLM 验证集；下游留出序列仅用于污染过滤，不使用测试标签或测试性能调参。
4. 先用足量启动子二分类训练，再加入蛋白结构分类；保留旧任务数据回放，每阶段记录新旧任务的独立开发集指标。
5. 首轮单 seed、有限 CPT 预算仅是领域适配初轮，不保证 CPT 提升。随后做数据量曲线、预算曲线与多 seed。few-shot 是待检验结果，CPT 不自动赋予随机分类头 zero-shot 能力。

## Preflight implementation correction

The initial data preparation was stopped before model training: a per-token `len(tokenizer)` call cost approximately 0.85 seconds per 100 calls under the installed tokenizer implementation. The vocabulary size is now cached once. The partial attempt and frozen original script remain in `artifacts/laya_biocpt_v1/data_slow_attempt` and `frozen_preparation`; the same deterministic sampling/encoding recipe was restarted without a scientific protocol change. Synthetic tests pass for initialization, old-row invariance under AdamW, encoder unfreezing, masked loss normalization and exact checkpoint reload.
