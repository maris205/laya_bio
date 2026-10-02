# Benchmark v2 主对照 #1 vs #2：共享 typed-decision scorer vs 编码器+任务头

日期：2026-10-02。代码：#1 = `train_benchmark_v2.py`（SharedDecision 共享打分器，frozen MVP1 路径）；#2 = `train_taskheads.py`（同 backbone + 每任务输出头）。结果：`artifacts/benchmark_v2_eval/{f2_init,f2_trained}/`、`artifacts/benchmark_v2_train/bv2_taskheads/`、`head2head_summary.json`。

## 对照设计（matched compute）

| 维度 | #1 共享 typed-decision scorer | #2 编码器+任务头 |
|---|---|---|
| backbone | 同一 Laya ModernBERT-large encoder（`bio.build(laya_model, no_cpt)`）| **同** |
| 数据 | benchmark v2 同 43 任务、max 768/任务 | **同** |
| schedule | family-balance，12 epoch × 30 bpf = **2520 updates** | **同** |
| 优化器/LR | AdamW，encoder 2e-5 / head 1e-4，cosine | **同** |
| seed | 20261001 | **同** |
| 评测子集 | 每任务 test 随机 200（seed 20261001）| **同** |
| **决策机制（唯一变量）** | 一个**共享** scorer 对**动态候选 marker** 打分；score 用锚点期望值；无任务专用参数 | 每任务一个**输出头**：noul/choice = 固定类数 softmax，score = **标量回归头**(MSE)；不消费候选文本作 marker |

#2 的局限（设计使然，正是要对照的点）：需**预先声明固定标签集**，不能动态候选；score 用标量回归而非锚点等级。

## 结果（43 任务，配对）

| family | metric | n | init | #1 共享 | #2 任务头 | #1−#2 |
|---|---|---|---|---|---|---|
| TAPE | spearman | 2 | −0.090 | **0.237** | −0.065 | **+0.302** |
| ProteinGym | spearman | 12 | −0.024 | 0.046 | −0.048 | +0.093 |
| dnagpt_pools | acc | 3 | 0.433 | 0.527 | 0.400 | +0.127 |
| local_snapshots | acc | 5 | 0.479 | 0.545 | 0.550 | −0.005 |
| GUE | acc | 12 | 0.478 | 0.457 | 0.485 | −0.028 |
| GenomicBenchmarks | AUROC | 7 | 0.484 | 0.519 | **0.636** | −0.117 |
| DeepSTARR | spearman | 2 | −0.032 | 0.048 | 0.136 | −0.087 |

**逐任务头对头**：#1 胜 11 / 平 19 / #2 胜 13。**相对 init 的 mean Δ**：#1 **+0.053**、#2 +0.035。

## 判读

1. **核心主张成立**：共享 typed-decision scorer **无任何任务专用参数、用动态候选文本**，在 matched compute 下与每任务头**逐任务打平（11/19/13）且总体略优（+0.053 vs +0.035）**。即"一个共享打分器 + 动态候选"能达到甚至略胜"每任务固定头"，这是 Jev 类统一小模型接口的价值点。
2. **互补强弱**：
   - **score 任务 #1 占优**（TAPE +0.30、ProteinGym +0.09）：#1 的锚点等级期望值机制比 #2 的标量回归头更稳；#2 的 score 头在本预算下甚至低于 init（标量 MSE 在原生尺度上欠拟合）。DeepSTARR 是例外（#2 更好）。
   - **noul 任务 #2 占优**（GB AUROC 0.636 vs 0.519）：专用二分类头在 AUROC 上更强；#1 的 noul 受候选文本极性/校准影响（与此前 noul AUROC 排查一致），是 #1 可改进点。
   - **choice 任务 ≈ 平**（GUE/local 微差，dnagpt #1 好）。
3. **绝对值都低**：两者都远未达专用 SOTA，受限于小预算（2520 upd）、每任务 768 上限、512 窗口、单 checkpoint 跨 43 任务摊薄。对照的意义在**相对**（同预算下机制差异），不在绝对水平。

## 局限与下一步

- #2 的 score 用标量回归头；若改成"每任务 5 级分类头"可能更公平，但那就接近 #1 的锚点机制——#2 的价值正是展示"固定头/标量回归"这一常规做法。
- 仅 43 任务（GUE/PG 被 cap）；#1 vs #2 在全 265 supported 上的对照待补。
- **#3 生成式全候选似然**（因果 LM backbone，自由生成/受约束生成/完整候选似然）尚未做，是主对照的第三条腿；需下载小因果 LM（如 Qwen3-0.6B）并写候选似然评测。
- noul：排查 #1 的候选文本极性/校准，可能缩小与 #2 的 AUROC 差距。
