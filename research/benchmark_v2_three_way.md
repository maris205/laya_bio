# Benchmark v2 三方主对照：#1 共享打分器 vs #2 任务头 vs #3 生成式候选似然

日期：2026-10-03。#1 `train_benchmark_v2.py`（SharedDecision，trained）、#2 `train_taskheads.py`（编码器+每任务头，trained）、#3 `eval_generative.py`（frozen Qwen3-0.6B 全候选似然）。结果 `artifacts/benchmark_v2_eval/{f2_init,f2_trained,gen3}/`、`bv2_taskheads/`、`three_way_summary.json`。

## 对照设置与可比性

| | #1 共享 typed-decision | #2 编码器+任务头 | #3 生成式候选似然 |
|---|---|---|---|
| backbone | Laya ModernBERT-large（423M）| **同 #1** | Qwen3-0.6B（596M，因果 LM）|
| 训练 | 43 任务/2520 upd/family-balance | **同 #1** | **frozen，未训练** |
| 决策 | 共享 scorer 打分动态候选 marker | 每任务固定类 softmax / score 标量回归头 | 每候选 length-normalized logprob，argmax |
| 序列处理 | BPE bio token，≤512 全输入 | 同 #1 | 原始字符 tokenizer，截断 120 字符（#3 局限）|

**可比性警示**：#1/#2 是 matched compute（同 backbone/数据/更新/seed/评测子集，唯一变量=决策机制）；**#3 是 frozen 不同 backbone**，只回答"现成小 LM 靠候选似然能否匹敌训练过的 typed-decision 模型"，**不是** matched compute。score 任务 #3 输出的是 5 锚点等级 accuracy，与 #1/#2 的 spearman **口径不同，不可直接比**。

## 结果

### 分类（noul/choice，27 任务，accuracy）

| family | n | init | #1 共享 | #2 头 | #3 生成 |
|---|---|---|---|---|---|
| GUE | 12 | 0.478 | 0.457 | 0.485 | 0.436 |
| GenomicBenchmarks | 7 | 0.487 | 0.483 | 0.536 | 0.493 |
| dnagpt_pools | 3 | 0.433 | 0.527 | 0.400 | 0.400 |
| local_snapshots | 5 | 0.479 | 0.545 | 0.550 | 0.432 |
| **ALL** | **27** | **0.475** | **0.488** | **0.501** | **0.446** |

逐任务最优（含并列）：#1 5 / #2 11 / #3 8 / 并列 3。

### score（16 任务）

| family | n | #1 spearman | #2 spearman | #3 level-acc(异口径) |
|---|---|---|---|---|
| TAPE | 2 | **0.237** | −0.065 | 0.300 |
| ProteinGym | 12 | 0.046 | −0.048 | 0.162 |
| DeepSTARR | 2 | 0.048 | 0.136 | 0.220 |

## 判读（诚实，不过度声称）

1. **当前预算下三方都接近随机**（分类 0.44–0.50）。2520 updates、每任务 ≤768 样本、512 窗口、单 checkpoint 跨 43 任务——预算是主要瓶颈，机制差异被淹没。
2. **#3 frozen 生成式与训练过的 #1/#2 在分类上接近**（0.446 vs 0.488/0.501，逐任务还赢 8 个）：说明在这么小的训练预算下，专门训练**尚未显著拉开**与现成小 LM 候选似然的差距。这既是诚实的负面信号，也指出"要证明 typed-decision 训练的价值，需要更大预算/更多数据"。
3. **#1 的清晰优势在 score**：#1 锚点期望值 spearman（TAPE 0.237）远好于 #2 标量回归头（−0.065，甚至低于 init）。即对连续目标，"有序等级 + 期望值"的 typed-decision 机制比朴素标量回归头更稳——这是 #1 机制的实证优点。
4. **#2 在分类 accuracy 上略胜 #1**（0.501 vs 0.488，逐任务 11 vs 5）：专用固定类头在分类上稍强；但 #1 用**无任务参数 + 动态候选**达到接近水平，且 #1 相对 init 的提升（+0.053）大于 #2（+0.035，见 head2head 文档）——两指标口径不同，需并列报告，不能只挑对自己有利的。
5. **noul 仍是 #1 弱项**（GB AUROC #1 0.519 < #2 0.636），候选文本极性/校准待修。

## 局限与下一步
- #3 frozen、异 backbone、序列截断 120 字符、score 口径不同——只作"现成 LM 基线"，不作 matched 对照。若要 matched #3，需在 Qwen 上做同数据微调的生成式/受约束解码，工作量大。
- 三方绝对值低，主因预算。下一步：(a) 加大预算（更多 epoch/数据）看 #1/#2 能否显著超过 frozen #3；(b) 修 #1 noul 校准；(c) score 给 #2 也配 5 级分类头做更公平比较；(d) 扩到全 265 supported 任务。
