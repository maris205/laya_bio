# Paper Plan 2 — Capacity & Causal Backbones

**Title（工作）**: Does Capacity Rescue the Shared Typed-Decision Scorer? A Scale- and
Backbone-Controlled Replication of BioDecisionBench's Architecture Comparison
**备选**: Scaling Typed-Decision Interfaces: Encoder vs Causal Backbones from 0.6B to 8B
on a Leak-Proof Biological Benchmark

**One-sentence contribution**: On the same leak-proof benchmark and the same seed-averaged
protocol, we replicate the shared-scorer-vs-task-heads comparison across backbone capacity
(423M encoder, Qwen3-0.6B, Qwen3-8B LoRA) and backbone type (masked encoder vs causal LM),
testing whether the heads-over-shared-scorer ordering of Paper 1 is a capacity artifact or
an interface property.

**Venue**: 先不绑定（NeurIPS D&B 续投或 ICLR empirical 均可）；Kaggle Gemma-4 paper track
为备选出口（若拿到 Gemma-4 权重则加 31B-QAT 推理点）。

## 与 Paper 1 的关系
- Paper 1 = 基准 + 协议 + 423M 编码器上的对照（4 regime 复现 heads ≥ shared）。
- Paper 2 = 把对照搬到 **causal LM backbone + 规模轴**：若 heads ≥ shared 在 8B 仍成立
  → 接口属性（强结论）；若 8B 上 shared 追平/反超 → 容量假说成立（也强）。两种结果都可发表。
- 全部复用：BioDecisionBench 划分/准入、seed 平均协议、按族早停、granularity ladder、
  方差方法论；#3 frozen Qwen3-0.6B 候选似然 = Paper 1 已有数字。

## Claims-Evidence（草案）
| # | Claim | Evidence | Status |
|---|---|---|---|
| S1 | 因果 backbone 上 typed-decision 三臂可实现且 matched | 新训练器（LoRA bf16/QLoRA），37 切片 3-seed | 待跑 |
| S2 | heads-vs-shared 排序对容量稳健（或反转） | 0.6B/8B × {shared, heads} 3-seed | 待跑 |
| S3 | 规模轴：frozen→LoRA→full? 的 noul/score 增益曲线 | 0.6B frozen（Paper1 #3）/0.6B LoRA/8B LoRA | 待跑 |
| S4 | granularity ladder 在 8B 复现（单任务≈族>全联合） | 5 代表任务 × 3 粒度 | 待跑 |
| S5 | 对比解码/措辞等机制修复跨 backbone 迁移 | 措辞消融 × 8B shared 臂 | 可选 |

## 实验矩阵（GPU 32GB ×1，bf16 LoRA；QLoRA 兜底）
1. **smoke**：promoter 单任务 #2Q/#1Q 各 300 updates，验 VRAM/loss/评测链路（~30min）
2. **37 切片主对照**：{#1Q shared-scorer-on-markers, #2Q per-task heads} × 3 seeds ×
   {Qwen3-8B LoRA}（~4-6h/seed/臂 → 分批 2 天）；Qwen3-0.6B LoRA 同矩阵 1 seed 作规模低点
3. **granularity ladder**：5 代表任务 × {single, family, all-241} × #2Q（8B）（~6-8h）
4. **机制**：措辞消融 × 8B shared（3 ckpt）（~2h）；对比解码在 8B 候选似然臂（~1h）
5. （可选）**Gemma-4-31B-QAT 推理点**：仅 frozen 候选似然 + 对比解码（需 Kaggle 条款）

## 训练器设计（scripts/benchmark_v2/qwen_decision.py）
- 渲染复用 Paper1 文本模板：question + sequence + 候选（#1Q 带 `[MARK]` 尾标记取 hidden；
  #2Q 无候选，pooled hidden → per-task head；score #2Q = 标量回归头）
- LoRA：peft r=16 α=32 targets q/k/v/o+gate/up/down；bf16 + grad-ckpt + micro-batch 8×accum 8；
  512-token 全输入；VRAM 超预算自动降 QLoRA(4-bit)
- 协议同 Paper1：family-balanced、warmup .15、clip .5、dev 早停（dev≥256）、3 seeds、
  per-task seed-mean → 宏平均、报 SEM+run-std

## 图表计划
- Fig1：规模×架构 noul/choice/score 折线（0.6B frozen/0.6B LoRA/423M/8B LoRA）
- Fig2：37 切片 8B 主对照条形（±SEM）+ Paper1 423M 并排
- Fig3：granularity ladder 8B 版（复用 fig5 生成器换数据源）
- Table1：主对照；Table2：ladder；Table3：VRAM/吞吐/更新数成本表（规模论文必报）

## Next Steps
- [x] Qwen3-8B 下载中（重试循环）
- [ ] qwen_decision.py + smoke
- [ ] 主对照 driver（seed 分批挂）
- [ ] ladder driver
- [ ] /paper-figure → /paper-write（双轨：中文稿 research/benchmark_v2_scale_draft.md 先行）
