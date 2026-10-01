# Benchmark v2 评测 harness 与首轮零样本结果

日期：2026-10-01。harness：`scripts/benchmark_v2/eval_benchmark_v2.py`；结果：`artifacts/benchmark_v2_eval/`。本文记录 harness 设计与**首轮零样本评测**，非训练后成绩。

## 1. harness 设计

复用 MVP1 frozen 码（`artifacts/laya_jev_multitask_v1/round/frozen_code/`）的 `Representation`/`build`/`render`/`pack`，让 checkpoint 以同一 typed-decision 路径消费 benchmark v2 统一记录。被评测 checkpoint = 推荐配方 `run_gfp_loss2x_xseed/.../fluo2p0`（cls_warmup64 + GFP loss×2，NO-CPT，seed 20260926），base=`artifacts/laya_model`，cpt_root=`artifacts/laya_biocpt_v2`。

每条统一记录映射到 MVP1 row schema：
- `noul`：choices=`["false: <no候选>", "true: <yes候选>"]`，label=1 iff answer==yes。
- `choice`：choices=candidates，label=candidates.index(answer)。
- `score`：benchmark 存 4 个内部分位切点（5 桶），harness 派生 5 个代表锚值（桶中点+两端外推），choices=`["value=<anchor>"×5]`，label=gold_level，解码 value=probs·anchors。

**架构约束**（MVP1 SharedDecision 的固有限制，非 harness bug）：单序列、DNA/蛋白、512-token 预算。因此 harness 支持 noul/choice/score 的单序列 DNA/蛋白任务，对其余按原因跳过并报告：`multi_noul`(DeepLoc)、`double_sequence`(gene_lan/protein_homology)、`modality_rna`(RNAcompete)、`over_budget`(>512 token)。

## 2. 覆盖（290 任务，max 100 样本/任务）

| 类别 | 数 |
|---|---|
| 评测到（supported） | **241**：score 197 · choice 37 · noul 7 |
| 跳过 | **49**：over_budget 24 · modality_rna 14 · double_sequence 10 · multi_noul 1 |

over_budget 24 主要是长 DNA（GB drosophila/mouse enhancer ~2kb、部分 GUE）与长蛋白；这是 MVP1 512 预算的固有排除，与主实验"全输入不截断、超预算排除并报率"的口径一致。

## 3. 首轮零样本结果（按族）

checkpoint 只在 3 个任务上训练过（启动子/7类结构/GFP荧光）；benchmark v2 的其余任务是**未训练**的，故这是共享决策接口对新问题/新候选的**零样本泛化探针**。

| 族 | 接口 | 指标 | 值 | 备注 |
|---|---|---|---|---|
| TAPE | score | spearman | **0.253** (n=2) | fluorescence 是训练过的任务→有信号 |
| local_snapshots | choice | acc | 0.487 (n=5) | 含训练过的 fold_class（smoke 58%）与未训练的 npp/signal/subcell |
| GUE | choice | acc | 0.577 (n=28) | 多为二分类，0.5=随机，略高于随机 |
| GenomicBenchmarks | choice/noul | acc | 0.676 (n=8) | 类别不均衡，acc 偏高 |
| dnagpt_pools | choice | acc | 0.400 (n=3) | splice 3类≈随机、tf/core 二分类 |
| ProteinGym | score | spearman | **-0.001** (n=193) | 未见蛋白，≈无信号 |
| DeepSTARR | score | spearman | -0.094 (n=2) | 未见 |

聚合（scale-free）：score mean spearman **0.000**（median 0.003，frac>0.2 仅 14%）；choice mean acc **0.539**；noul mean acc 0.751 但 **mean AUROC 0.387**（<0.5）。

**判读**：
1. harness 机械正确——训练过的族（TAPE fluorescence spearman 0.25、fold_class 58%）出信号，未见族≈随机，说明 checkpoint 读取与解码路径无误。
2. 共享 typed-decision 接口**能机械处理**任意新问题/新候选（无任务头），但**生物学知识不零样本迁移**到未见任务——这是预期的，也界定 benchmark v2 的用法：要出成绩须在其 train 上训练（下一步）。
3. **noul AUROC<0.5 需排查**：可能是零样本失败，也可能是 noul 候选文本极性（"false:/true:" 约定）或正类概率取位的符号问题；样本仅 7（多数 noul 是双序列被跳过），暂不下结论。

## 4. 指标口径注意

- **score 的 RMSE 不可跨 assay 平均**：ProteinGym 各 assay 的 DMS_score 原生尺度差异极大（有的上千），原始 RMSE 均值被单个大尺度 assay 主导（首轮 `eval_summary.json` 的 mean_rmse≈8.5e4 即此假象）。跨 assay 比较只用 **spearman**（尺度无关）或按 gold_std 归一化的 RMSE（median 1.08）。`eval_summary_clean.json` 为修正聚合。
- choice/noul 的 acc 受类别不均衡影响，须并看 macro-F1/AUROC。

## 5. 下一步

1. 在 benchmark v2 train 上训练（联合或分族），再评 test——这才是基准的正式用法与成绩来源。
2. 排查 noul AUROC 符号/极性。
3. 架构扩展以覆盖被跳过的 49 任务：双序列编码（gene_lan/protein_homology）、multi-label sigmoid 头（DeepLoc）、RNA tokenizer（RNAcompete）、长序列预算。
4. score 增加标量回归对照；choice 增加候选重排鲁棒性（reverse）评测。
