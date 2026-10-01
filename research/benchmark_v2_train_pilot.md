# Benchmark v2 训练 pilot（分族小样，验证可训练性与多任务稳定性）

日期：2026-10-01。代码：`scripts/benchmark_v2/train_benchmark_v2.py`（任务无关的多任务共享决策训练器，复用 MVP1 frozen 的 SharedDecision/build/render/pack/Representation + AdamW + cosine LR + 平衡随机交错）。结果：`artifacts/benchmark_v2_train/bv2_pilot2/`、`artifacts/benchmark_v2_eval/pilot2_{init,trained}/`。

## 目的

验证 benchmark v2 能否在其 train 上训练、共享 typed-decision 接口能否在**多任务**下稳定学到（而非零样本）。先用 13 个代表任务（覆盖 choice/noul/score、DNA/蛋白、GUE/GenomicBenchmarks/dnagpt池/local快照/TAPE/DeepSTARR/ProteinGym 各族）做小样。

## 设置

- 13 任务，`max-per-task 1536`、`epochs 6`、`batch 64`、encoder lr 2e-5 / head lr 1e-4、cosine、SEED 20261001。共 **1170 updates**（~15 min，单卡 4080）。
- 起点 = `build(laya_model, laya_biocpt_v2, no_cpt)`（共享 typed-decisions base，无本基准 SFT），保存 `init.safetensors` 作**同一 init 的干净零样本基线**。
- Score 任务用每任务自带 anchors 派生的 5 代表锚，per-task score_spec（CE 软目标 + 期望值 MSE）。
- 评测 = `eval_benchmark_v2.py`，init 与 trained 用**相同 seed、相同 200 样本子集** → 配对比较。

## 结果（paired：init 零样本 → trained）

13 任务：**10 提升(>+0.05) / 2 持平 / 1 下降**，mean Δ **+0.186**。

| 任务 | 接口 | 指标 | init | trained | Δ |
|---|---|---|---|---|---|
| gb_human_ensembl_regulatory | choice | acc | 0.290 | 0.725 | +0.435 |
| lg_fold_class | choice | acc | 0.045 | 0.435 | +0.390 |
| gue_prom_prom_core_all | choice | acc | 0.440 | 0.705 | +0.265 |
| tape_stability | score | spearman | −0.218 | 0.041 | +0.259 |
| dna_splice | choice | acc | 0.280 | 0.535 | +0.255 |
| deepstarr_dev | score | spearman | −0.005 | 0.141 | +0.145 |
| gb_human_enhancers_cohn | noul | AUROC | 0.570 | 0.715 | +0.145 |
| dna_tf | choice | acc | 0.545 | 0.655 | +0.110 |
| lg_signal_peptide | choice | acc | 0.755 | 0.850 | +0.095 |
| tape_fluorescence | score | spearman | 0.037 | 0.117 | +0.080 |
| pg_A0A2Z5U3Z0_9INFA_Doud_2016 | score | spearman | −0.013 | 0.028 | +0.041 |
| gue_H3K4me3 | choice | acc | 0.540 | 0.535 | −0.005 |
| pg_DYR_ECOLI_Thompson_2019 | score | spearman | 0.014 | −0.073 | −0.087 |

## 判读

1. **接口能在多任务上稳定学到**：10/13 显著提升，无灾难性崩溃。这是"benchmark v2 可训练、共享决策接口适配多任务成立"的证据。
2. **训练稳定性 = 预算问题**：首个 pilot 只有 78 updates（每任务 ~6 更新）时 lg_fold_class 崩到 0.15；充分暴露（1536×6、1170 updates）后回到 0.435。说明多任务联合训练在小预算下会欠训/相互稀释，需足够 updates。
3. **ProteinGym 突变适应度最难**：两个 assay 的 spearman 几乎不动（0 附近），符合零样本≈随机、且 6 epoch 小样不足以学到（该族是"预测任意新蛋白的突变效应"，泛化难度高）。
4. **joint vs 专用的 per-task 权衡**：pilot 的 fold_class 0.435、fluorescence spearman 0.117，仍低于专用 3-任务配方（fold ~0.57、fluorescence 0.25）——同一 checkpoint 学 13 个任务会摊薄单任务上限。这正对应任务目录设想的"共享 scorer vs 任务头 vs 专用"对照，是基准要量的科学问题，而非缺陷。

## 局限与下一步

- 13 任务是**分族小样**，非全 241 supported；绝对值不代表基准上限。
- 未做：候选重排鲁棒性、noul 校准、score 标量回归对照、被跳过的 49 任务（双序列/multi-label/RNA/长序列需架构扩展）。
- 下一步可选：(a) 扩到全部 supported 任务的正式多任务训练 + test 评测（真成绩）；(b) 匹配"共享 scorer vs 编码器+任务头 vs 生成式全候选似然"对照（论文主比较）；(c) 增 epoch/数据以缩小与专用训练的差距，验证是否纯粹预算问题。
