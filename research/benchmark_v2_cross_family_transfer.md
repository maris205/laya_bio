# Benchmark v2 跨族 held-out 零样本迁移（对 #1 迁移主张最公平的检验）

日期：2026-10-04。代码：`train_benchmark_v2.py --exclude-family`（新增整族留出）+ `eval_benchmark_v2.py`。结果 `artifacts/benchmark_v2_train/bv2_xfam/`、`artifacts/benchmark_v2_eval/xfam_gb_{init,trained}/`、`cross_family_transfer.json`。

## 动机

`benchmark_v2_zeroshot_transfer.md` 的同族 held-out 测试（GUE→GUE、PG→PG）偏弱：held-out 任务与训练任务同族同类型，测不出"通用决策函数"。本文做**整族留出**：训练时完全不见 GenomicBenchmarks（GB）族，再零样本评 GB——检验在其它族上联合训练能否迁移到**全新任务族**。

## 设置

- 训练：`--all --exclude-family GenomicBenchmarks`，6 族 29 任务（GUE12/PG12/dnagpt3/local5/TAPE2/DeepSTARR2），family-balance、warmup 0.15、clip 0.5、dev 早停（best-dev step1600, dev 0.4902）、2160 updates、max 2048/任务。
- 评测：held-out 的 8 个 GB 任务，trained(`model.safetensors`) vs frozen(`init.safetensors`，同 base 未训练)，同 seed/200 样本子集，配对。
- GB 从未进入训练；#2 任务头无法参与（无 GB 头），故本测试只针对 #1。

## 结果：弱正跨族迁移，≈ in-domain

| GB held-out 任务 | prim | metric | frozen | trained | Δ迁移 |
|---|---|---|---|---|---|
| drosophila_enhancers_stark | noul | AUROC | 0.350 | 0.600 | **+0.250** |
| human_nontata_promoters | noul | AUROC | 0.560 | 0.743 | **+0.183** |
| demo_human_or_worm | noul | AUROC | 0.482 | 0.625 | +0.143 |
| human_enhancers_cohn | noul | AUROC | 0.606 | 0.660 | +0.054 |
| human_ocr_ensembl | noul | AUROC | 0.527 | 0.564 | +0.037 |
| human_ensembl_regulatory | choice | acc | 0.290 | 0.285 | −0.005 |
| demo_coding_vs_intergenomic | noul | AUROC | 0.346 | 0.168 | −0.178 |
| dummy_mouse_enhancers | noul | AUROC | 0.660 | 0.476 | −0.183 |

**mean Δ = +0.038，正迁移 5/8**（>0.03 判正）。对照：GB 在 full2 中被直接训练时的 in-domain mean Δ ≈ **+0.035**（注：full2 预算/早停配置与 xfam 不完全匹配，仅作量级参照）。

## 判读（诚实）

1. **跨族零样本迁移弱正、且 ≈ in-domain**：在 6 个族上联合训练后，对**完全未见的第 7 族（GB）**零样本，平均提升 +0.038，与"直接训练 GB"的 in-domain 提升（+0.035）量级相当。这暗示收益主要来自**通用的共享决策改进**（能跨族迁移），而非族特定拟合——对 #1"通用决策函数"主张是**温和的正面证据**。
2. **比同族 held-out 略好**：同族零样本（zeroshot_transfer.md）是 wash（GUE −0.018）；跨族 GB 这里 +0.038、5/8 正。差异可能因 GB 以 noul 为主（7/8），而 noul AUROC 在本设置下迁移较正；同族测试的 GUE 以 choice 为主（迁移 −0.018）。
3. **但效应小、方差大、不一致**：范围 −0.183 到 +0.250；两个任务明显负迁移（demo_coding、dummy_mouse）。**不能宣称强正迁移**——准确表述：存在弱的、任务依赖的跨族零样本迁移，平均量级与 in-domain 相当，但远未达到"零样本即可用"。
4. 结合前序：#1 在已见任务不优于 #2（seed 平均）、同族零样本 ≈ base、跨族零样本弱正（+0.038）。**跨族测试是三个迁移测试里唯一略正的**，但幅度不足以扭转"#1 性能不占优"的总体结论；它支持的是"#1 的共享决策确有一定通用性"这一较弱主张。

## 局限与下一步
- 只留出一个族（GB）、单次 run（未 seed 平均，方差未知）；noul 偏多。应补：留出多个族、多 seed、并留出 score 族（TAPE/DeepSTARR）测 score 跨族迁移。
- in-domain 对照来自不同配置的 full2，非严格 matched；严格版应在 xfam 同配置下加训 GB 作 in-domain 臂。
- 若要把"通用决策函数"作为正面主张，需更大预算/更多族/seed 平均确认 +0.038 是否稳健、能否随规模增长。
