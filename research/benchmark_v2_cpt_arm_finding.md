# CPT 臂发现：benchmark v2 全部训练此前跑在 no_cpt 基座上（2026-10-07）

## 现象

排行榜（B 实验）首族 dnagpt_pools 专项训练后，#1 与 #2 两臂测试 acc **逐任务完全相同**
（core 0.475 / splice 0.560 / tf 0.455），macro_f1 0.24–0.32。核对测试集多数类占比
（0.503 / 0.552 / 0.516，200 样本子集内 0.475/0.55/0.455）确认：**两臂都坍缩为常数预测**。
此前 joint-37 对照里 dnagpt 族 #1 0.519 / #2 0.514 同样贴近多数类均值——该族在 no_cpt 下
从未被真正学会。

## 根因

MVP1 frozen 码 `build(model_dir, cpt_root, arm)` 仅在 `arm=='cpt'` 时加载
`laya_biocpt_v2/round/cpt/model.safetensors`（生物 CPT 权重）。benchmark v2 两个训练器
传的是 `no_cpt`/`None`——即**只有生物 BPE 扩表、无生物 CPT 的基座编码器**。
这是 pilot 文档记录的刻意选择（"干净零样本起点"），不是手误；但后果是：

1. DNA 族任务在 no_cpt 基座上普遍无泛化信号：train loss 0.59→0.31 而 dev 从 step 400 起
   钉死在多数类水平（0.479），纯记忆无迁移。
2. §6 的"专用锚点"（promoter 87–89%、fold 56–58%）来自 **cpt 臂**（MVP1 轮次目录
   `cpt_promoter` 等，日志 test acc 0.883/0.885）——与 no_cpt 的 joint 数字对比是跨臂错配。
3. 排行榜若用 no_cpt 永远出不了好数字。

## 处置

- 两训练器加 `--arm {no_cpt,cpt}`，**默认 no_cpt**：既有全部实验（3-seed 主对照、消融、
  迁移、适配成本）口径不变、可复现，论文主表继续以 no_cpt 为"干净迁移 regime"。
- 排行榜 driver 与 stage2（cpt 主对照 3-seed）用 `--arm cpt`：实用配方口径。
- 冒烟验证：cpt 臂 200 updates dev 0.491（> no_cpt 平台 0.479）且仍上升。
- 论文叙事：双口径互补——主对照（no_cpt，架构结论）+ 排行榜/主对照-cpt（实用配方）；
  §4.1 backbone 描述与 §6 锚点句补臂口径声明；若 cpt 主对照结论与 no_cpt 一致，
  "架构结论对编码器初始化稳健"成为额外主张；若翻转，本身即发现。
- 免费消融：s1c_20261001（cpt）vs s1_20261001（no_cpt）= CPT 对联合决策训练的贡献，
  单 seed 同配置。

## 运行队列（2026-10-07 12:57 起）

1. `leaderboard_driver.sh`（cpt，7 族 × #1/#2，单 seed）≈8–14h
2. `stage2_cpt_main.sh`（cpt，joint-37，3 seeds × #1/#2，逐 seed 剪枝）≈9h，链式等待 1 完成
3. 聚合：`gen_leaderboard.py` → TABLE_7；cpt 主对照聚合脚本待写（仿 seed_avg 聚合）

## 首族 cpt 结果（2026-10-07 13:13，lb2_dnagpt_pools）

专项+cpt：dna_core **0.475→0.625**（macro_f1 0.322→0.613，真学会）；dna_splice 0.560、
dna_tf 0.455 仍钉多数类（dev 曲线 0.465–0.523 噪声波动，best 0.523@1600）。判读：cpt
修复了 core 的泛化；splice/tf 为旧盲测池长序列任务（motif 埋于数百 bp），对该编码器+
预算仍难——旧论文此二题高分来自生成式指令微调 LLM，机制不同。排行榜如实报告弱项，
"好结果"成色由 local_snapshots（promoter 锚点）与 GUE（DNABERT-2 公开对照）决定。
