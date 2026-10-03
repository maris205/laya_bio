# Benchmark v2 按族早停：突破单一 checkpoint 的平台

日期：2026-10-03。承接稳定化实验（`benchmark_v2_stabilization.md`）。代码：`train_benchmark_v2.py` 加 `--per-family-early-stop`（dev_eval 返回每任务指标→按族聚合→每族保存各自 best-dev checkpoint `best_<family>.safetensors`）。结果 `artifacts/benchmark_v2_train/bv2_pf/`、`artifacts/benchmark_v2_eval/pf_{global,<family>}/`、`per_family_vs_global.json`。

## 动机

稳定化实验发现：单一全局 best-dev checkpoint 对异质任务是折中——不同族在训练轨迹上**峰值 step 不同**，全局选一个 step 必然牺牲某些族。本轮验证"按族早停"（每族用各自 best step 的 checkpoint）能否突破该平台。

## 设置

同 stab 预算（37 任务、5600 updates、max 2048、warmup 0.15、clip 0.5、dev-eval 每 200 步），加 `--per-family-early-stop`。**同一训练轨迹**，只比较 checkpoint 选择：全局 best（step 2200，promoted 为 model.safetensors）vs 每族 best（best_<family>.safetensors）。

各族 best step（差异极大，证实动机）：

| family | best step | best dev |
|---|---|---|
| GUE | 200 | 0.469 |
| GenomicBenchmarks | 1200 | 0.579 |
| TAPE | 1600 | 0.619 |
| dnagpt_pools | 1800 | 0.522 |
| ProteinGym | 2200 | 0.525 |
| DeepSTARR | 3600 | 0.596 |
| local_snapshots | 5200 | 0.523 |

## 结果（test，同一 run，仅 checkpoint 选择不同）

| family | metric | n | 全局 best | 按族 best | Δ |
|---|---|---|---|---|---|
| local_snapshots | acc | 5 | 0.442 | **0.549** | +0.106 |
| dnagpt_pools | acc | 3 | 0.435 | **0.540** | +0.105 |
| GenomicBenchmarks | auroc | 8 | 0.487 | **0.569** | +0.082 |
| TAPE | spearman | 2 | 0.094 | **0.167** | +0.073 |
| GUE | acc | 12 | 0.438 | **0.491** | +0.053 |
| DeepSTARR | spearman | 2 | 0.053 | 0.081 | +0.028 |
| ProteinGym | spearman | 5 | 0.094 | 0.094 | +0.000 |

逐任务：**按族胜 21 / 平 9 / 全局胜 7**；**每个族 ≥ 全局**（无一族因按族早停变差）。

（注：跨族"ALL"均值混合了 acc/auroc/spearman 不同口径，仅作粗略参考；结论以每族 within-metric 的 Δ 为准，全部 ≥0。）

## 判读

1. **按族早停确实突破单一 checkpoint 平台**：同一训练轨迹下，仅把"全局一个 step"换成"每族各自 best step"，所有族 test 指标持平或提升，多数显著提升（local/dnagpt +0.10、GB +0.08、TAPE +0.07）。这直接证实稳定化实验的猜想——平台部分来自"异质任务共享单一 checkpoint 选择"的折中，而非纯粹容量上限。
2. **机制清晰**：族峰值 step 从 200（GUE）到 5200（local）跨 26 倍，全局 step 2200 对 GUE 偏晚、对 local/DeepSTARR 偏早；按族选点各取所需。
3. **ProteinGym 持平**：其 best step（2200）恰接近全局 best，故无增益；score 小数据族（TAPE）增益明显（呼应稳定化文档的预判）。
4. 这是**推理期零成本**的改进（训练时多存几个 checkpoint，评测时每族用各自的），不需重新训练或改架构。

## 局限与下一步
- 按族粒度仍粗：族内任务（如 GUE 12 个、ProteinGym 5 个）峰值也可能不同，**按任务早停**（每任务存 best）可能再提升，但存储 = 任务数 × 1.6G，需权衡（可只存 head/scorer 差异或低精度）。
- 仍未解决 joint 训练的**根本干扰**：按族早停是从同一受干扰轨迹里挑各族最优点，若改为**分组训练**（每族/每组独立模型）可能进一步去干扰，但牺牲"单一共享模型"的简洁性——这正是"共享 vs 专用"权衡的延伸。
- 下一步可选：(a) 按任务早停；(b) 分组训练对照；(c) 把按族早停设为默认，重跑 #1/#2/#3 三方对照（更公平的 #1）；(d) 回到多模态试点。
