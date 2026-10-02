# Benchmark v2 全量训练第一版真成绩

日期：2026-10-02。代码 `scripts/benchmark_v2/train_benchmark_v2.py`（任务无关、family-balanced 多任务共享决策训练器）。结果 `artifacts/benchmark_v2_train/bv2_full{,2}/`、`artifacts/benchmark_v2_eval/{full,f2}_{init,trained}/`。

## 设计

单一共享 Laya-JEV 模型在 benchmark v2 的 supported 任务（单序列 DNA/蛋白 noul/choice/score）上训练，test 评测。起点 `build(laya_model, laya_biocpt_v2, no_cpt)` 并保存 `init.safetensors` 作**同一 init 的干净零样本基线**；评测对 init 与 trained 用**相同 seed、相同 200 样本子集**做配对差。

**为什么 family-balance 而非 per-task**：supported 任务 265 个里 217 是 ProteinGym assay、28 是 GUE，若按任务等预算，这两个大族会淹掉其余多样族；改按族等预算（每族每 epoch 固定 batch 数，族内 round-robin 轮转任务）。

## 两次运行

### full（全 96 纳入任务，欠训对照）
`--all --cap ProteinGym=60`，family-balance，epochs 4 × bpf 10 = **280 updates**（每任务~3）。结果 mean Δ **+0.034**（32↑/52平/12↓）。小族有信号（TAPE +0.15、GB +0.12、DeepSTARR +0.10、dnagpt +0.09），大族被稀释（GUE 28 任务、ProteinGym 49 任务几乎没学到）。**结论：预算不足，非方法失败。**

### full2（充分预算的第一版真成绩）
cap GUE=12、ProteinGym=12（共 43 任务），epochs 12 × bpf 30 = **2520 updates**（每族~360，每任务 30–180）。mean Δ **+0.053**（18↑/19平/6↓）。

| 族 | n | 指标 | init(零样本) | trained | Δ |
|---|---|---|---|---|---|
| TAPE | 2 | spearman | −0.090 | 0.237 | **+0.327** |
| dnagpt_pools | 3 | acc | 0.433 | 0.527 | +0.093 |
| DeepSTARR | 2 | spearman | −0.032 | 0.048 | +0.081 |
| ProteinGym | 12 | spearman | −0.024 | 0.046 | +0.070 |
| local_snapshots | 5 | acc | 0.479 | 0.545 | +0.066 |
| GenomicBenchmarks | 7 | AUROC | 0.484 | 0.519 | +0.035 |
| GUE | 12 | acc | 0.478 | 0.457 | **−0.021** |

单任务最大提升：tape_stability spearman −0.218→0.324、pg_MTHR −0.166→0.177、dna_splice acc 0.28→0.56、gb_nontata AUROC 0.478→0.723。

## 判读

1. **基准能训练、能判别**：充分预算下多数族提升、部分任务学得强；GUE（组蛋白二分类）卡在多数类基线学不动——基准正确地暴露了"哪些任务在该架构/预算下难学"。这正是评测基准该有的功能。
2. **预算敏感**：full（280 upd）mean Δ +0.034 < full2（2520 upd）+0.053 < pilot13（每任务~90 upd）+0.186。每任务更新数越低，多任务摊薄越重，单任务上限越低。要逼近专用性能需更大预算或减小任务间稀释。
3. **joint vs 专用的差距**：full2 的 local_snapshots（含 fold_class）acc 仅 0.545、比 pilot 里专用充分训练的 fold 0.435→仍低于专用 3-任务配方 ~0.57；体现共享单一 checkpoint 学多族的 per-task 上限损失，是"共享 scorer vs 任务头 vs 专用"主对照要量的东西。
4. **ProteinGym 部分 assay 可学**（spearman 到 0.23），但整族 mean 仅 0.046——任意新蛋白突变适应度仍是最难的方向。

## 局限
- 绝对值低主要受限于：每任务小数据上限（max 768）、短预算（2520 upd）、512-token 窗口（长序列/长上下文任务受损）、单一 checkpoint 跨 43 任务摊薄。这不是基准上限，是"首个 joint 小样"的量级标定。
- 未训/未评：被跳过的 49 任务（双序列/multi-label/RNA/超长）；noul/score 校准；候选重排鲁棒性。

## 下一步
(a) 更大预算或分族专项训练以逼近专用；(b) 主对照：共享 scorer vs 编码器+任务头 vs 生成式全候选似然（matched compute）；(c) GUE 学不动的归因（长上下文？类别不均衡？问题/候选文本？）。
