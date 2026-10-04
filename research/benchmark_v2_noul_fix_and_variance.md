# #1 noul 措辞修复 + 一个更重要的方差发现

日期：2026-10-04。承接三方主对照（`benchmark_v2_three_way.md` §0）。代码 `noul_ablation.py`；结果 `artifacts/benchmark_v2_eval/{noul_ablation.json,pf2_*,three_way_pf2_summary.json}`、`bv2_pf2/`。

## 1. noul 措辞修复（受控消融，干净证据）

#1 noul AUROC(0.577) 落后 #2 专用头(0.620)。AUROC 是排序指标，温度校准改不了它——是判别力问题。利用 #1 动态候选特性做**纯推理消融**（同一 bv2_pf 全局 checkpoint，7 个 GB noul 任务，只改候选文本）：

| 候选措辞 | mean AUROC |
|---|---|
| `false: no` / `true: yes`（旧） | 0.479 |
| **`no` / `yes`（原始，最佳）** | **0.545** |
| `negative` / `positive` | 0.535 |
| `false` / `true` | 0.511 |
| descriptive 长句 | 0.493 |
| `no, it does not` / `yes, it does` | 0.434 |

**结论**：冗余的 `false:/true:` 前缀最差；改用 benchmark 原始候选 `["no","yes"]` 在**同一 checkpoint 上 +0.066 AUROC**。这是受控（唯一变量=措辞）的干净证据。已把 trainer + harness 的 noul 候选改为原始 `list(candidates)`（commit c8162c1）。#2 用固定头、#3 已用原始 yes/no，均不受影响。

## 2. 端到端重训无法确认修复——因为 run 方差更大

为端到端验证，用修复后措辞重训 #1 按族早停（bv2_pf2，同 37 任务/同 5600 预算/同 seed）。结果**反而更差**：

| prim | metric | #1 pf(旧措辞) | #1 pf2(新措辞) | #2 头 |
|---|---|---|---|---|
| choice | acc | 0.512 | 0.501 | 0.500 |
| noul | auroc | 0.577 | 0.545 | 0.620 |
| score | spear | 0.107 | **0.015** | −0.007 |

**关键**：score 从 0.107 暴跌到 0.015，但 **score 任务根本不用 noul 措辞**——所以 pf↔pf2 的差异**不是措辞造成的，是纯 run 方差**。逐族 score 全跌：TAPE +0.167→+0.049、ProteinGym +0.094→+0.002、DeepSTARR +0.081→+0.011。

根因（per-family best step 在两次 run 间差异巨大）：

| family | pf best step (dev) | pf2 best step (dev) |
|---|---|---|
| GUE | 200 (0.469) | 5000 (0.481) |
| GenomicBenchmarks | 1200 (0.579) | 200 (0.521) |
| ProteinGym | 2200 (0.525) | 600 (0.556) |
| TAPE | 1600 (0.619) | 1600 (0.579) |
| DeepSTARR | 3600 (0.596) | 2600 (0.585) |
| local_snapshots | 5200 (0.523) | 2600 (0.478) |

同配置同 seed，但 GPU 非确定性使两条轨迹分叉；**per-family 早停用 96 个 dev 样本选 step，噪声大**，pf2 选到的 step（如 GB@200 欠训、ProteinGym@600）dev 分数尚可但 test 泛化更差。

## 3. 方法论结论（重要）

1. **noul 措辞修复成立**，但证据来自**受控推理消融**（+0.066，同 checkpoint），**不能**用 pf-vs-pf2 端到端对比证明——后者被 run 方差淹没。
2. **单次 run 的三方数字有 ±0.05–0.1 的噪声带**：per-family 早停 + 小 dev + GPU 非确定性使同配置重跑结果显著不同。此前 §0 的"刷新三方"应视为**一次抽样**，非稳定期望。
3. **per-family 早停在 96 dev 样本上不稳定**：选点噪声大，dev→test 泛化不一致（pf2 dev 与 pf 相近但 test 更差）。

## 4. 对主对照结论的影响

- 核心主张（#1 共享打分器 ≈/≥ #2 任务头、二者 > frozen #3）**方向不变**：choice/score 上 #1 领先、noul 上 #2 领先，在 pf 与 pf2 两次抽样里都成立（noul #2 始终 ≥ #1）。
- 但**精确数值不可靠**，需 seed 平均才能给论文主表。
- noul 差距：受控消融说明措辞可给 #1 +0.066，理论上能把 #1 noul 推到 ~0.64（可能反超 #2 的 0.620），但需在同一稳定 checkpoint 上验证，而非重训对比。

## 5. 下一步（修正方法论）
1. **seed 平均**：#1/#2 各跑 3 seed，报 mean±std，才能得到可靠主表（当前单次数字噪声太大）。
2. **稳定 per-family 选点**：加大 dev 样本（96→256+）或改用 dev 滑动平均，减少选点噪声。
3. **noul 措辞**：在稳定 checkpoint 上用原始 `no/yes`（已设为默认），并验证是否反超 #2。
4. 在此之前，三方主对照标注为"单次抽样、含方差"，不过度解读小数点差异。
