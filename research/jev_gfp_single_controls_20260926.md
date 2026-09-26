# JEV GFP 单任务 seed 对照结果

完成记录日期：2026-09-26 UTC。运行目录为 [`artifacts/laya_jev_gfp_single_seed_controls/run`](../artifacts/laya_jev_gfp_single_seed_controls/run)，机器可读汇总为 [`summary.json`](../artifacts/laya_jev_gfp_single_seed_controls/run/analysis/summary.json)。

## 结论

GFP 单任务对照完成：新增 5 个 fluorescence-only run，并复用原已审计 `cpt_fluorescence` 作为 seed 20260926 CPT 单任务参照。两个 smoke 与五个正式训练均完成，新增模型均保留；所有正式 run 都完成 384 次 fluorescence 更新、每个 GFP 训练实体三次曝光、精确 checkpoint 重载检查、有限且非零的 encoder/scorer 梯度核验；未读取 test 标签，也未执行 test 推理。

最重要的诊断是：**seed 20260928 NO-CPT 的联合训练近常数失败，在匹配的 GFP 单任务训练中没有复现。** 对应联合 run 的 dev RMSE/MAE/Spearman/预测 SD 为 0.8345/0.6241/0.1929/0.0127；匹配单任务末轮为 0.6303/0.4781/0.5392/0.5637，并且固定 train 诊断也恢复到 RMSE 0.6123、MAE 0.4696、预测 SD 0.5562。这说明该 seed 的联合近常数输出不能归因于数据 split、初始预测、GFP batch 顺序或 Score 目标在单任务预算下必然失败；联合训练的任务交替、optimizer/dropout 历史、任务竞争或调度路径应进入下一轮重点诊断。

同时，这轮也不支持“单任务 GFP 已稳定解决”的说法。六个单任务参照中只有 seed 20260928 NO-CPT 同时通过 RMSE 与 MAE 门槛；其余单任务最多只通过 RMSE，MAE 均未低于训练中位数常数 0.5095。因此当前 GFP Score recipe 仍然 seed 敏感，不能因为一个单任务恢复样本就宣布小样本三曝光设置稳定。

## 末轮结果

冻结常数基线：GFP 训练均值 RMSE 0.8357，训练中位数 MAE 0.5095。预测 SD < 0.05 预标记为近常数输出。

| Seed | Arm | Single RMSE ↓ | Single MAE ↓ | Single Spearman ↑ | Single pred SD | Single gate | Matched joint RMSE ↓ | Matched joint MAE ↓ | Matched joint pred SD | Joint gate |
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 20260926 | NO-CPT | 0.8106 | 0.6460 | 0.2817 | 0.1123 | fail | 0.6993 | 0.4921 | 0.3689 | pass |
| 20260926 | CPT | 0.7066 | 0.5411 | 0.4341 | 0.3626 | fail | 0.7512 | 0.6229 | 0.3148 | fail |
| 20260927 | NO-CPT | 0.7686 | 0.5953 | 0.3675 | 0.2848 | fail | 0.6910 | 0.5279 | 0.4212 | fail |
| 20260927 | CPT | 0.8207 | 0.6250 | 0.3016 | 0.0966 | fail | 0.8160 | 0.6249 | 0.1137 | fail |
| 20260928 | NO-CPT | **0.6303** | **0.4781** | **0.5392** | 0.5637 | **pass** | 0.8345 | 0.6241 | **0.0127** | fail |
| 20260928 | CPT | 0.8296 | 0.6178 | 0.2627 | 0.0819 | fail | 0.7583 | 0.5200 | 0.2408 | fail |

三 seed 描述统计：

| Metric | NO-CPT single mean ± SD | NO-CPT joint mean ± SD | CPT single mean ± SD | CPT joint mean ± SD |
|---|---:|---:|---:|---:|
| RMSE ↓ | 0.7365 ± 0.0943 | 0.7416 ± 0.0806 | 0.7856 ± 0.0686 | 0.7752 ± 0.0355 |
| MAE ↓ | 0.5731 ± 0.0861 | 0.5480 ± 0.0683 | 0.5946 ± 0.0465 | 0.5893 ± 0.0600 |
| Spearman ↑ | 0.3961 ± 0.1311 | 0.3610 ± 0.1462 | 0.3328 ± 0.0899 | 0.3516 ± 0.0691 |
| Prediction SD | 0.3203 ± 0.2278 | 0.2676 ± 0.2223 | 0.1804 ± 0.1580 | 0.2231 ± 0.1017 |

These SDs are descriptive over three fixed training seeds only; development data are reused and GFP variants share the native parent split. They are not statistical significance intervals.

## Audit Notes

The independent summary checked:

- 6 single-task references: 5 new retained checkpoints plus the reused seed 20260926 CPT single-task output.
- 6 matched joint runs from the prior seed confirmation round.
- Status, update counts, exact reload, retained model hashes when present, no test inference.
- 384 fluorescence updates per single-task run; 1,152 updates per joint run.
- Finite and nonzero encoder/scorer gradients for all single-task traces.
- Per-entity exposure histogram `{3: 8192}` for each single-task formal run.
- Matching score spec, data manifest, initial model hash, initial shared head hash, and fluorescence batch-order hash between matched single and joint runs.
- Identical epoch-0 fluorescence development predictions for each matched single/joint pair.

## Decision

This round narrows the next question: the most suspicious mechanism is no longer “GFP Score cannot train under this budget at all.” The cleaner next diagnostic is targeted at joint-training dynamics, especially for seed 20260928 NO-CPT, where the single-task run succeeds but the joint run collapses. Good next experiments are schedule/interaction tests that keep data, seed, total target exposure, JEV output, and evaluation fixed while changing only the joint training path: for example task blocking versus interleaving, fluorescence loss/task weighting, optimizer-state isolation, or gradient conflict measurements.

At the same time, because four of five newly considered single-task arms and the reused CPT single-task arm still fail the MAE gate, the paper story should keep GFP as an unstable diagnostic, not a solved capability claim. No Qwen/OmniGene4 comparison should be launched until this Laya-side instability is better isolated or deliberately moved to a full-data/dose experiment.

