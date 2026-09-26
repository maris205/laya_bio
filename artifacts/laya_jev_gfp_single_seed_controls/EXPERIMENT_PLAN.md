# JEV GFP Single-Task Seed Controls

Date: 2026-09-25 UTC

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence |
|---|---|---|
| C1: GFP instability is separable from the classification result. | The paired seed confirmation showed stable Noul/Choice behavior but unstable Score behavior. | Same-data GFP-only controls reproduce or remove the failure under matched seeds and supervision budget. |
| C2: A joint-vs-single comparison can diagnose the next mechanism to study. | We need to know whether to inspect multitask scheduling/optimizer history or Score training itself. | Same seed/arm single-task controls are compared against the already audited joint runs using RMSE, MAE, Spearman, prediction SD, and train diagnostics. |

## Fixed Runs

No test labels will be read and no test inference will be run. The development split is reused only for diagnosis against the already frozen criteria.

| Run ID | Encoder | Seed | Task | Updates | Status Source |
|---|---|---:|---|---:|---|
| reuse-cpt-gfp-20260926 | CPT | 20260926 | fluorescence only | 384 | Existing audited `cpt_fluorescence` |
| no-cpt-gfp-20260926 | NO-CPT | 20260926 | fluorescence only | 384 | New formal run |
| no-cpt-gfp-20260927 | NO-CPT | 20260927 | fluorescence only | 384 | New formal run |
| cpt-gfp-20260927 | CPT | 20260927 | fluorescence only | 384 | New formal run |
| no-cpt-gfp-20260928 | NO-CPT | 20260928 | fluorescence only | 384 | New formal run |
| cpt-gfp-20260928 | CPT | 20260928 | fluorescence only | 384 | New formal run |

The five new formal runs use the immutable round-1 JEV trainer with an injected seed. Each run uses the same 8,192 GFP training entities, full 5,362-row development split, three complete target-task exposures, JEV Score anchors, optimizer, and single-task replay of joint global learning-rate positions.

## Preflight and Smoke

- Local GPU must report memory used < 500 MiB before launch.
- Available disk under `/root/autodl-tmp` must exceed 20 GiB.
- Run one NO-CPT and one CPT fluorescence-only train smoke before formal training.
- Each smoke must finish one fluorescence update, produce finite nonzero encoder/scorer gradients, and pass exact checkpoint reload.

## Metrics

| Metric | Role | Rule |
|---|---|---|
| GFP RMSE | Primary error metric | Passes if below train-mean constant baseline 0.8357 |
| GFP MAE | Practical gate | Passes if below train-median constant baseline 0.5095 |
| Spearman | Ranking diagnostic | Report only; not a standalone pass gate |
| Development prediction SD | Collapse diagnostic | SD < 0.05 is predeclared near-constant behavior |
| Fixed train diagnostic RMSE/MAE/SD | Training-behavior diagnostic | If train and development are both near constant, treat as training behavior rather than dev-only noise |

## Audit Requirements

- Verify run status, update count, exact reload status, retained model hash for new formal runs, and absence of test inference.
- Verify all formal traces have 384 fluorescence updates with finite nonzero encoder and scorer gradients.
- Verify each trained GFP entity appears exactly three times.
- Compare each single-task run to its matched joint run on seed, arm, data manifest, score spec, initial shared head hash, and fluorescence batch-order hash.
- Compare epoch-0 fluorescence development predictions between matched single and joint runs to confirm identical initialization behavior.

## Decision Rules

- If seed 20260928 NO-CPT recovers in GFP-only training while the joint run remains near constant, prioritize multitask scheduling, optimizer history, task weighting, or gradient-conflict analysis.
- If seed 20260928 NO-CPT remains near constant in GFP-only training, pause joint-specific explanations and inspect Score loss, target scaling, learning rate, number of updates, and supervision dose.
- If several GFP-only runs miss the MAE gate, do not treat the current 8,192-entity, three-exposure recipe as stable; move to a data-dose or full-pool experiment.
- Do not claim CPT is generally beneficial or harmful from this control unless matched seed differences are stable and diagnostic.

