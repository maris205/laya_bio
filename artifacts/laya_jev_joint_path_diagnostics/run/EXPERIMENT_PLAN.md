# JEV joint-path diagnostic plan

Updated: 2026-09-26 UTC.

## Claim under test

The seed 20260928 NO-CPT GFP collapse is caused by the joint training path, not by the fixed data split, seed, JEV Score output, GFP sample exposure, or evaluation code.

## Frozen scope

- Seed: 20260928.
- Arm: NO-CPT.
- Data: existing JEV multitask train/dev manifest from `artifacts/laya_jev_multitask_v1/data`.
- Output contract: unchanged Noul, Choice, Score panels and shared marker scorer.
- GFP exposure: unchanged 8,192 train entities x 3 exposures = 24,576 presentations.
- Evaluation: unchanged dev set, train diagnostic subset, RMSE/MAE/Spearman/prediction-SD gates.
- Test access: none.

## First diagnostic run

| Run | Change | Expected evidence |
|---|---|---|
| task_block_gfp_last | Replace per-batch random task interleaving with per-epoch task blocks: promoter, structural_class, fluorescence. Keep per-task shuffles, total updates, optimizer, LR schedule, loss, model, seed, data, and evaluation fixed. | If GFP recovers relative to the matched interleaved joint baseline, task path/order is a credible trigger. If it stays near-constant, move to task weighting or optimizer-state isolation. |

Matched references:

- Failed interleaved joint baseline: `artifacts/laya_jev_multitask_seed_confirm/run/round/seed_20260928_no_cpt_joint`.
- Recovered GFP-only reference: `artifacts/laya_jev_gfp_single_seed_controls/run/round/seed_20260928_no_cpt_gfp_single`.

## Gates

- GFP RMSE must be below the training-mean constant baseline of 0.8357.
- GFP MAE must be below the training-median constant baseline of 0.5095.
- Prediction SD below 0.05 marks near-constant output.
- Train diagnostic and dev must be interpreted together.

## Decision rule

- If `task_block_gfp_last` restores GFP without hurting the basic joint contract, continue path diagnostics around task ordering.
- If it remains near-constant, run a second small diagnostic changing only fluorescence task weighting or optimizer-state isolation.
- If multiple joint-path variants still collapse while GFP-only remains healthy, update the roadmap away from this joint recipe instead of repeatedly tuning the same dev slice.
