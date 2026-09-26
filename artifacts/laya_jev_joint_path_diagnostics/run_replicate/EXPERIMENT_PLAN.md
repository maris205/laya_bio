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

## Replicate round: run-to-run variance (frozen 2026-09-26)

Motivation: two identical `task_block_gfp_last_fluo0p5` runs (seed 20260928, same data order and code) had bit-identical training traces for 137 steps, then diverged from floating-point GPU nondeterminism; their epoch-2 GFP dev SD was 0.058 vs 0.198. Every joint-path comparison so far is a single run, so run-to-run variance must be measured before interpreting the GFP-last vs GFP-first gap as a real effect.

| Run | Change | Purpose |
|---|---|---|
| task_block_gfp_last rep1, rep2 | None (identical to `run/round/seed_20260928_no_cpt_joint_task_block_gfp_last`) | Variance of the recovered path |
| task_block_gfp_first rep1, rep2 | None (identical to `run_followup/round/seed_20260928_no_cpt_joint_task_block_gfp_first`) | Variance of the weak path |

Output root: `run_replicate`. Seed, data, NO-CPT initialization, output contract, exposure, optimizer, LR schedule, loss, evaluation and no-test rule are unchanged. One smoke per variant, then four 1,152-update formal runs, alternating variants so both conditions share similar machine state.

Decision rule (with the original run, n = 3 per variant):

- If every GFP-last run beats every GFP-first run on GFP dev RMSE and MAE, and the gap between means exceeds the within-variant range, treat GFP-last recency as a reproducible effect and proceed to optimizer-state isolation or gradient-conflict probes.
- If ranges overlap, record the task-order effect as not separable from run-to-run noise at n = 3; do not build mechanism claims on it.
- In either case, record the per-variant gate pass rate (RMSE and MAE gates) as the practical stability measure.
