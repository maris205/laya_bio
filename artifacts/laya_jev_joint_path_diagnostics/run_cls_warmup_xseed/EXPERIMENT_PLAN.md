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

## Block-evaluation round: where does GFP-first fail? (frozen 2026-09-27)

Motivation: the train-only interference probe showed that 256 post-GFP classification updates do not collapse a well-trained GFP state, so forgetting after the GFP block is not the main GFP-first failure mechanism. Hypothesis: the GFP-first GFP block never reaches a good GFP state, with epoch 1 (GFP trained from initialization during LR warmup) the first suspect.

| Run | Change | Purpose |
|---|---|---|
| task_block_gfp_first_blockeval | Identical training to `task_block_gfp_first`; after each 128-update task block, score the fixed train-diagnostic panel (1,024 rows per task) | Trace GFP state after every block |
| task_block_gfp_last_blockeval | Same for `task_block_gfp_last` | Matched reference trace |

Output root: `run_blockeval`. Evaluation is eval-mode inference on train rows only (no RNG use, no optimizer change); training data, order, seed, optimizer, LR, loss and the epoch-end dev evaluations are unchanged. No test access.

Reading rule:

- If GFP-first's GFP train-diagnostic prediction SD stays below 0.1 right after its epoch-1 GFP block while GFP-last reaches SD above 0.3 after its first GFP block, epoch 1 is implicated; the next step is an epoch-1 swap control.
- If GFP-first reaches a good state after some GFP block and then loses it during later classification blocks, forgetting under the real optimizer state and higher LR is implicated (the probe used fresh AdamW at epoch-3 LR).
- If both orders look alike after their GFP blocks, the difference arises from later interaction and is not localized by this round.

## Round-robin granularity round: a practical joint recipe? (frozen 2026-09-27)

Motivation: random per-batch interleaving (switch every ~1 update) collapsed GFP on seed 20260928, while 128-update task blocks with GFP last recovered it; block diagnostics showed GFP fails when first trained from initialization and that high-LR blocks cause strong cross-task forgetting (promoter 82.6% -> 51.9% after one structure block). A fixed-cycle rotation every K updates sits between these regimes.

| Run | Schedule | Reps |
|---|---|---|
| round_robin_k8_gfp_last | Each epoch: cycle promoter -> structural_class -> fluorescence, 8 batches per task per turn (16 cycles) | 2 |
| round_robin_k32_gfp_last | Same with 32 batches per turn (4 cycles) | 2 |

Output root: `run_round_robin`. Per-task shuffles use the same RNG stream as the task-block variants, so each task sees the same batches per epoch; only the interleaving changes. Seed 20260928, NO-CPT, data, output contract, exposure (1,152 updates), optimizer, LR schedule, loss and dev evaluation unchanged. A train-diagnostic evaluation is recorded every 128 updates (eval mode, no RNG or optimizer change). No test access.

Success rule (per K, both reps required):

- GFP dev passes both gates (RMSE < 0.8357, MAE < 0.5095) and prediction SD >= 0.05;
- promoter dev accuracy >= 85% and structure dev accuracy >= 54% (no material loss versus task_block_gfp_last: 86.7-87.8% / 54.2-55.4%).

If a K passes, it becomes the candidate joint recipe for cross-seed confirmation (seeds 20260926/20260927). If K=8 fails and K=32 passes, granularity is a controlling factor. If both fail, task-block GFP-last remains the only working path and mechanism controls (a)/(b) are next.

## LR-position swap round: is the GFP-first failure an LR effect? (frozen 2026-09-27)

Motivation: block diagnostics showed GFP-first leaves GFP constant after its epoch-1 and epoch-2 GFP blocks and starts learning only in epoch 3, while GFP-last starts in its first GFP block. Two explanations are confounded: (i) LR position (GFP-first GFP blocks get mean LR factors 0.78 incl. warmup / 0.74 / 0.28; GFP-last gets 0.88 / 0.42 / 0.11), and (ii) model/optimizer state already shaped by classification when GFP-last's first GFP block starts.

| Run | Update order | LR taken from | Reps |
|---|---|---|---|
| task_block_gfp_first_lr_of_last | GFP-first | GFP-last position of the same (epoch, task, index) update | 2 |
| task_block_gfp_last_lr_of_first | GFP-last | GFP-first position of the same update | 2 |

Only the schedule step used by the frozen LR function is remapped (also logged as `joint_schedule_step`); data, batch contents, update order, seed, optimizer, loss, exposure and evaluations are unchanged. Train-diagnostic evaluation after every 128-update block. Output root `run_lr_swap`. No test access.

Reading rule:

- If GFP-first with GFP-last LR passes the GFP gates 2/2 and GFP-last with GFP-first LR fails at least 1/2, LR position is the primary cause; next test a practical GFP-specific LR recipe.
- If GFP-first with GFP-last LR still leaves GFP constant after its epoch-1 GFP block (SD < 0.05) in both reps, LR position is not sufficient; the cause is state shaped by earlier classification updates (model or optimizer), and optimizer-state isolation is next.
- Mixed outcomes are recorded as partial LR contribution without a mechanism claim.

## Classification-warmup round: can standard interleaving avoid the GFP collapse? (frozen 2026-09-28)

Motivation: the LR-swap round showed GFP learns when its first training happens after classification updates, and collapses into a constant-output state when trained from initialization, independent of LR position. If so, the standard random interleave (which collapsed GFP on this seed) should work once a short classification-only warmup precedes any GFP update.

| Run | Schedule | Reps |
|---|---|---|
| interleave_cls_warmup64 | Frozen random interleave; the first 64 promoter and 64 structural_class batches of epoch 1 move to the front (128 classification-only updates), everything else in original order | 2 |
| interleave_cls_warmup128 | Same with 128 + 128 (all epoch-1 classification batches first, 256 updates) | 2 |

Batches, total updates (1,152), exposures (3 per entity), LR curve, optimizer, loss, seed 20260928 NO-CPT and evaluations are unchanged; LR warmup (57 updates) falls inside the classification warmup. Train-diagnostic evaluation every 128 updates. Output root `run_cls_warmup`. Matched failure reference: `artifacts/laya_jev_multitask_seed_confirm/run/round/seed_20260928_no_cpt_joint` (GFP dev SD 0.0127). No test access.

Success rule (per warmup length, both reps required): GFP dev RMSE < 0.8357, MAE < 0.5095 and SD >= 0.05; promoter dev accuracy >= 85%; structure dev accuracy >= 54%. A passing warmup becomes the candidate recipe for cross-seed confirmation (seeds 20260926/20260927). If GFP starts but a classification threshold fails, record the trade-off. If GFP stays constant after the warmup in both reps, the "first GFP training from initialization" account is incomplete and optimizer-state isolation is next.

## Cross-seed confirmation of the classification-warmup recipe (frozen 2026-09-28)

The 128-update classification warmup (`interleave_cls_warmup64`) passed 2/2 on the collapse seed 20260928. This round checks whether it holds on the other two seeds without harming the seeds where plain interleaving already worked.

| Run | Seed | Reps | Matched plain-interleave reference (NO-CPT joint) |
|---|---:|---:|---|
| interleave_cls_warmup64 | 20260926 | 2 | `laya_jev_multitask_v1/round/no_cpt_joint`: GFP 0.6993/0.4921 (pass), promoter 88.12%, structure 55.48% |
| interleave_cls_warmup64 | 20260927 | 2 | `laya_jev_multitask_seed_confirm/run/round/seed_20260927_no_cpt_joint`: GFP 0.6910/0.5279 (MAE fail) |

Everything else as in the classification-warmup round (NO-CPT, same data, frozen per-seed interleave with only epoch-1 classification batches moved forward, 1,152 updates, same LR, loss and evaluation, train diagnostics every 128 updates). Output root `run_cls_warmup_xseed`. No test access.

Success rule (per seed, both reps): GFP dev RMSE < 0.8357, MAE < 0.5095, SD >= 0.05; promoter >= 85%; structure >= 54%.

- Both seeds pass: the recipe is confirmed on 3/3 NO-CPT seeds (with 20260928); next is the CPT arm and writing it up as the recommended joint recipe.
- One seed fails: record the seed-dependent pass rate; compare against the plain-interleave reference on that seed before any further change.
- Classification materially below the seed's plain-interleave reference (more than 2 points on promoter or structure in both reps) is recorded as a cost even if the thresholds pass.
