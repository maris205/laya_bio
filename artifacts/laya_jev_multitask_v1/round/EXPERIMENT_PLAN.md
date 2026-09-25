# Laya decision round: one model, multiple tasks, JEV-style outputs

**Date:** 2026-09-25. **Status:** preparation and train-only preflight; freeze before production development evaluation.

**User objective:** a single model checkpoint jointly handles biological tasks with a unified JEV-style output contract, with useful task quality and practical inference cost. The backbone is replaceable (Laya first, Qwen adaptation as the next candidate, OmniGene4 later). Few-shot/zero-shot and a universal CPT advantage are not current requirements.

The completed fixed-classifier CPT comparison is archived in [its completed plan](EXPERIMENT_PLAN_biocpt_v2_complete.md) and [report](../research/laya_biocpt_round1.md). The old no-CPT microfit/GFP extension remains paused. This round is a new, bounded experiment with biological IDs, the corrected CPT checkpoint, substantial supervision, a shared typed candidate scorer, and a continuous Score objective. It does not resume the old 2,048-step diagnostic queue.

## Claim and decision map

| Question | Minimum useful evidence | Runs |
|---|---|---|
| Can a single Laya checkpoint learn Noul, Choice and Score with useful quality? | Per-task development results vs train-fitted constant baselines; valid typed outputs; no separately routed task head | CPT joint, no-CPT joint |
| Does joint training materially degrade otherwise learnable tasks? | Same-interface single-task anchors with identical per-task examples, presentations, candidate permutations and joint-position LR schedule | Three CPT single-task anchors |

These are engineering decisions on reused development sets, one seed. They are not paper-ready generalization or architecture-superiority claims. Fixed-head promoter results around 90% used a different interface and 16,766 labels; they are context, not a matched score threshold for this 8,192-label round.

## Data and admission

| Task | Primitive | Train rows | Dev rows | Primary metric |
|---|---|---:|---:|---|
| Promoter detection | Noul (false / true) | 8,192 | 1,052 | Accuracy and macro-F1 |
| Protein structural class | Choice (7 classes) | 8,192 | 939 | Accuracy and macro-F1; all-class support/recall |
| GFP log fluorescence | Score (5 native-unit anchors) | 8,192 | 5,362 | Native-unit RMSE; MAE, Spearman secondary |

Classification sampling preserves the original training class proportions; the GFP subset uses a deterministic ID hash. No development-based subset selection. Entire existing development partitions are retained. These are thousands of labels per task, not the entire available training pool (16,766 / 15,593 / 21,446); this first decision round does not test data saturation.

The original corrected DNA/protein BPE and added-ID map are reused without refitting. Biological sequences are encoded directly as the new modality-specific IDs, while instructions/options use original text IDs. All full panels must fit 512 tokens; no sequence, option or question is truncated. Observed train maximums are 116 / 309 / 207 tokens; no rows excluded for length.

Cross-task exact sequence/RC train–dev intersections are zero. Formal calibration/test sequence memberships are reserved using inputs only, without targets. Native GFP train/valid are used; its test file is not opened. All 59,002 unique protein 15-mers from the entire GFP train+valid pool were checked against all saved CPT protein train/validation snapshots (including samples not actually presented): zero overlapping rows. This permits this bounded extension from the existing CPT checkpoint; it does not prove homology independence. GFP variants share a parent protein, and source memberships have historical use.

## Unified output and model

One ModernBERT encoder feeds the original Laya two-layer typed transformer, type embedding, and one shared scalar marker scorer. There are no task-specific classifiers or regressors. All options for one entity are scored in one encoder call. Original shared-head weights are restored identically for every run; the only no-CPT/joint versus CPT/joint encoder difference is the recorded CPT adaptation. Unsupervised MLM head and unused action head are not used for SFT. No inherited temperature calibration is applied to this changed domain/model.

- **Noul:** false/true probabilities and boolean answer.
- **Choice:** probabilities over supplied labels, selected label/index. Candidate order is deterministically permuted per entity and epoch during training, with the target remapped. Canonical evaluation is primary; reverse-order evaluation is a fixed robustness diagnostic with probabilities mapped back to semantic labels.
- **Score:** distribution over five ordered native-unit anchor values, argmax level and expected native value. Anchors evenly span the chosen 8,192 training values' min/max. Targets interpolate between adjacent anchors, so their expected values preserve continuous training targets exactly. Loss is 0.5 soft-target CE + 0.5 MSE of expected value standardized by training-only SD. This replaces the old pilot's hard quantile-bin CE; it is disclosed as a protocol change, shared by every current run. Prediction range is bounded by train anchors; report out-of-range dev fraction. Auxiliary nearest-anchor accuracy never substitutes for continuous RMSE.

No instruction-generalization, free-text generation or universal task-semantic claim follows from valid constrained output.

## Fixed run matrix and training budget

| Run | Backbone | Tasks | Updates | Per-task presentations |
|---|---|---|---:|---:|
| CPT-JOINT | Corrected CPT | All 3 | 1,152 | 24,576 each |
| NO-CPT-JOINT | Same expanded original encoder | All 3 | 1,152 | 24,576 each |
| CPT-PROMOTER | Corrected CPT | Promoter only | 384 | 24,576 |
| CPT-STRUCTURE | Corrected CPT | Structural class only | 384 | 24,576 |
| CPT-SCORE | Corrected CPT | GFP only | 384 | 24,576 |

Total 3,456 production updates, 221,184 entity presentations. Three epochs, effective batch 64, micro-batch 8 (subject only to train-only feasibility smoke). Homogeneous task batches; each three-update cycle contains one batch from each task in seeded random order. All examples appear exactly once per epoch. Anchors skip other tasks in this same joint schedule, preserve target batch IDs and permutations, and use the learning rate at that joint schedule position. They therefore match target-task exposure and LR but not total updates or the intervening optimizer/dropout history. Differences are operational joint-training effects, not a pure isolation of gradient interference.

Seed 20260926; FP32 parameters/BF16 autocast; gradient checkpointing for encoder and typed layers; AdamW; encoder LR 2e-5, typed head/type embeddings/scorer LR 1e-4; 5% warmup and cosine to 10%; weight decay .01 except embeddings/norms/biases; clip norm 1. All parameters trainable. No best-dev selection, hyperparameter sweep or result-dependent budget extension. Last epoch is primary; all initial and per-epoch dev points remain visible.

## Verification and outputs

Before launch: CPU contract checks cover continuous targets/gradient normalization, semantic remapping under option permutation, padding invariance and lack of task-specific heads; GPU smoke trains one batch per task using train only, checks finite gradients and exact checkpoint reload. Original model/data/CPT hashes and all code are frozen before production.

For each run: retain losses/gradients/LR, per-epoch complete dev predictions, initial/final fixed 1,024-per-task training diagnostic predictions, actual exposure counts, shared-head/batch/panel hashes, and checkpoint reload evidence. Training metrics use the fixed training diagnostic subset, not the full training population. Joint runs additionally benchmark 64 predetermined dev inputs/task, with cached tokenization, full panel assembly/packing/transfer/model/softmax/return included; report batch-1 p50/p95 and batched throughput. These timings alone do not establish a speed advantage over untested Qwen/OmniGene4.

Independent final audit: recompute every metric from predictions, verify source/code/model hashes, task coverage, exposure counts, initial shared-head equality, paired batch/panel order, output validity and Score mapping. Bootstrap intervals on existing dev are descriptive and do not include seed variation. Preserve CPT-joint and no-CPT-joint final weights; save primary CPT-joint optimizer/RNG state. Single-task final weights may be removed only after exact reload to bound disk use; predictions and hashes remain. Historical weights/source data are untouched.

## Decision gates, set before production results

- **Engineering failure:** invalid outputs, nonfinite optimization, missing task gradients, input truncation/leak admission failure, or failed save/reload. Stop and fix the implementation; do not count this as backbone failure.
- **Classification usefulness:** beat the training-majority constant's development accuracy and macro-F1. For a joint-retention flag, a drop exceeding 3 accuracy percentage points OR .03 macro-F1 relative to the matched single-task anchor is material under this bounded protocol. Small differences require seed confirmation, not a categorical conclusion.
- **Score usefulness:** beat training-mean RMSE and training-median MAE; report Spearman and prediction spread. A joint RMSE increase over 10% relative to its single-task anchor is a material-retention flag. These are practical decision thresholds, not statistical significance tests.
- **Continue Laya:** all three tasks show useful learning and no material joint-retention failure, then confirm seeds/scale and compare inference cost with a real alternative.
- **Consider Qwen next:** material failures remain after this correctly implemented fixed round, especially if single-task controls also fail. Preserve negative outcomes and propose a matched Qwen vocabulary/CPT/JEV-SFT experiment; do not spend more rounds trying to force Laya success. A negative finite-budget run does not prove architectural impossibility.

## Execution and cost

Data admission is complete. CPU tests and a three-update train-only GPU smoke precede freezing the production controller. GPU is the existing local RTX 4080 SUPER reporting 32 GiB. Free disk at admission is approximately 17 GiB. Final runtime forecast is recorded from measured smoke before development outcomes. No new CPT is planned. No few-shot/zero-shot experiments or expansion to the full task suite in this round.

### Preflight completed; production settings frozen

CPU contract checks passed. Train-only GPU smoke completed one effective 64-entity update per task, with positive encoder/scorer gradients and exact reload. Peak allocated GPU memory 8,170,972,160 bytes (7.61 GiB). Observed seconds/update: promoter 1.5003, structure 2.3762, GFP 1.5183. Micro-batch 8 and evaluation batch 16 are retained. The approximate production forecast is 115–150 minutes including evaluation/checkpoint overhead; short-smoke timing is uncertain. Frozen run order: CPT joint → no-CPT joint → CPT promoter → CPT structure → CPT GFP. No development outcomes were inspected to choose this recipe.
