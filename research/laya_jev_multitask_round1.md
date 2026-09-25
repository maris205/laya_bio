# One Laya model, three tasks, JEV-style output: decision round

**Status: complete; the full five-run matrix, independent audit, portable inference checks, plots and compact archive passed.** The user-defined objective is one model with a shared JEV-style interface and useful task quality/inference efficiency. Few-shot is not a current objective. This is a bounded engineering decision on whether to continue the Laya backbone before considering Qwen biological adaptation and, later, OmniGene4.

## Frozen setting

Each task has 8,192 training examples. Full development sets contain 1,052 promoter, 939 protein-structure and 5,362 GFP examples. All inputs are untruncated; maximum model lengths are 116/309/207 tokens. The three outputs are Noul (boolean distribution), Choice (seven supplied structural labels), and Score (ordered distribution plus expected native log-fluorescence value). One 423,122,945-parameter model shares every learned module across tasks; task names choose text templates only, never model-specific parameters.

The corrected biological CPT checkpoint and expanded BIO IDs are reused. The full GFP train/valid 15mer bank has no overlap with the saved CPT protein train/validation snapshots. Formal calibration/test memberships are reserved using sequences only. GFP test is not opened; native GFP variants share a parent, so this is not a family-independent test.

The fixed matrix is CPT joint and no-CPT joint (1,152 updates each), plus three same-interface CPT single-task references (384 updates each). Three epochs, effective batch 64; identical target-task exposures, permutations and joint-position learning-rate schedule. Joint runs add other-task updates and optimizer/dropout history. All final-step outcomes will be retained, including negative ones.

The continuous Score target interpolates between five train-fitted anchors and uses soft-target CE plus standardized MSE. This differs from the older hard-bin pilot; none of its historical numbers is a matched control. RMSE/MAE are primary Score measures, not bin accuracy. The previous fixed-head promoter round also uses a different interface/data budget and cannot serve as a matched single-task reference.

## Execution and checks

CPU tests passed for continuous targets, gradient normalization, option remapping and padding. The train-only GPU smoke exercised all three primitives with nonzero encoder/scorer gradients and exact checkpoint reload. Peak GPU allocation was 7.61 GiB. The production recipe was frozen before development evaluation, with a 115–150 minute forecast from smoke timing.

The shared-model inference entry point is [laya_jev_infer.py](../scripts/laya_jev_infer.py). After completion, both joint checkpoints will receive portable configuration/tokenizers and a raw-sequence mixed-task inference check. Raw-to-answer latency will be measured separately from cached-tokenization model timing. No speed advantage over unmeasured alternatives is claimed.

- [Frozen plan](../artifacts/laya_jev_multitask_v1/round/EXPERIMENT_PLAN.md)
- [Launch manifest](../artifacts/laya_jev_multitask_v1/round/manifest.json)
- [Admission and evidence archive](../artifacts/laya_jev_multitask_v1/README.md)
- [Active plan and decision gates](../refine-logs/EXPERIMENT_PLAN.md)

Local full experiment root: `/root/autodl-tmp/jev_gene/artifacts/laya_jev_multitask_v1`. Previous results and checkpoints remain intact. The old GFP microfit extension stays paused.

## CPT joint result

The fixed third-epoch CPT joint model reaches promoter Accuracy **88.88%**, macro-F1 **0.8888**; protein-structure Accuracy **56.87%**, macro-F1 **0.4390**; GFP native RMSE **0.7512**, MAE **0.6229**, Spearman **0.3682**. GFP improves RMSE over the train-mean constant (0.8357), but its MAE remains worse than the train-median constant (0.5095). It has not passed the full predeclared usefulness gate. Two rare structural classes retain zero recall.

The fixed 1,024-per-task training diagnostic has promoter Accuracy 96.00%, structural Accuracy 68.36%, and GFP RMSE 0.7509. This is a diagnostic subset, not full-training evaluation. The canonical/reversed structural panels score 56.87%/56.76%; semantic argmax agrees on 89.35% of examples. This single reversal is not exhaustive invariance evidence.

All 24,576 selected training entities were presented exactly three times, and checkpoint reload predictions match exactly. The first joint fit took 34.37 minutes. Cached-tokenization p50 single-request latency was about 24 ms/task; the completed raw-to-answer and mixed-task checks are reported below.

## Completed matched matrix

| Task | Metric | No-CPT joint | CPT joint |
|---|---|---:|---:|
| Promoter | Accuracy | 88.12% | 88.88% |
| Promoter | Macro-F1 | 0.8812 | 0.8888 |
| Protein structure | Accuracy | 55.48% | 56.87% |
| Protein structure | Macro-F1 | 0.4959 | 0.4390 |
| GFP | RMSE ↓ | 0.6993 | 0.7512 |
| GFP | MAE ↓ | 0.4921 | 0.6229 |
| GFP | Spearman ↑ | 0.4589 | 0.3682 |

The completed CPT single-task references score 88.50%/0.8850 promoter Accuracy/Macro-F1, 58.47%/0.4507 structure Accuracy/Macro-F1, and 0.7066/0.5411/0.4341 GFP RMSE/MAE/Spearman. CPT joint relative to these references changes promoter Accuracy by +0.38 percentage points, structure Accuracy by −1.60 points and GFP RMSE by +6.32%; none crosses the predeclared material-retention threshold. The controls match target examples, presentations, permutations and LR positions, but joint runs include other-task updates and different optimizer/dropout history.

The no-CPT joint model beats the frozen train-fitted constant baselines on all three tasks under the predeclared usefulness rule. It is therefore the final engineering choice for the shared checkpoint in this bounded round. CPT's mixed effects must not be interpreted as a blanket Laya backbone failure. This is one seed; no architecture-superiority conclusion is made. Both arms expand the biological vocabulary and update all encoder/embedding parameters during supervised training. “No CPT” means no additional unsupervised neural adaptation, not an unchanged original model.

Reversing the structural Choice panel preserves the selected semantic label on 89.35% of CPT and 90.73% of no-CPT requests. Similar aggregate accuracy therefore does not imply per-example candidate-order invariance. This limitation remains explicit. No new test inference, family-independent GFP claim, unseen-task claim or cross-backbone speed claim is made.

## Independent verification and portable inference

The final audit independently reconstructed 31,929 encoded rows, recomputed 54 metric points from 30 saved prediction files, reproduced frozen schedules/panel hashes and exposure histograms, rechecked all 33,280 CPT protein rows for the GFP 15-mer admission, verified both retained model hashes and confirmed the retained state dictionaries contain no task-specific output heads. CPT-joint optimizer/RNG recovery remains available at step 1,152 with 201 parameter states. The audit status is PASS.

Both retained joint checkpoints now include self-contained tokenizer/representation assets, encoder configuration and task templates without duplicating model weights. Same-shaped raw-input inference reproduces saved probabilities exactly. Each 48-request interleaved Noul/Choice/Score check preserves all argmax decisions; changed mixed-batch shapes produce maximum BF16 probability differences of 0.0100 (CPT) and 0.0181 (no-CPT).

Warm-model raw-sequence-to-typed-answer p50 latency is 23.55/23.94/23.68 ms for CPT promoter/structure/GFP and 32.68/32.91/24.04 ms for no-CPT. Batch-16 throughput is 421.01/246.69/276.73 and 348.60/235.45/277.11 requests/s, respectively. These short single-run measurements include tokenization, rubric/panel construction, transfers, model, softmax and typed answer construction, but exclude file/network I/O and do not compare an alternative backbone.

The compact archive contains 87 files: summary, audit, PNG/PDF figure, frozen recipe/code, traces, status/configuration and losslessly compressed predictions. It excludes raw sequences, model states and test predictions. The local retained joint weights remain under `/root/autodl-tmp/jev_gene/artifacts/laya_jev_multitask_v1`.
