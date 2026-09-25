# One Laya model, three tasks, JEV-style output: decision round

**Status: both joint models complete; three CPT single-task references are running.** The user-defined objective is one model with a shared JEV-style interface and useful task quality/inference efficiency. Few-shot is not a current objective. This is a bounded engineering decision on whether to continue the Laya backbone before considering Qwen biological adaptation and, later, OmniGene4.

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

## First completed joint model (controls pending)

The fixed third-epoch CPT joint model reaches promoter Accuracy **88.88%**, macro-F1 **0.8888**; protein-structure Accuracy **56.87%**, macro-F1 **0.4390**; GFP native RMSE **0.7512**, MAE **0.6229**, Spearman **0.3682**. GFP improves RMSE over the train-mean constant (0.8357), but its MAE remains worse than the train-median constant (0.5095). It has not passed the full predeclared usefulness gate. Two rare structural classes retain zero recall.

The fixed 1,024-per-task training diagnostic has promoter Accuracy 96.00%, structural Accuracy 68.36%, and GFP RMSE 0.7509. This is a diagnostic subset, not full-training evaluation. The canonical/reversed structural panels score 56.87%/56.76%; this single reversal is not exhaustive invariance evidence.

All 24,576 selected training entities were presented exactly three times, and checkpoint reload predictions match exactly. The first joint fit took 34.37 minutes. Cached-tokenization p50 single-request latency is about 24 ms/task; raw-to-answer timing and mixed-task portable export remain pending. No-CPT joint training is now running; the three CPT single-task anchors follow. No final backbone decision is made before those controls.

## Matched joint comparison (single-task references pending)

| Task | Metric | No-CPT joint | CPT joint |
|---|---|---:|---:|
| Promoter | Accuracy | 88.12% | 88.88% |
| Promoter | Macro-F1 | 0.8812 | 0.8888 |
| Protein structure | Accuracy | 55.48% | 56.87% |
| Protein structure | Macro-F1 | 0.4959 | 0.4390 |
| GFP | RMSE ↓ | 0.6993 | 0.7512 |
| GFP | MAE ↓ | 0.4921 | 0.6229 |
| GFP | Spearman ↑ | 0.4589 | 0.3682 |

The no-CPT joint model beats the frozen train-fitted constant baselines on all three tasks under the predeclared usefulness rule. Thus CPT's mixed effects must not be interpreted as a blanket Laya backbone failure. This is one seed, with single-task retention references still pending; no final superiority or architecture conclusion is made. Both arms expand the biological vocabulary and update all encoder/embedding parameters during supervised training. “No CPT” means no additional unsupervised neural adaptation, not an unchanged original model.

Reversing the structural Choice panel preserves the selected semantic label on 89.35% of CPT and 90.73% of no-CPT requests. Similar aggregate accuracy therefore does not imply per-example candidate-order invariance. This limitation remains explicit.
