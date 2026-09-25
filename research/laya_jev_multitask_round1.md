# One Laya model, three tasks, JEV-style output: decision round

**Status: running; no completed-round result yet.** The user-defined objective is one model with a shared JEV-style interface and useful task quality/inference efficiency. Few-shot is not a current objective. This is a bounded engineering decision on whether to continue the Laya backbone before considering Qwen biological adaptation and, later, OmniGene4.

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
