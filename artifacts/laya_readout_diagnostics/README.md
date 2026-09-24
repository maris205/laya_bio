# GFP readout and loss diagnostics

These runs fit the same 32 stratified GFP training examples as the earlier microfit study. They do not read development or test samples and are not generalization results. Score thresholds, anchors and normalization retain the earlier 1,024-training-example definitions.

The first queue is a fixed 2 x 3 comparison of CLS/attention-masked mean pooling and joint/CE-only/MSE-only supervision. The conditional queue tests removal of the typed transformer head and replacement of the pretrained readout by a fresh LayerNorm, with an additional native MSE-only arm if its joint-loss scalar fit fails. Status files record when each design was frozen, the conditions, commands and exact source hashes. These are post-hoc diagnostics following earlier development results, not preregistered benchmark experiments.

A dash in the report means that output had no corresponding supervision. Raw predictions still include both output branches for transparency; an unsupervised branch is not a valid performance comparison. Single-loss arms also change the corresponding loss coefficient from 0.5 to 1.

The frozen code directories contain the exact implementations used. Each run temporarily saved a complete FP32 checkpoint, checked reloaded probabilities and scalar predictions, and recorded the weight SHA-256 before removing that run's temporary weights to bound disk usage. No historical checkpoints were removed. This archive contains configs, metrics, traces, per-example predictions, feature/gradient probes and plotting outputs; it does not contain trained weights.

See the [full report](../../research/laya_readout_diagnostics.md). Use the frozen versions to reproduce a run and the current [summary script](../../scripts/summarize_laya_readout.py) to aggregate the original queue directories.
