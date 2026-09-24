# GFP 128-to-512 update convergence check

This run uses the same 32 fitted training examples, original initialization, pooling/readout, optimizer, learning rate, warmup, loss and seed as the earlier `native_readout` arm. Only the training update budget increases to 512. The added evaluations use the same training examples. There is no development or test evaluation.

The run starts from the original Laya weights and exactly replays the recorded first 128 optimization steps and all 32 reference predictions before continuing. `prefix_check.json` records this check. Timing may differ. Equality of recorded optimization values and predictions is not a comparison of every weight tensor at step 128, because the historical temporary checkpoint was not retained.

The reference snapshot and frozen source files have hashes in the run status. The initial model/tokenizer/vendor hashes and Python/package versions were verified against the previous provenance record before launch. The final checkpoint is retained locally after save/reload verification; large weights are not part of this Git archive.

The protocol reports final classification and scalar fitting criteria separately, alongside the first observed passing evaluation if any. It does not select a favorable intermediate checkpoint. The longer budget is not an equal-budget comparison with a 128-update candidate scorer.

See the [full report](../../research/laya_convergence.md) and [summary/plotting script](../../scripts/summarize_laya_convergence.py).
