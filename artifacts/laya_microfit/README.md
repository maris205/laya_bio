# Training-only 32-example memorization checks

Each task uses 32 deterministically selected training examples, stratified by label. No development or test rows are read. These are fitting diagnostics, not benchmark or generalization results.

The initial four runs use 128 full-batch updates, LR 2e-5 with an eight-update warmup and no subsequent decay. A predeclared conditional candidate-only control uses LR 1e-4 after a failed final training-panel fit criterion. A separate, post-hoc splice experiment changes only candidate-order resampling after the fixed-order run revealed a gap between its training and canonical panels. A second post-hoc control applies the same LR 1e-4 to the GFP shared-head baseline after the candidate control succeeds. All seven runs are complete; see `comparison.json` and `all_order_evaluation.json`.

Configurations preserve selected IDs, subset hashes, label counts and training definitions. Summaries include numerical/reload checks and input-rotation diagnostics. Model weights and per-example predictions remain local. The code snapshots preserve the exact versions used by each queue; the main scripts subsequently gain an optional order-augmentation switch.

See the [full report](../../research/laya_microfit.md). The five-bin Score objective and expected-anchor decoding differ from continuous scalar regression; report both classification and continuous-value errors without treating a hard-bin anchor reference as a universal error bound.
