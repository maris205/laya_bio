# Splice / GFP development diagnostics

Matched single-task controls use 96 optimizer steps selected from the original 576-step joint schedule. All per-task batches, candidate permutations, and learning rates are preserved; this matches task exposure, not total compute or optimizer history. Each task uses the existing 1,024 train / 128 dev entities, seed 20260924. Checkpoints and per-example predictions remain local.

`input_and_cpu_controls.json` contains source-row checks, actual input token lengths, prediction distributions, and fixed position-based CPU controls. `corrected_data_manifest.json` versions a splice candidate-name correction without changing numeric targets or split membership. The interpretation and incomplete upstream-lineage caveat are in the [report](../../research/laya_task_diagnostics.md).

`frozen_code/` is the code used by the four single-task runs and the text-correction arm. Its old builder still contains historical splice names, because the historical data were intentionally held fixed for those comparisons. The corrected arm reads the separately versioned data manifest; current `scripts/laya_multitask_data.py` uses corrected names for new builds. The current training entry point only adds CLI help relative to the frozen trainer. These snapshots preserve recorded source hashes.

The first launch failed during import because a dependency was missing from its snapshot; no model loaded or optimizer update occurred. The successful queue has a separate v2 directory and includes that dependency.

All five training runs and the joint training-subset evaluation are complete. `neural_comparison.json` records verified checkpoint hashes, 480 finite loss/gradient records, exact reload agreement, and the invariant that only splice candidate text changed in the corrected dataset. The queue status files record successful completion at 2026-09-24 07:05 UTC. No held-out test predictions were generated.
