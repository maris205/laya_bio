# Frozen final-checkpoint test protocol

Authorized after completion of the three-seed raw/full-BPE experiments and B1/text-only controls. This document and the evaluator are hashed before the first test prediction.

- Evaluate all three seeds (20260922, 20260923, 20260924) of raw candidate input, full biological BPE candidate input, full-BPE B1 fixed-class output, and trained text-only control: 12 existing final checkpoints.
- Primary model: full-BPE candidate model. Primary comparisons: implemented B1 and trained text-only. Raw candidate input is the representation reference. Do not select the best seed or checkpoint.
- Use the previously frozen common complete-input eligibility list. Expected test sample counts: promoter detection 2,145; protein fold 1,952. Report task metrics separately.
- Preserve canonical candidate/class order, checkpoint-specific saved representation, max context 1,024, BF16 inference and batch size 16. No truncation is allowed.
- Each model must reproduce its complete saved selection-dev logits within 1e-5 before test inference. The model state is loaded strictly from disk.
- Report accuracy and macro-F1 as main classification metrics; also balanced accuracy, MCC, NLL, Brier and 15-bin ECE. Report both raw and calibrated probability metrics using only temperatures already fitted on the calibration split. No test temperature fitting, updates, early stopping, checkpoint choice or hyperparameter choice.
- Report all seed rows, mean and sample standard deviation across three seeds, paired full-BPE-minus-reference differences and relative changes. Standard deviations measure seed variability, not uncertainty due to test sampling; no significance claims or IID confidence intervals.
- Lock SHA-256 hashes of input files, eligibility list, historical run manifests, scripts, vendor model code, source summaries, saved dev predictions, all checkpoint weights/configs/tokenizers/representations and this document before evaluation. Recheck at completion; stop on mismatch or invalid predictions. Refuse duplicate outputs.
- Save per-example IDs, labels, logits and probabilities; independently reload the saved predictions and recompute report metrics. Test evaluation ends after these 12 conditions. No additional perturbation, ablation or training is selected from the test results.

Interpretation remains limited to this eligible benchmark subset and the implemented controls. Historical BPE corpus overlap has not been fully established; homology-independent generalization is not claimed. Existing dev sequence-shuffle and candidate-permutation diagnostics retain their separate status.

Outputs: `artifacts/laya_locked_test/manifest.json`, per-model predictions/summary, `results.json`, and `research/laya_locked_test_results.md`.
