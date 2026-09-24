# Laya fixed-checkpoint test results

All 12 locked checkpoints evaluated once on the common eligible test subset. Values are mean ± sample standard deviation across three seeds, not confidence intervals. No model selection, training, or temperature refitting used test results.

DNA n=2,145; protein fold n=1,952. All inputs are complete (no truncation).

| Model | DNA accuracy (%) | DNA macro-F1 (%) | Fold accuracy (%) | Fold macro-F1 (%) |
|---|---:|---:|---:|---:|
| raw | 91.10 ± 0.35 | 91.09 ± 0.35 | 59.97 ± 0.56 | 50.63 ± 1.67 |
| full_bpe | 90.16 ± 0.14 | 90.16 ± 0.14 | 59.05 ± 0.33 | 50.28 ± 0.42 |
| b1 | 88.76 ± 0.37 | 88.76 ± 0.37 | 51.14 ± 5.36 | 38.00 ± 4.15 |
| text_only | 49.84 ± 0.57 | 33.26 ± 0.25 | 29.10 ± 0.00 | 6.44 ± 0.00 |

## Calibration (temperatures fitted previously on calibration split)

| Model | DNA raw NLL | DNA calibrated NLL | Fold raw NLL | Fold calibrated NLL |
|---|---:|---:|---:|---:|
| raw | 0.24 ± 0.01 | 0.22 ± 0.01 | 1.19 ± 0.03 | 1.04 ± 0.02 |
| full_bpe | 0.31 ± 0.01 | 0.25 ± 0.00 | 1.18 ± 0.02 | 1.06 ± 0.01 |
| b1 | 0.30 ± 0.01 | 0.28 ± 0.01 | 1.24 ± 0.06 | 1.24 ± 0.06 |
| text_only | 0.69 ± 0.00 | 0.69 ± 0.00 | 1.63 ± 0.00 | 1.63 ± 0.00 |

## All seeds

| Model | Seed | DNA accuracy (%) | Fold accuracy (%) | Fold macro-F1 (%) |
|---|---:|---:|---:|---:|
| raw | 20260922 | 91.14 | 60.55 | 52.56 |
| full_bpe | 20260922 | 90.30 | 59.38 | 50.70 |
| b1 | 20260922 | 88.48 | 45.65 | 33.44 |
| text_only | 20260922 | 49.51 | 29.10 | 6.44 |
| raw | 20260923 | 91.42 | 59.94 | 49.72 |
| full_bpe | 20260923 | 90.16 | 58.71 | 49.86 |
| b1 | 20260923 | 89.18 | 56.35 | 41.54 |
| text_only | 20260923 | 50.49 | 29.10 | 6.44 |
| raw | 20260924 | 90.72 | 59.43 | 49.61 |
| full_bpe | 20260924 | 90.02 | 59.07 | 50.28 |
| b1 | 20260924 | 88.62 | 51.43 | 39.02 |
| text_only | 20260924 | 49.51 | 29.10 | 6.44 |

## Paired differences: full BPE minus reference

| Reference | Task | Accuracy delta (pp) | Relative accuracy change (%) | Macro-F1 delta (pp) |
|---|---|---:|---:|---:|
| raw | promoter_detection | -0.93 ± 0.29 | -1.02 | -0.93 ± 0.29 |
| raw | fold_class | -0.92 ± 0.49 | -1.54 | -0.35 ± 1.33 |
| b1 | promoter_detection | +1.40 ± 0.42 | +1.58 | +1.40 ± 0.42 |
| b1 | fold_class | +7.91 ± 5.69 | +15.46 | +12.28 ± 4.56 |
| text_only | promoter_detection | +40.33 ± 0.58 | +80.92 | +56.90 ± 0.29 |
| text_only | fold_class | +29.95 ± 0.33 | +102.93 | +43.84 ± 0.42 |

## Scope

The fixed-class B1 comparison concerns the implemented CLS readout with two task-specific output matrices. This is not evidence against every possible fixed-class architecture. The BPE vocabulary was trained on historical external corpora whose overlap with this benchmark is not fully established. The eligible subset excludes examples that failed earlier shared complete-input checks. No claim of homology-independent generalization or statistical significance is made. Previously measured candidate-order sensitivity remains a dev diagnostic; no additional test perturbations were selected.

Machine-readable source: `artifacts/laya_locked_test/results.json`; per-sample logits and labels: `artifacts/laya_locked_test/*/predictions.jsonl`.
