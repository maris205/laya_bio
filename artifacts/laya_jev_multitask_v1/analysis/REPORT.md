# One Laya checkpoint, three tasks, JEV-style output

Fixed final-epoch results. Each task has 8,192 training examples; three epochs; one seed; reused development sets; no new test inference. All models use a shared typed candidate scorer with no task-specific output head.

| Task | Metric | Constant | CPT single | No-CPT joint | CPT joint |
|---|---|---:|---:|---:|---:|
| promoter | accuracy | 0.4952 | 0.8850 | 0.8812 | 0.8888 |
| promoter | macro_f1 | 0.3312 | 0.8850 | 0.8812 | 0.8888 |
| structural_class | accuracy | 0.3014 | 0.5847 | 0.5548 | 0.5687 |
| structural_class | macro_f1 | 0.0662 | 0.4507 | 0.4959 | 0.4390 |
| fluorescence | rmse | 0.8357 | 0.7066 | 0.6993 | 0.7512 |
| fluorescence | mae | 0.5095 | 0.5411 | 0.4921 | 0.6229 |
| fluorescence | spearman | — | 0.4341 | 0.4589 | 0.3682 |

## Predeclared decision checks

These flags use the practical thresholds frozen before production; they are not tests of statistical significance.

```json
{
  "promoter": {
    "cpt_joint_beats_constants": true,
    "no_cpt_joint_beats_constants": true,
    "cpt_joint_accuracy_delta_to_single": 0.0038022813688213253,
    "cpt_joint_macro_f1_delta_to_single": 0.003793714239616164,
    "material_joint_retention_flag": false
  },
  "structural_class": {
    "cpt_joint_beats_constants": true,
    "no_cpt_joint_beats_constants": true,
    "cpt_joint_accuracy_delta_to_single": -0.015974440894568787,
    "cpt_joint_macro_f1_delta_to_single": -0.011668569274779939,
    "material_joint_retention_flag": false
  },
  "fluorescence": {
    "cpt_joint_beats_constants": false,
    "no_cpt_joint_beats_constants": true,
    "cpt_joint_RMSE_ratio_to_single": 1.0631640536288156,
    "material_joint_retention_flag": false
  }
}
```

## Paired changes

- promoter, cpt_minus_no_cpt_joint: +0.7605 pp; descriptive 95% interval [-1.0456, +2.4715].
- promoter, cpt_joint_minus_cpt_single: +0.3802 pp; descriptive 95% interval [-0.9506, +1.7110].
- structural_class, cpt_minus_no_cpt_joint: +1.3845 pp; descriptive 95% interval [-1.1715, +4.0469].
- structural_class, cpt_joint_minus_cpt_single: -1.5974 pp; descriptive 95% interval [-4.1534, +0.9585].
- fluorescence, cpt_minus_no_cpt_joint: +0.0519 native RMSE; descriptive 95% interval [+0.0419, +0.0619].
- fluorescence, cpt_joint_minus_cpt_single: +0.0446 native RMSE; descriptive 95% interval [+0.0343, +0.0553].

## Engineering decision

NO-CPT joint is the recommended checkpoint from this bounded round: it is the only shared model that passes all three frozen usefulness gates. CPT joint slightly improves classification Accuracy but worsens GFP RMSE, MAE and Spearman, and does not pass the GFP MAE constant-baseline gate. This is a one-seed development decision, not a backbone-superiority claim.

Independent audit: **PASS**. It reconstructed 31,929 encoded rows, recomputed 54 metric points, checked 30 prediction files, and verified both retained model hashes, frozen schedules/exposures, shared initialization and absence of task-specific heads.

## Portable mixed-task inference and raw-input cost

| Arm | Task | p50 (ms) | p95 (ms) | Batch-16 requests/s |
|---|---|---:|---:|---:|
| no_cpt_joint | promoter | 32.68 | 40.59 | 348.60 |
| no_cpt_joint | structural_class | 32.91 | 41.06 | 235.45 |
| no_cpt_joint | fluorescence | 24.04 | 30.70 | 277.11 |
| cpt_joint | promoter | 23.55 | 25.16 | 421.01 |
| cpt_joint | structural_class | 23.94 | 28.08 | 246.69 |
| cpt_joint | fluorescence | 23.68 | 26.27 | 276.73 |

Both exported joint checkpoints reproduce the saved homogeneous-batch probabilities exactly for the same raw inputs. In the 48-request interleaved Noul/Choice/Score check, both preserve every argmax; maximum probability differences versus the original homogeneous BF16 evaluation are 0.0181 (no-CPT) and 0.0100 (CPT). The difference reflects changed batch shapes and padding.

## Interpretation boundaries

- Classification output validity is enforced by the candidate interface and is separate from biological accuracy.
- Score outputs are an ordered distribution with a native-unit expected value. Interpolated training targets and a continuous loss differ from the earlier hard-bin pilot.
- Single-task anchors match target examples, presentations, permutations and LR positions; joint training adds other-task updates and changes optimizer/dropout history.
- Training diagnostics use fixed 1,024-entity subsets per task. Development uses all 1,052 / 939 / 5,362 admitted examples.
- Reverse Choice order is a fixed diagnostic, not exhaustive instruction or order invariance.
- Native GFP train/valid variants share a parent. The whole saved CPT protein sample has no 15mer overlap with GFP train/valid; this is not a global homology guarantee.
- Raw-sequence timings include tokenization, textual rubric/panel assembly, device transfer, model execution, softmax and typed-answer construction on a warm loaded model. They exclude file/network I/O and do not establish a speed advantage over Qwen or OmniGene4.
- A CPT failure is not automatically a Laya failure: inspect the matched no-CPT joint model. Negative results apply to this finite budget and recipe.

Local raw evidence: `/root/autodl-tmp/jev_gene/artifacts/laya_jev_multitask_v1`.
