# JEV joint-path diagnostic tracker

Updated: 2026-09-26 UTC.

| Step | Status | Notes |
|---|---|---|
| Freeze diagnostic plan | Complete | First run is `task_block_gfp_last` for seed 20260928 NO-CPT. |
| GPU and disk preflight | Complete | RTX 4080 SUPER idle; about 89 GiB free under `/root/autodl-tmp`. |
| Smoke: task_block_gfp_last | Complete | Completed promoter, structural_class, fluorescence smoke updates with finite non-zero gradients. |
| Formal: task_block_gfp_last | Complete | 1,152 updates; model retained; checkpoint reload exact; no test inference. |
| Summary audit | Complete | Wrote `run/analysis/summary.json`; audit status pass. |
| Follow-up smoke: task_block_gfp_first seed 20260928 | Complete | Smoke passed; formal run completed. |
| Follow-up formal: task_block_gfp_first seed 20260928 | Complete | 1,152 updates; GFP dev RMSE/MAE/Spearman/SD = 0.8296/0.4935/0.3785/0.1552; gate passes but weak. |
| Follow-up smoke: task_block_gfp_last seed 20260927 | Complete | Smoke passed; formal run completed. |
| Follow-up formal: task_block_gfp_last seed 20260927 | Complete | 1,152 updates; GFP dev RMSE/MAE/Spearman/SD = 0.5791/0.3142/0.6059/0.5485; gate passes. |
| Follow-up summary audit | Complete | Combined `analysis/summary.json` written; audit status pass, 3 variants. |
| Weighting run v2 (fluo x0.5) | Interrupted | Killed externally at step 992/1152 (no traceback; session ended). Kept as incomplete record, excluded from summary. |
| Weighting smoke x0.5 / x2.0 (v3) | Complete | Both smoke runs passed in `run_weighting_v3`. |
| Weighting formal: task_block_gfp_last fluo x0.5, seed 20260928 | Complete | 1,152 updates; reload exact; GFP dev RMSE/MAE/Spearman/SD = 0.6610/0.4256/0.5117/0.4344; gate passes. |
| Weighting formal: task_block_gfp_last fluo x2.0, seed 20260928 | Complete | 1,152 updates; reload exact; GFP dev RMSE/MAE/Spearman/SD = 0.6298/0.4017/0.5292/0.4789; gate passes. |
| Weighting summary audit | Complete | `analysis/summary.json` rewritten; audit pass, 5 variants, 0 pending. |

| Replicate round (run_replicate) | Complete | gfp_last rep1/rep2 RMSE/MAE 0.5317/0.3150, 0.5890/0.4393; gfp_first rep1/rep2 0.8617/0.5121, 0.8291/0.6441 (SD 0.0216). Rule output: reproducible_gfp_last_effect (gate pass 3/3 vs 1/3). |

Current status: replicate round complete; GFP-last effect is reproducible at n = 3 on seed 20260928. Weighting matrix complete. Under GFP-last, fluorescence loss weight x0.5, x1.0 and x2.0 all restore GFP and pass both gates; GFP is non-monotone in the weight (x1.0 best), so within 0.5-2x the loss weight is not the controlling factor. Earlier status: follow-up matrix complete. `task_block_gfp_last` restored GFP across seeds 20260928 and 20260927, while `task_block_gfp_first` only partially recovered seed 20260928; this supports a GFP-last recency/path effect, not arbitrary task-block sufficiency.
