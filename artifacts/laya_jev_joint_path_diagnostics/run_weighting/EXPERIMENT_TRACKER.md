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

Current status: follow-up matrix complete. `task_block_gfp_last` restored GFP across seeds 20260928 and 20260927, while `task_block_gfp_first` only partially recovered seed 20260928; this supports a GFP-last recency/path effect, not arbitrary task-block sufficiency.
