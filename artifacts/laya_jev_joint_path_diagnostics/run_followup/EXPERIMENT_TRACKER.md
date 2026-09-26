# JEV joint-path diagnostic tracker

Updated: 2026-09-26 UTC.

| Step | Status | Notes |
|---|---|---|
| Freeze diagnostic plan | Complete | First run is `task_block_gfp_last` for seed 20260928 NO-CPT. |
| GPU and disk preflight | Complete | RTX 4080 SUPER idle; about 89 GiB free under `/root/autodl-tmp`. |
| Smoke: task_block_gfp_last | Complete | Completed promoter, structural_class, fluorescence smoke updates with finite non-zero gradients. |
| Formal: task_block_gfp_last | Complete | 1,152 updates; model retained; checkpoint reload exact; no test inference. |
| Summary audit | Complete | Wrote `run/analysis/summary.json`; audit status pass. |
| Follow-up smoke: task_block_gfp_first seed 20260928 | Pending | Distinguish task block from GFP-last recency. |
| Follow-up formal: task_block_gfp_first seed 20260928 | Pending | 1,152 updates; retain model; no test inference. |
| Follow-up smoke: task_block_gfp_last seed 20260927 | Pending | Cross-seed check for the restored path. |
| Follow-up formal: task_block_gfp_last seed 20260927 | Pending | 1,152 updates; retain model; no test inference. |
| Follow-up summary audit | Pending | Write combined `analysis/summary.json` after both follow-up runs complete. |

Current status: first diagnostic complete; follow-up matrix prepared. The `task_block_gfp_last` path restored GFP from near-constant output: dev RMSE/MAE/Spearman/prediction SD = 0.5953/0.3578/0.5588/0.5422.
