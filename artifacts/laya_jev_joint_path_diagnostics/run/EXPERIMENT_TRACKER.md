# JEV joint-path diagnostic tracker

Updated: 2026-09-26 UTC.

| Step | Status | Notes |
|---|---|---|
| Freeze diagnostic plan | Complete | First run is `task_block_gfp_last` for seed 20260928 NO-CPT. |
| GPU and disk preflight | Complete | RTX 4080 SUPER idle; about 89 GiB free under `/root/autodl-tmp`. |
| Smoke: task_block_gfp_last | Pending | Must complete promoter, structural_class, fluorescence smoke updates with finite non-zero gradients. |
| Formal: task_block_gfp_last | Pending | 1,152 updates; retain model; no test inference. |
| Summary audit | Pending | Write `run/analysis/summary.json` after formal run completes. |

Current status: plan prepared; training not yet complete.
