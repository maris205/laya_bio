# JEV GFP Single-Task Seed Control Tracker

| Run ID | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| smoke-no-cpt-gfp-20260927 | Check fluorescence-only path | NO-CPT, seed 20260927 | train-only | finite loss/gradients, exact reload | MUST | COMPLETE | One effective fluorescence update; passed |
| smoke-cpt-gfp-20260927 | Check fluorescence-only CPT path | CPT, seed 20260927 | train-only | finite loss/gradients, exact reload | MUST | COMPLETE | One effective fluorescence update; passed |
| reuse-cpt-gfp-20260926 | Existing single-task anchor | CPT, seed 20260926 | dev/train diagnostic | RMSE, MAE, Spearman, prediction SD | MUST | COMPLETE | Existing audited round-1 output; model not retained |
| no-cpt-gfp-20260926 | Complete seed 20260926 pair | NO-CPT, seed 20260926 | dev/train diagnostic | RMSE, MAE, Spearman, prediction SD | MUST | COMPLETE | RMSE 0.8106, MAE 0.6460; MAE gate failed |
| no-cpt-gfp-20260927 | Matched single vs joint control | NO-CPT, seed 20260927 | dev/train diagnostic | RMSE, MAE, Spearman, prediction SD | MUST | COMPLETE | RMSE 0.7686, MAE 0.5953; MAE gate failed |
| cpt-gfp-20260927 | Matched single vs joint control | CPT, seed 20260927 | dev/train diagnostic | RMSE, MAE, Spearman, prediction SD | MUST | COMPLETE | RMSE 0.8207, MAE 0.6250; MAE gate failed |
| no-cpt-gfp-20260928 | Check near-constant joint failure | NO-CPT, seed 20260928 | dev/train diagnostic | RMSE, MAE, Spearman, prediction SD | MUST | COMPLETE | RMSE 0.6303, MAE 0.4781; passed and reversed joint collapse |
| cpt-gfp-20260928 | Matched single vs joint control | CPT, seed 20260928 | dev/train diagnostic | RMSE, MAE, Spearman, prediction SD | MUST | COMPLETE | RMSE 0.8296, MAE 0.6178; MAE gate failed |
