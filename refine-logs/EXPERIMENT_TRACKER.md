# Jev-style biological decision experiment tracker

Updated 2026-09-24. **Formal suite prospective; two six-task 100-update engineering pilots are complete; the three-pass development round is complete (dev only).** Task IDs are defined in the [24-view catalog](../research/biological_decision_task_catalog.md). Core task IDs C01–C12 are distinct from run IDs below.

| Run/work ID | Purpose | Status |
|---|---|---|
| LEGACY-12 | Existing two-task raw/BPE/B1/text-only × 3 seeds | COMPLETE, unchanged |
| SOURCE-LOCAL | Eight local single-sequence files, 79,090 rows; schema and membership audit | COMPLETE; homology/tokenizer admission pending |
| SOURCE-HF | Four DNA pools and eight gene_lan_transfer configs; pinned downloads and checks | COMPLETE; source cards do not fully define provenance |
| SOURCE-PAIR | Two local protein homology pair views; parser and exact endpoint split check | COMPLETE; biological homology-cluster independence unverified |
| SOURCE-LIT | TAPE, DeepLoc, GO and optional benchmarks | TAPE fluorescence + DeepLoc data downloaded; GO/stability and full admission TODO |
| DATA-V2 | Global entity/RC/homology groups, preserved old test, fresh evaluation, common length subset | TODO |
| CONTRACT-V2 | Choice/Score/Noul, masks, pair roles, native-unit Score mapping, generator constraints | Candidate/shared-head implementation + 5 regression checks PASS; generator path TODO |
| PILOT-C | Six tasks, original Laya shared scorer, 100 updates | COMPLETE; reload PASS, engineering dev only |
| PILOT-HM | Matched shared encoder with task heads, 100 updates | COMPLETE; reload PASS, engineering dev only |
| PILOT-G | Same six tasks, generator SFT | TODO; common generator-tokenizer eligibility pending |
| MAIN-C-s1..s3 | Shared typed scorer, admitted core suite | TODO |
| MAIN-HM-s1..s3 | Shared encoder with proper task-specific output heads | TODO |
| MAIN-G-s1..s3 | Joint SFT; same checkpoint for free/constrained/likelihood inference | TODO |
| ANCHOR-HS | C01/C07/C09/C11 × 3 seeds | TODO; 12 independent fits |
| B1-QUALITY | Primitive-appropriate quality, strict validity and calibration | TODO |
| B3-SEMANTICS | Questions, labels, candidate order/IDs, symmetric pair swaps | TODO |
| B4-COST | Full task/panel latency, throughput, memory, failures | TODO |
| LOTO-C06 | Exclude C06/C11 localization supervision, target C06, C/G × 3 seeds | TODO; 6 fits |
| LOTO-C10 | Exclude stability supervision; external score anchors required; C/G × 3 seeds | TODO; 6 fits |
| BPE-V2 / LOWSHOT | Representation and sample-efficiency extensions | OPTIONAL, not scheduled |

Main count: 9 joint + 12 anchor fits = **21**. Transfer adds **12**, total **33** if all gates pass. The count is not a completed run report or a compute reservation. New GPU-hour estimates depend on the six-task pilot and final admitted data/label counts. Tasks failing provenance, grouping, context length or independent-evaluation gates remain visibly unadmitted.

Pilot results and their narrower admission scope are tracked in the [execution report](../research/laya_multitask_pilot.md). Exact/RC and pair-endpoint separation do not close the global homology gate.

| Additional run | Scope | Status |
|---|---|---|
| DEV-ROUND1-C/HM | Same 6,144-entity training subset; each task three passes; 576 updates/model | COMPLETE 2026-09-24 04:34 UTC; both 576 steps, hashes/reload PASS; 55.54 minutes total; dev only |
| DEV-DIAG-S/GFP-C/HM | Four single-task controls, 96 updates each, exact target-batch/LR replay | COMPLETE 2026-09-24 07:01 UTC; initialization/batches/reload checks PASS; no clear recovery from single-task training |
| DEV-DIAG-S-NAMES | Splice candidate-name correction, 96-update candidate control; existing joint train-subset evaluation | COMPLETE 2026-09-24 07:05 UTC; corrected names still majority-only; full upstream lineage remains unverified |

See [diagnostic report](../research/laya_task_diagnostics.md). Historical splice label-name matching is internal consistency only; biological interpretation of the historical names is withdrawn. Current builds use the source-code-supported correction documented in the report. No formal-suite runs or new test inference were added.

### Training-only microfit follow-up

| Run | Scope | Status |
|---|---|---|
| MICROFIT-BASE/LR | Four base runs and conditional GFP candidate LR control; 32 training examples, 128 updates each | COMPLETE; candidate can fit Choice/Score; fixed Choice panels show order sensitivity |
| MICROFIT-ORDER | Per-update Choice permutation; six-order evaluation on the same fitted sequences | COMPLETE; all six orders 100% on the 32 training examples; no generalization claim |
| MICROFIT-HM-LR | Additional matched 1e-4 GFP shared-head control | COMPLETE; still fails fitting; readout/loss diagnosis required |

All seven runs completed by 2026-09-24 07:56 UTC. [Report and evidence](../research/laya_microfit.md). No dev or test samples were read in these checks.

### GFP readout/loss follow-up

| Run | Scope | Status |
|---|---|---|
| GFP-POOL-LOSS | CLS/mean × joint/CE/MSE; six matched 128-update fits | COMPLETE; no supervised branch passes its final fit gate |
| GFP-ADAPTER | Three conditional structural ablations and one MSE-only control | COMPLETE; fresh readout improves MAE/RMSE, but no final fit gate passed |

All ten completed by 2026-09-24 09:09:47 UTC. [Report and evidence](../research/laya_readout_diagnostics.md). No dev/test rows were read. Next: fixed-configuration longer-budget convergence check; not yet run.
