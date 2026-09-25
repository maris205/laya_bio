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

All ten completed by 2026-09-24 09:09:47 UTC. [Report and evidence](../research/laya_readout_diagnostics.md). No dev/test rows were read. Follow-up: the fixed-configuration longer-budget check below is now complete.

### GFP convergence follow-up

| Run | Scope | Status |
|---|---|---|
| GFP-CONVERGENCE-512 | Same native-readout configuration, 32 training examples, update budget 128→512 | COMPLETE 2026-09-24 09:59:24 UTC; first 128 optimization records and predictions exactly match; both fitting gates pass at updates 384, 448 and 512 |

[Report and evidence](../research/laya_convergence.md). Final Accuracy 96.88%, NLL 0.04707, scalar MAE 0.03367, RMSE 0.04596. No dev/test evaluation. Follow-up: the 1,024-example GFP training/development validation below is complete.

### GFP full-subset development validation

| Run | Scope | Status |
|---|---|---|
| GFP-DEV-1024 | Mean pooling + fresh readout, 1,024 train / 128 dev, 512 updates / 16 epochs, warmup + cosine LR | COMPLETE; near-constant predictions persist; best-dev step 192 RMSE 0.86940, final RMSE 0.87097; position ridge RMSE 0.69972 |

[Report and evidence](../research/laya_gfp_development.md). Run finished 2026-09-24T12:00:36.724564+00:00. All 512 updates finite, each training example seen exactly 16 times, nine evaluation points retained, selected weights save/reload probability and scalar errors both zero. No new test access. This reused development set is not a confirmatory evaluation. Next: training-only diagnosis of sample-scale / per-example-exposure / optimizer sensitivity; do not expand the formal multitask suite yet.

### GFP training-only scale / learning-rate controls

| Run | Scope | Status |
|---|---|---|
| GFP-SCALE-LR | N32 high/low LR ×512; N1024 low LR ×1,024; historical N1024 high LR training-only reference | COMPLETE 2026-09-24 13:57:47 UTC; 55.00 min, 2,048 new updates; N32 low-LR RMSE 0.02420; N1024 low-LR RMSE 0.80900→0.70786 between 512→1,024 updates |

[Report and evidence](../research/laya_gfp_scale_probe.md). All new outputs are training-only. Matched initialization and batch orders verified; all recorded metrics recomputed; all three final checkpoints reload exactly. Lower LR improves tiny-set fitting, while the longer large-subset run partly recovers from near-constant prediction but still underfits (normalized RMSE 0.87320). Same-exposure comparisons do not fully isolate sample size from update count/LR history. Next: a predeclared longer-budget convergence check; exact optimizer continuation would require replaying the recorded prefix because checkpoints contain model state only. No further run or dev/test evaluation launched.

## Active route change — 2026-09-24

The user paused the direct-SFT/GFP diagnostic route. The proposed 2,048-update GFP extension is PAUSED and will not launch. Existing results/checkpoints remain preserved. [Route-change record](ROUTE_CHANGE_20260924.md). Active work now follows vocabulary expansion → embedding/MLM-head adaptation → biological CPT → matched no-CPT/CPT single-task SFT with thousands of examples → cumulative multitask SFT.

### 2026-09-24: conventional CPT first paired result (partial)

CPT completed 128 new-row/head warmup + 1,024 full-encoder updates with verified old-row invariance and exact checkpoint reload. On promoter detection with 4,096 labels and matched three-epoch SFT, no-CPT/CPT development accuracy is 82.51%/87.74%, a +5.23 pp paired difference (descriptive group-bootstrap 95% interval [+3.14,+7.32] pp). Both 16,766-example fits remain in the frozen running queue. One seed; no test inference, few-shot or zero-shot claim. [Interim report](../research/laya_biocpt_round1.md).

### 2026-09-24: protein-source defect; supersede v1 and repeat complete matrix

A whole-file scan of historical protein_uni_16.txt (16,955,660,631 bytes; SHA256 fc8de807efdfc94b51d54bfd73a66403b76c8a49bd0b056184356736bc3d5876) finds zero N/n characters. Preserve v1 CPT/4K diagnostic results; stop its incomplete full-data queue. Switch to historical LucaOne protein data, require all 20 canonical residues, verify old-source rejection and unchanged DNA/SFT hashes, and repeat the same complete CPT + four-SFT matrix as v2. No changes motivated by selecting better development scores. Root cause upstream unknown. [Superseded diagnostic](../research/laya_biocpt_source_diagnostic.md), [corrected round](../research/laya_biocpt_round1.md).

### 2026-09-24: corrected-source 4K pair complete

With canonical-residue-complete admitted protein data, 128+1,024 CPT updates and identical 192-update promoter SFT produce 82.51%/87.55% no-CPT/CPT development accuracy (+5.04 pp; descriptive paired-group 95% interval [+3.14,+7.03] pp). Macro-F1: 0.8244/0.8755. Protein N receives 27,991 natural occurrences and 4,196 targets. Old-row warmup invariance and all checkpoint reload checks pass. No-CPT repetition is exact, not an independent seed. Full-data pair remains running. [Corrected primary report](../research/laya_biocpt_round1.md).

### 2026-09-24 UTC: corrected traditional CPT/SFT round complete

All five fits completed in 70.54 minutes. Promoter dev no-CPT/CPT accuracy: 4K 82.51%/87.55% (+5.04 pp, descriptive paired-group 95% interval [+3.14,+7.03]); full16,766 90.02%/89.92% (−0.10 pp, interval [−1.43,+1.24]). Full-data train accuracy 92.32%/95.12%: stronger training fit without dev-accuracy improvement. All 24 metric points independently reproduced; 1,152 CPT sampling/masking steps and saved RNG/sampler state match replay; all retained checkpoint hashes and reloads pass. Actual CPT input 6,770,136 tokens / 990,605 masked targets. Protein N has 24,210 natural visible inputs and 4,196 targets after the source repair. Single seed, reused dev, no test inference or few/zero-shot claim. Old route remains paused. [Final report](../research/laya_biocpt_round1.md).

## 2026-09-25: one-model JEV-style three-task decision round

User confirmed the primary objective is one checkpoint, multiple tasks, unified JEV-style output; few-shot is deferred. Continue Laya for one bounded target-form experiment before deciding whether to switch to Qwen adaptation. [Active plan](EXPERIMENT_PLAN.md).

| Run | Purpose | State |
|---|---|---|
| JEV-DATA | 8,192/task, full dev, direct BIO IDs, GFP–CPT admission | COMPLETE; zero 15mer overlaps and no truncation |
| JEV-SMOKE | Shared typed scorer; all three primitives; reload | PASS |
| CPT-JOINT / NO-CPT-JOINT | One model across Noul/Choice/Score | RUNNING queue; 1,152 updates each |
| CPT-PROMOTER / CPT-STRUCTURE / CPT-SCORE | Same-interface single-task references | PLANNED; 384 updates each |

Old GFP microfit extension remains paused. Previous conventional-head CPT results remain archived; this round has not yet produced development results.
