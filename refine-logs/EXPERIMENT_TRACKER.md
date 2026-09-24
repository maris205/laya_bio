# Jev-style biological decision experiment tracker

Updated 2026-09-24. **Prospective; no new model training or inference.** Task IDs are defined in the [24-view catalog](../research/biological_decision_task_catalog.md). Core task IDs C01–C12 are distinct from run IDs below.

| Run/work ID | Purpose | Status |
|---|---|---|
| LEGACY-12 | Existing two-task raw/BPE/B1/text-only × 3 seeds | COMPLETE, unchanged |
| SOURCE-LOCAL | Eight local single-sequence files, 79,090 rows; schema and membership audit | COMPLETE; homology/tokenizer admission pending |
| SOURCE-HF | Four DNA pools and eight gene_lan_transfer configs; pinned downloads and checks | COMPLETE; source cards do not fully define provenance |
| SOURCE-PAIR | Two local protein homology pair views; parser and exact endpoint split check | COMPLETE; biological homology-cluster independence unverified |
| SOURCE-LIT | TAPE, DeepLoc, GO and optional benchmarks | PRIMARY SOURCES VERIFIED; full data admission TODO |
| DATA-V2 | Global entity/RC/homology groups, preserved old test, fresh evaluation, common length subset | TODO |
| CONTRACT-V2 | Choice/Score/Noul, masks, pair roles, native-unit Score mapping, generator constraints | TODO |
| PILOT-C/HM/G | Six tasks C01/C02/C04/C07/C09/C11; 100 updates per system | TODO after admission |
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
