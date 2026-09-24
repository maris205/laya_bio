# Experiment plan: Jev-style decision models for biological sequences

Date: 2026-09-24. **Status: formal design remains prospective; two separate six-task engineering pilots are complete, and the three-pass development round is complete (dev only).** The existing twelve-model results remain an initial study. This redesign was written after those test results were seen; it is not preregistration of the original experiment.

## Problem anchor

A biological decision model should reuse one sequence/question/candidate interface across biological decision tasks and preserve the identity of the supplied output labels. The core question is whether that interface gives a useful combination of task reuse, predictive quality, and operational reliability relative to fixed-head and sequence–language alternatives. BPE is a supporting representation choice.

Many downstream tasks are categorical, ordinal, binary, or multilabel. The expanded design maps them to Choice, Score, and Noul; sequence generation and structure reconstruction remain outside its scope. Existing checkpoints only implement the tested Choice path. Shared encoders with multiple heads already support several tasks. Candidate membership is guaranteed by index lookup, but biological correctness, question understanding, order stability, and unseen-task transfer must be measured separately.

## Claim map

| Claim | Minimum convincing evidence | Current evidence | Required blocks |
|---|---|---|---|
| C1: A shared candidate scorer provides useful biological decisions while preserving supplied label identities | Joint-task accuracy/F1; exact ID/label lookup; compare with both shared fixed heads and free/constrained generation | Joint DNA/protein results; structural in-set output; BPE-matched B1 comparison | B1, B4 |
| C2: Task and candidate inputs enable reuse beyond a fixed output inventory | Multiple tasks in one modality; controlled question/label changes; held-out-task evaluation before a transfer claim | Not established: present task identity is confounded with modality | B2, B3 |

Anti-claims to rule out: one model per task is necessary; any zero-invalid-label result demonstrates Jev superiority; BPE is the cause of the interface benefit; a scorer merely learns class positions or modality; constrained generation cannot be equally valid.

## Comparison systems

Three families, with the generator's output interfaces sharing **one identical trained checkpoint**:

| ID | System | Task-specific parameters | Output path | Role |
|---|---|---|---|---|
| C | Laya-Bio shared candidate scorer | No task-specific output matrix | Candidate probabilities → argmax index → exact supplied label | Main method |
| H-M | Same Laya initialization, shared encoder, separate task heads | One linear matrix per trained task | Head probabilities → canonical label | Essential fixed-head control |
| H-S | Same initialization, separate encoder/head per task | Whole model per task | Head probabilities → canonical label | Sharing/accuracy/storage comparison |
| G-F | Joint sequence–language SFT model | No task-specific matrix | Greedy unconstrained answer text | Free generation control |
| G-C | Same checkpoint as G-F | None added | Token-prefix constraints over complete allowed labels plus EOS | Constraint control |
| G-L | Same checkpoint as G-F | None added | Teacher-forced log-likelihood of each exact allowed label plus EOS | Finite-choice likelihood control |

C versus H-M uses the same raw representation, backbone initialization, data exposure, optimizer budget, and training seeds. Retain the existing BPE-matched comparison as historical evidence; the old study lacks a raw H-M run. Pooling differences must still be disclosed. A shared C encoder plus opaque/nonsemantic label ablation can probe semantics but must not be described as an exact causal isolation of the whole architecture.

The concrete resource-conscious generator starting point is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), revision `c1899de289a04d12100db370d81485cdf75e47ca` (public metadata checked 2026-09-24). Jointly SFT its canonical-label answers on exactly the same biological training examples, without extra CPT. Disable thinking in the documented chat template. It is a small sequence–language baseline, **not a reproduction of LLaMA-Gene or ChatNT** and not a parameter/pretraining-matched causal control for ModernBERT. Report those differences. A published biological generator is a useful later external baseline if compute permits, not a replacement for the same-checkpoint decoding comparison.

## Typed output design and 12-task scope

The [task catalog](../research/biological_decision_task_catalog.md) provides 12 priority tasks and 12 extension views with primary sources, audit status, and caveats. These are proposed tasks, not new results. Core IDs C01–C12 are promoter, splice, TF binding, structural class, Sec/Tat, single-label prokaryotic localization, protein homology pairs, DNA–protein coding pairs, GFP fluorescence, stability, multilabel localization, and GO molecular function. C08 is a translation-consistency diagnostic, not independent evidence for broad natural biological prediction.

- **Choice:** mutually exclusive candidates; CE; exact supplied ID lookup.
- **Noul:** probability that a proposition is true, implemented initially through shared yes/no candidate scoring and binary CE. Use it for binary tasks and separately for each multilabel proposition. Multilabel probabilities are marginals, not one softmax across labels. Known-target masks and native annotation-recovery conventions must be explicit; absent GO annotation is not an experimentally confirmed negative.
- **Score:** distribution over ordered levels, with expected level index. For native continuous assays, the initial design uses five ordered bins, thresholds fit on training data only and anchors equal to training-bin means; merge tied thresholds and report the resulting level count. Predict the original assay value by the anchor expectation. Use CE initially, an ordinal-loss ablation only if scheduled before evaluation. Report native-unit MAE/RMSE and Spearman, oracle quantization error and out-of-range coverage. Add H-M/H-S scalar MSE regression controls; an ordinal comparison alone cannot establish an advantage for continuous prediction.
- Shared scoring parameters can implement all three paths; any type-specific projection must be counted and shared across tasks of that type. This is proposed implementation, not a capability established by released weights.
- Preserve sequence roles and boundaries for pairs. Protein homology supports an A/B swap diagnostic; DNA/protein role exchange is not a meaning-preserving transformation.

For G-F/G-C/G-L, use the same trained generator for every inference mode. Multilabel inference asks one yes/no proposition per label, avoiding enumeration of all label subsets; Score uses the ordered level strings. The same proposition exposures apply to C and G, and total sequence encodings and full-label-panel runtime must be reported. A supplementary numeric-generation baseline may address native continuous values, but is outside the minimum run count. For GO start with the 32 most frequent training MF terms (ties by stable GO ID), freeze ontology/date and ancestor propagation, and call the resulting task **GO-MF-Lite**, not full GO prediction.

## Data and split gate

1. Preserve `legacy_v1` numbers, manifests and all old test memberships. The four upstream DNA repositories expose a single pool named `train`, which contains old test members; that pool must never be ingested as training wholesale. The source-only audit inspected public labels for provenance/translation checks and is not a blind performance evaluation.
2. Core pilot set: C01/C02/C04/C07/C09/C11, spanning all three primitives, two modalities, pairs, and multilabel output. Expand to all 12 only after admission. Local/upstream audits are complete for available files; TAPE, DeepLoc and GO data still need downloading, version pinning, and admission. Sources and hashes are in `artifacts/laya_task_expansion/`.
3. Freeze source file/revision, label ontology, provenance, component terms, task semantics and tokenizer eligibility before training. Some source cards contain only YAML and do not establish biological provenance. Quarantine conflicting sequence groups rather than resolving them with test labels. Do not equate synthetic similarity with natural homology.
4. Build a global cross-task sequence/RC/homology-parent registry. All annotations and all pair endpoints for a group remain on one side of train/dev/calibration/test. Use genomic locus/chromosome grouping when coordinates exist; otherwise disclose unavailable coordinates and use sequence-cluster grouping without claiming chromosome independence. Protein clustering tool/version/identity/coverage must be fixed and reported. Exact deduplication alone is insufficient.
5. For pair sources, partition endpoint/parent groups before generating negatives, or split existing pairs by connected components. Report giant components; if independent partitions cannot be formed, rebuild with documented construction or reject the task. Check length/composition and alignment-similarity baselines. Add deterministic standard-code translation for C08: it perfectly separates the downloaded original and rand_v2 pools, so model accuracy there is not evidence of general biological reasoning.
6. The existing local audits found subcellular train members in old fold/signal test. Preserve old test groups globally and remove affected training membership in the new protocol. Do not reshuffle the old evaluation into training. Retain native public benchmark evaluation definitions as a separate comparable view, subject to cross-task leakage audit.
7. Confirmatory claims require a frozen, independently held-out evaluation manifest created before model development on that evaluation. Public legacy benchmarks and previously inspected labels are exploratory replication evidence; if no new uninspected independent set is available, explicitly keep that claim exploratory. Neither fresh random splits nor a new filename restore blindness.
8. Compute actual model-tokenizer lengths for the full question, all sequences and output definitions. Freeze a common full-input subset and excluded counts per task/class/model; no silent truncation. Long DNA–protein pairs and localization proteins make this a material gate. Across systems, use the same eligible biological entities.
9. Use independent biological annotations for multiple questions about the same sequence. Template edits do not create a new task. Multilabel thresholds and temperature fits use calibration only, never test; benchmark-specific Fmax may be reported only as a clearly marked oracle-threshold metric, alongside frozen-threshold F1.

## B1 — Valid labels and useful predictions (MUST)

**Claim:** C1. **Main table:** rows C/H-M/H-S/G-F/G-C/G-L; columns per-task and within-primitive summaries, strict invalid rate, canonical mapping-failure rate, wrong-in-set rate; an additional cost table covers B4.

Use the same held-out examples, candidate definitions, and three seeds. G-F/G-C/G-L use identical prompts and checkpoint. For G-F/G-C, greedy decoding with no sampling; `max_new_tokens` is fixed from the longest tokenized allowed label including EOS plus a documented margin, not chosen after seeing test outputs. Verify the constrained trie against the exact tokenizer/context boundary; EOS is legal only at an allowed-label leaf. Truncation, runtime failure, or no legal terminal output is a failed response, not dropped from the denominator.

For G-L, score the **complete** label plus EOS by sum of conditional token log-probabilities and normalize across candidates. Record length bias and an optional length-normalized dev diagnostic; never select the formula on test. These are candidate-renormalized scores, not the probability of successful unconstrained generation. For G-C, probabilities require this separate complete-label scoring pass; include that cost when comparing probability-producing interfaces.

Metrics have separate definitions:

1. Strict contract violation: decoded answer, after a predeclared leading/trailing whitespace rule, is not exactly one permitted canonical string/ID. Explanations and synonyms violate the strict contract even if interpretable.
2. Semantic mapping failure: a frozen alias table cannot map the response to exactly one allowed class; multiple/contradictory labels are ambiguous. Report benign aliases separately from invented labels. No fuzzy substring extraction or LLM relabeling after outcomes are seen.
3. Wrong-in-set: a valid canonical class differs from the gold class. Do not relabel this as output drift.
4. End-to-end accuracy/F1: failed or invalid outputs are errors on the original denominator. Also report alias-normalized performance and valid-only accuracy, clearly secondary.
5. NLL, Brier, ECE and calibration coverage for systems with a defined full candidate distribution; never substitute self-reported prose confidence.
6. Binary tasks: AUROC/AUPRC, MCC and frozen-threshold F1. Multilabel: micro/macro-AUPRC, micro/macro-F1, per-label support and missing-target handling; do not use subset accuracy alone. Score: native-scale MAE/RMSE, Spearman, level calibration and discretization diagnostics. Do not average accuracy, AUPRC, and correlation into a fabricated overall score. Invalid Score outputs remain in a reported failure rate; rank/error metrics on valid outputs must disclose coverage and a predeclared train-mean fallback score on the full denominator.

**Success:** a useful accuracy/reuse/cost tradeoff against H-M and G-C/G-L, while keeping the output contract intact. Lower invalid rate than G-F alone is insufficient. **Failure:** if G-C/G-L equal or dominate the method, narrow the contribution to an open biological encoder adaptation; do not claim a unique anti-hallucination mechanism. Structural zero-invalid behavior is not a statistical performance discovery.

## B2 — Task sharing and held-out task transfer (MUST for generality)

**Claim:** C2. Compare the joint C and H-M models with H-S across the admitted suite, with four independent-model anchors (C01 promoter, C07 homology pairs, C09 fluorescence, C11 multilabel localization). Match per-task example presentations across joint and single-task runs and report total compute/storage separately. Joint sampling uses a frozen balanced task schedule for the redesigned study; do not claim that this is the old study's sampler. Report per-task results so gains on large tasks cannot hide negative transfer.

For each task, provide two alternate, meaning-preserving question templates, checked without inspecting test predictions. For transfer, train separate leave-one-task-out variants excluding the localization family (C06 and C11; target C06) or C10 stability, including equivalent annotations from other sources; evaluate C and G-F/G-C/G-L without held-out-task adaptation; H-M has no corresponding head and is `not applicable` for strict zero-shot, rather than assigned an arbitrary low accuracy. If low-shot adaptation is later claimed, give all systems the same budgets (e.g. 16/64/256 examples per class) and label those runs separately.

**Success:** useful performance on same-modality tasks and preserved performance under verified question changes; held-out-task scores are additionally required for an unseen-task claim. **Failure:** success only on fixed trained schemas supports multi-task supervised reuse, not general task understanding. No numerical transfer gain is assumed in advance. A checkpoint trained on all 12 tasks cannot be called zero-shot on any of those tasks. Keeping DeepLoc (C11) while withholding only prokaryotic localization (C06) would instead be a related-domain/schema transfer test, not the strict localization-family test. Score transfer requires externally specified levels and numerical anchors fixed without looking at held-out-task target values; if no such scale is defensible, C10 remains an adaptation experiment rather than strict zero-shot native-scale prediction. Do not derive held-out-task bins from its training targets while claiming no task supervision. Remove duplicate/related-source annotations and held-out-task calibration from each transfer training pool; report residual pretraining overlap uncertainty.

## B3 — Label identity and decision stability (MUST)

**Claim:** C2 and boundary of C1. Predefine original labels, independently verified aliases/short descriptions, and three deterministic candidate permutations per example. Align predictions to semantic IDs before comparison. Return opaque IDs with visible descriptions in a separately marked interface stress test, and randomize the ID/description association per example; use a no-description version to test the shortcut. The opaque-ID response format is a changed prompt condition for the generator, not automatically an in-distribution test.

Report accuracy/F1, strict validity, mapping failure, semantic prediction agreement, and probability divergence by intervention; do not equate unchanged accuracy with per-example stability. Audit ambiguous ontology names before aliases are admitted. Candidate removal with the true label absent belongs to a separate abstention/open-set experiment and is not pooled with this closed-set block.

**Success:** changes preserve useful semantic decisions, not just valid strings. **Failure:** order/wording sensitivity prevents a broad “no semantic drift” claim even if all outputs remain legal. Existing 9.04% BPE protein order flips already show this distinction.

## B4 — Cost and uncertainty (MUST for efficiency claims)

Use the same RTX 4080 SUPER, precision, and software versions. Fix input-length and candidate-count bins, 20 warm-up calls, 200 measured calls per bin (or all examples if fewer), and three repetitions. Measure CUDA-synchronized batch-1 median/p95 latency, a fixed feasible batch throughput, and peak allocated GPU memory. Include tokenization, construction of constraints, answer parsing, and complete candidate scoring if probabilities are requested; report model-only time separately. Report OOM/timeout coverage; no dropping failed long examples.

Time all labels in the requested multilabel panel; do not count one yes/no call as a complete 32-label prediction. Report number of encoder/generator calls and chunking overhead. Use a separate calibration split for all fitted temperatures. Show NLL/Brier/ECE together with accuracy. The original 41-minute training runs are not inference-latency evidence. **Failure:** no measured throughput/latency benefit means no speed claim, regardless of non-autoregressive output.

## Training recipe and compute planning

- Seeds: 20260922, 20260923, 20260924; new data manifests remain distinct from old ones.
- C/H-M: raw tokenizer, original Laya initialization, BF16, effective batch 32, initial AdamW LR 2e-5, weight decay 0.01, warm-up 5%; no additional CPT. CE/BCE or MSE according to the predeclared output path. H-M uses appropriate binary/sigmoid/scalar heads rather than an incompatible all-purpose softmax.
- Task-balanced schedule: after admission, set per-task entity presentations to three times the smallest admitted task's training count (sample larger tasks without replacement per cycle). Record both biological-entity presentations and expanded label-proposition presentations. Normalize the multilabel loss per observed label then per example, so many-label tasks do not dominate just by vocabulary size. Match entity exposures for C/H-M/G; disclose the additional sequence calls needed by proposition methods.
- G: canonical-answer SFT, answer-token loss only, no thinking; same entity/panel exposure schedule. Use BF16 and gradient accumulation/activation checkpointing as needed on the available RTX 4080 SUPER (the current driver reports approximately 32 GiB; record each run’s actual capacity). A 100-update train/dev pilot determines feasible batch/memory/time before freezing any recipe adjustment. Full tuning versus LoRA must be disclosed if a memory gate forces a change.
- Main suite: C/H-M/G × 3 seeds = **9 joint fits**. H-S for four anchors × 3 seeds = **12 fits**, giving **21 main fits**. H-S is therefore not an all-12-task claim. Generator free/constrained/likelihood modes share the same 3 trained G checkpoints. Score tasks in H-M carry a scalar control and an ordinal control; count both heads/losses and disclose joint supervision. If interference makes this comparison unsuitable, use separately budgeted controls instead of hiding extra fits.
- Transfer extension: two leave-one-task-out settings (C06 and C10) × two eligible interface models (C and G) × three seeds = **12 additional fits**. Strict zero-shot fixed heads are not applicable. Full plan with transfer is **33 fits**; excluding those runs permits only supervised multitask claims.
- Pilot compute allowance: at most 2 GPU-hours before re-estimating. Do not inherit the earlier three-task 12–30-hour estimate for this expanded suite. After the pilot publish `sum(fits × admitted updates × measured seconds/update) + all-mode evaluation + calibration + timing`, with task-length and label-panel factors and 20% scheduling contingency. Dataset sizes after grouping/eligibility and multilabel expansion are not yet known; a total GPU-hour promise now would be unsupported.
- Failure gates: inadequate independent evaluation, unresolved label provenance, unacceptable full-input coverage, giant paired components, or inference cost from many labels. A failed task is reported as unadmitted, not replaced silently with a near-duplicate.

## Run order and decision gates

| Stage | Work | Gate / status |
|---|---|---|
| M0a | Existing local pools + user-supplied upstream audits | DONE source/schema audits only; no frozen v2 split |
| M0b | Acquire TAPE/DeepLoc/GO; full provenance/group/tokenizer audit | TODO; freeze 6-task pilot then 12-task scope |
| M0c | Implement typed scorer, pair serializer, masked targets, score bins, parsers/trie | TODO; deterministic synthetic/dev checks and tiny train overfit |
| M1 | PILOT-C/HM/G, each 100 updates on six tasks | TODO; memory/loss/coverage/cost gate; no test access |
| M2 | C/H-M/G × 3 seeds plus four H-S anchors × 3 | TODO; 21 main fits, fixed recipe and per-task exposure |
| M3 | Development contract, semantic, calibration and generator modes | TODO; freeze aliases, thresholds, constraints and metrics |
| M4 | One frozen evaluation and same-hardware cost measurement | TODO; all failures/seeds reported, no test-based retuning |
| M5 | Separate leave-C06-out and leave-C10-out training/evaluation | TODO; 12 extra fits required before unseen-task claims |

The first model runs are **PILOT-C**, **PILOT-HM**, and **PILOT-G** on the six-task development suite, only after M0. The subsequent six-task engineering pilot is documented in [the execution report](../research/laya_multitask_pilot.md). It uses development data with exact/RC and endpoint-component guards while the full global homology/provenance gate remains open. It is not the confirmatory M2/M4 experiment. Its dev diagnostics cannot be promoted to new blind-test evidence.

## Scope cuts

Additional CPT, another BPE search, more upstream encoders, multi-agent reasoning, proprietary Jev replication, and broad open-ended biological reasoning are outside the core story. A larger biological generator is an optional external validation after the shared-checkpoint decoding comparison. Do not claim “first” without a separate novelty search.

## Sources and evidence status

- [Official Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [TypeSafe interface documentation](https://docs.typesafe.ai/introduction): naming and decision-interface motivation only; no imported biological performance claim.
- [Laya implementation](https://github.com/NandhaKishorM/laya): open backbone and candidate interface, pinned to the original project revision in the paper.
- [PICARD, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.779/): precedent for constrained decoding, not a biological benchmark result.
- The old frozen results and [new framing note](../research/jev_style_reframing.md) distinguish completed evidence from this plan. No external reviewer score or new model result has been fabricated.
