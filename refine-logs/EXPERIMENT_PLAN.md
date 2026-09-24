# Active experiment plan: biological vocabulary → CPT → supervised classification

Date: 2026-09-24. **Active revision: v2, correcting the protein source after the v1 composition audit.** The user paused the previous direct-supervision/GFP diagnostic route. Its reports and checkpoints remain intact; the proposed 2,048-update GFP extension will not run. See [route-change record](ROUTE_CHANGE_20260924.md) and [archived interface plan](EXPERIMENT_PLAN_interface_20260924_paused.md).

## Problem and claims

**Problem:** Small diagnostic training sets and a text-derived encoder without biological CPT do not establish a useful biological adaptation pipeline. Establish a conventional, well-audited baseline with thousands of labeled examples before expanding tasks.

**Primary hypothesis:** Biological masked-language CPT, including correct adaptation of newly added token embeddings, improves downstream classification or its labeled-data efficiency compared with the same vocabulary/model trained directly with supervision. This is a hypothesis, not an assumed benefit.

**Supporting hypothesis:** Cumulative multitask SFT can learn additional tasks while retaining previously learned tasks. This requires explicit per-task development metrics and replay of prior training tasks.

Do not infer zero-shot capability from MLM improvement. A fresh fixed classification head is not a meaningful zero-shot system; a semantic candidate interface requires separate alignment and evaluation.

## Paused evidence

The old two-task study and six-task engineering pilots remain no-additional-CPT evidence. The latest GFP train-only diagnosis reached 0.70786 training RMSE on 1,024 examples after 1,024 updates, still underfitting. It is not a CPT result or a sufficient-data SFT evaluation. No historical manuscript numbers are overwritten.

## First-round systems

| Arm | Vocabulary and initialization | Before SFT | Classifier and SFT protocol |
|---|---|---|---|
| A: no CPT | Same new DNA/protein BPE, mean of original raw-fragment embeddings | No unsupervised embedding adaptation or CPT | Identical fresh mean-pooled LayerNorm/linear head; same data/order/seed/budget |
| B: CPT | Exactly the same vocabulary/initial encoder | New-row/MLM-head warmup, then full-encoder MLM CPT | Identical fresh classifier initialization and SFT protocol |

The backbone starts from the original Laya ModernBERT encoder. The previous Laya contextual typed head is not used in this conventional fixed-head baseline. The original checkpoint lacks its upstream MLM prediction head, so a new MLM head must be initialized and trained explicitly. MLM-head learning is logged separately from subsequent full-encoder CPT. CPT uses ModernBERT masked prediction, not an autoregressive objective.

Historical raw-tokenizer results are context, not a matched control for this new classifier. A raw-tokenizer no-CPT control can be added later if vocabulary effects need isolation; the primary A/B comparison holds vocabulary expansion fixed.

## Data and representation

- First supervised task: promoter detection, binary, **16,766 train / 1,052 selection-dev** in the existing frozen formal split. The 4,096-example subset is a deterministic balanced subset of full train.
- Next supervised task: protein structural class, seven classes, **15,593 train / 939 selection-dev** before common input-eligibility checks. Minority classes need per-class reporting.
- Initial CPT snapshot: 32,768 DNA + 32,768 protein + 8,192 text training examples; validation 512 DNA + 512 protein + 128 text examples. Deterministic random byte-offset sampling of local historical corpora is length-biased and is not called record-uniform sampling.
- Fresh 1,024-entry source BPE per biological modality, fit only on admitted CPT training samples. Source padding/unknown entries are excluded from additions; each biological piece gets a distinct modality-specific model ID. Natural-language IDs remain unchanged. Mandatory alphabet entries prevent silent character loss; unobserved added tokens are explicitly reported.
- Natural-language fragments initialize each new row by the mean of their original tokenizer embeddings. All original IDs and rows must remain unchanged at initialization. Biological encoding must round-trip exactly.
- A whole corpus line is excluded if it shares a 31-mer with protected DNA sequences (both strands), or a 15-mer with protected protein sequences. Protected sequences are the first two tasks' selection-dev/calibration/test memberships; targets are not used for this filtering and no test inference is performed. CPT train is similarly guarded against its own sampled biological validation sequences.
- This filtering is conservative and does not establish global homology or parent independence. New downstream tasks require a new contamination admission audit. Existing downstream evaluation memberships have been used historically; no new blind-test claim.
- CPT windows are at most 256 model tokens and log cropping; supervised promoter inputs must fit in full with zero truncation.

## Embedding and MLM acceptance checks

1. Preserve base token IDs and verify original embedding rows at initialization.
2. Track input and masked-target occurrences separately for every added ID. Decoder gradients alone do not establish meaningful input exposure.
3. Warmup freezes encoder parameters and old embedding rows. Old-row gradients are masked, the embedding parameter group has **zero weight decay**, and optimizer state is fresh; assert actual old-row invariance. Merely zeroing gradients under AdamW is insufficient.
4. Verify new embedding and MLM-head gradients are finite and nonzero, record per-row weight changes, and distinguish unobserved alphabet reserve rows.
5. After warmup, remove the old-row mask and train the encoder; log old/new-row changes and trainable parameter counts. Input embeddings and the MLM decoder remain tied.
6. Use fixed masked validation inputs for comparable MLM curves, plus dynamic training masks. Report validation losses by DNA/protein/text, including warmup and full-CPT phases separately.
7. Save and reload exact model predictions; save final CPT optimizer/RNG/sampler state so future CPT budget extensions need not silently restart AdamW.

Technical references: [ModernBERT MLM implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py), [PyTorch AdamW](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html). Runtime behavior is also checked against the locally installed versions.

## Initial bounded run matrix

Values below are the intended first recipe; verify feasibility and embedding behavior in a short smoke test before freezing the production snapshot. Any preflight-driven adjustment must be recorded before examining development results.

| Run | Purpose | Data | Intended budget | Status |
|---|---|---|---|---|
| DATA-01 | Admission, new vocabulary and full-task data | Filtered CPT + frozen SFT splits | CPU preprocessing | COMPLETE |
| EMB-01 | Validate embedding/decoder training correctness and cost | Synthetic tiny model + a short real-model train-only smoke | No result claim | PASS |
| CPT-01 | Warmup new rows/MLM head, then adapt encoder | Balanced CPT mixture | 128 warmup + 1,024 full-CPT updates; effective batch 32 | COMPLETE (v2) |
| A-4K / B-4K | Paired data-efficiency point | 4,096 promoter train | 3 epochs, effective batch 64 | COMPLETE (v2) |
| A-FULL / B-FULL | Paired sufficient-data anchor | 16,766 promoter train | 3 epochs, effective batch 64 | RUNNING (v2) |

CPT intended per-effective-batch mixture: 14 DNA / 14 protein / 4 text. MLM masking: 15% eligible body tokens with 80% mask / 10% within-modality random token / 10% unchanged. Ensure at least one eligible target per sequence. Target counts determine loss normalization across micro-batches. Use BF16 autocast and FP32 parameters; gradient checkpointing and clipping 1.0.

Warmup intended LR 1e-3 for new embedding rows and 1e-4 for the MLM head, old rows invariant. Full CPT intended encoder/embedding LR 2e-5, MLM-head LR 1e-4, linear warmup then cosine to 10% of peak. Embeddings/norms/biases have no weight decay; other matrices use 0.01. Freeze exact schedules after smoke timing.

SFT intended encoder/embedding LR 2e-5, fresh classifier LR 1e-4, 5% warmup then cosine; same seed, head initialization, batch sequence, three epochs and final-step reporting in A/B at each sample count. Full-data epochs have a normalized final partial batch. The 4K/full comparison is fixed-epoch, not equal-compute; report updates and presentations. No early stopping or selective omission of negative outcomes. Initial run uses one seed; three-seed confirmation and further data sizes follow only after the pipeline is operational.

## Metrics and decision gates

- CPT: masked NLL and masked accuracy by modality; fixed validation masks; actual input/masked-token counts; new-row coverage, gradient norms and weight changes; old-row invariance during warmup.
- SFT: fixed selection-dev Accuracy, Macro-F1, per-class recall/confusion, NLL; full-train metrics at the beginning/end and development curves per epoch. Failed/nonfinite runs are recorded, not dropped.
- Main A/B verdict uses paired development performance at the fixed final SFT step, not the best point selected retrospectively. Bootstrap uncertainty on a reused development set is descriptive, not fresh-test confirmation.
- Engineering gate: exact encoding/initialization, finite meaningful gradients, immutable old rows during adaptation, and checkpoint round-trip pass before production training.
- Scientific gate: CPT need not win. If it loses, inspect representation/forgetting/MLM exposure and retain A as a legitimate baseline. Do not repeatedly tune solely to force CPT superiority.
- No test metrics are computed in this first round. Dataset size is not a universal sufficiency guarantee; inspect class support and learning curves.

## Subsequent updates to both branches

1. Compare fixed CPT budgets and matched SFT training, while retaining the no-CPT reference.
2. Add seeds and a meaningful labeled-data curve, including few-shot points only after the full-data anchor is established.
3. Train A alone, then A+B, then A+B+C on each branch, continuing its own encoder and retaining previous task data. Record every old and new task after each stage. Single-task anchors and per-task exposure counts distinguish interference from budget changes.
4. Evaluate zero-shot only for a trained semantic label/candidate interface with a clearly withheld task; never relabel random-head behavior as biological zero-shot.

## Compute and storage

One local RTX 4080 SUPER, approximately 32 GiB reported GPU memory. About 15 GiB disk space was free at admission. Keep the final CPT model and resume state, and final full-data A/B weights. Small-data checkpoints may be removed only after verified reload as a predeclared policy; retain their predictions/metrics. Do not delete historical assets. Publish measured seconds/update after smoke before forecasting total runtime. This is the first bounded comparison round, not an unbounded search or a claim that all corpus bytes have been pretrained.

## Production recipe frozen after train-only smoke

GPU smoke passed on 2026-09-24: two new-row/head warmup updates, two full-encoder updates, and one 64-example SFT update. Old rows remained exactly invariant during warmup; new embeddings and MLM head had finite nonzero gradients; the encoder acquired gradients after unfreezing; saved/reloaded MLM logits were exactly equal. Peak allocated GPU memory was 7,995,594,240 bytes. Full-CPT updates took about 0.73 s and the first SFT update 1.28 s. These short-run timings imply roughly 60–75 minutes for the complete bounded round including initial/final full-training evaluation, periodic MLM validation, model construction and checkpoint writes.

Production micro-batch is frozen at 8. All planned budgets, schedules, data sizes and final-step metrics above are retained. The CPT training snapshot contains 73,728 sequences and 13,611,857 model tokens, but only 36,864 presentations are scheduled in this first CPT run; do not equate snapshot size with processed tokens or an entire epoch. The corrected snapshot has three unobserved added input tokens: DNA:N and protein:J/O; report their exposure separately. Canonical residue N is present in the corrected protein corpus. Original base vocabulary 50,368 expands to 52,412. SFT inputs require at most 93 model tokens, with zero truncation. Raw sample and source-BPE hashes are identical before/after the preprocessing speed fix.

## Source-quality correction, frozen before v2 training

The initial historical `protein_uni_16.txt` source failed a composition audit discovered while examining uncovered embedding rows. A complete 16,955,660,631-byte scan found zero uppercase N and zero lowercase n across 16,466,361 newline-delimited records. N is the canonical amino-acid code for asparagine, so this is not an acceptable missing reserve symbol for broad protein pretraining. Upstream processing history is unknown; no claim is made about how N disappeared. The current sampler only uppercases and slices and does not delete residues.

The v1 CPT and paired 4K results are preserved as a superseded source diagnostic (+5.23 pp on promoter development), not the primary clean-source result. Its full-data no-CPT run was stopped during epoch two and its full-data CPT run was not started. These outcomes are disclosed rather than dropped from the record.

The v2 protein source is the local historical `protein_lucaone_15g.txt`. Its admitted training sample contains all 20 canonical amino acids, including 459,623 N characters. A new pretraining admission gate rejects a broad protein corpus if any canonical amino acid is absent. An end-to-end negative test with the previous source fails on N exactly as expected. This is a composition check, not a complete upstream-provenance guarantee.

DNA source/BPE and all three promoter SFT JSONL files have identical hashes to v1. All optimizer settings, initial model, classifier, seeds, batch sizes, budgets and endpoint definitions remain fixed. Protein samples and protein BPE are regenerated from the substituted source; their MLM losses are not directly compared to v1 as a controlled vocabulary-only effect. Repeat all four SFT runs in v2, including both no-CPT baselines, to preserve an uncomplicated complete paired record. This source correction follows already-observed v1 development outcomes and remains exploratory; no blind confirmation claim.

Primary artifacts: `/root/autodl-tmp/jev_gene/artifacts/laya_biocpt_v2`. The complete corrected round is expected to require about 65–75 minutes after launch. The faulty source and all diagnostic artifacts remain intact.

## Corrected 4K result (full pair still running)

No-CPT/CPT development accuracy is 0.825095/0.875475 at the fixed 192-update endpoint, a +5.038 pp paired difference; macro-F1 is 0.824445/0.875456. Descriptive paired-group 95% interval: [+3.137,+7.034] pp. The repeated no-CPT baseline exactly matches the original 192-step trace and all six prediction files. Both full-data 786-update fits continue under the same frozen recipe.
