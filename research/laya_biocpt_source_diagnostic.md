# Biological vocabulary → CPT → conventional SFT: first paired round

**Status: superseded source diagnostic, 2026-09-24.** The protein source was subsequently found to contain zero N characters in a complete 16,955,660,631-byte scan. CPT and the 4K pair remain recorded, but the full-data no-CPT run was stopped during epoch two and full-data CPT was not started. These values are not the primary corrected-source result. See the [active corrected round](laya_biocpt_round1.md).

The current experiment follows the [frozen plan](../artifacts/laya_biocpt_v1/round/EXPERIMENT_PLAN.md). It replaces the previous small-data interface diagnostic as the active route, while preserving that route's results. This is a conventional mean-pooled encoder with a fresh LayerNorm/dropout/linear classification head; it does not evaluate the previous semantic candidate interface.

## Completed downstream comparison

Promoter detection uses 1,052 examples in the existing selection-development split. Both arms use the same newly expanded biological vocabulary, original encoder weights before adaptation, classifier initialization, training examples and order, three SFT epochs, optimizer schedule, and final-step reporting. CPT adds new-embedding/MLM-head warmup followed by full-encoder masked-language adaptation. No test inference is performed.

| Training examples | Arm | SFT updates | Train accuracy | Dev accuracy | Dev macro-F1 | Dev NLL |
|---:|---|---:|---:|---:|---:|---:|
| 4,096 | No additional CPT | 192 | 83.40% | 82.51% | 0.8244 | 0.3873 |
| 4,096 | CPT | 192 | 90.70% | 87.74% | 0.8774 | 0.2920 |
| 16,766 | No additional CPT | 786 planned | Stopped/not started | — | — | — |
| 16,766 | CPT | 786 planned | Stopped/not started | — | — | — |

The completed paired effect is **+5.23 accuracy percentage points**. A 10,000-replicate paired group bootstrap gives a descriptive 95% percentile interval of **[+3.14, +7.32] pp**. The development groups number 1,052. One seed and a historically used development set do not constitute independent test confirmation or training-seed uncertainty.

At the fixed 4K endpoint, class-0/class-1 recall changes from 89.44%/75.71% without CPT to 88.87%/86.63% with CPT. The improvement is primarily in class-1 recall. At epoch one, CPT reaches 82.98% accuracy versus 75.76% for no CPT; these learning-curve points supplement rather than replace the predeclared final-step comparison.

## CPT and embedding checks

The original 50,368-entry vocabulary is expanded to 52,412 entries using fresh DNA/protein BPE vocabularies fitted only on admitted CPT training data. Each modality adds 1,022 distinct model IDs. Biological encoding uses explicit piece-to-ID maps; natural-language IDs remain unchanged. New rows start at the mean of the original embeddings of their raw text fragments. All original IDs and rows are preserved at initialization.

CPT runs 128 embedding/MLM-head warmup updates followed by 1,024 full-encoder updates, with effective batch 32 and a 14 DNA / 14 protein / 4 text mixture. Warmup masks old-row gradients and gives the embedding parameter group zero AdamW weight decay with fresh optimizer state. **Old rows remain exactly invariant throughout warmup.** Full adaptation removes the mask, unfreezes the encoder, and starts the declared full-CPT optimizer.

The starting checkpoint does not include an upstream MLM prediction head, so this head is freshly initialized and trained. MLM loss changes include learning in that head and are not alone evidence of downstream usefulness or preserved language performance.

| Fixed validation checkpoint | DNA NLL | Protein NLL | Text NLL |
|---|---:|---:|---:|
| Initial random MLM head | 14.4105 | 14.3941 | 12.8057 |
| Warmup end, step 128 | 6.0647 | 6.0184 | 6.0736 |
| Full CPT end, step 1,152 | 5.5070 | 5.4390 | 2.2541 |

Of 2,044 added tokens, 2,040 occur naturally before corruption during the actual run and 2,037 appear as masked prediction targets. Median natural occurrence counts are 1,215.5 for DNA rows and 1,114 for protein rows; median masked-target counts are 181 and 169. The standalone added token DNA:N and protein:J/N/O have no natural occurrence in the encoded CPT snapshot, while rare protein:B/U/Z occur but receive no masked target in this run. These rows are not called fully trained merely because output-softmax gradients can update them. An independent replay audit will also distinguish natural inputs visible after corruption from random replacement tokens.

The CPT final checkpoint reproduces logits exactly after reload. The saved continuation state contains all 173 optimizer parameter states at full-phase update 1,024, sampler positions and RNG states. Both completed SFT runs also reproduce predictions exactly after reload. Their 4K model files are then removed under the predeclared storage policy, while every prediction remains available.

## Data and scope

The admitted CPT snapshot has 73,728 training sequences and 17,002,452 input tokens, but this first run processes 36,864 sequence presentations rather than the entire snapshot. Raw corpus sampling is length-biased random-byte sampling. Whole biological lines sharing a 31-mer with protected DNA sequences (both strands), or a 15-mer with protected protein sequences, are excluded; analogous guards separate biological CPT train and validation samples. Held-out sequences are used only for exclusion, not their targets. These guards do not prove global homology or parental independence.

The SFT sets are 4,096 balanced examples and the complete 16,766-example training split. All promoter inputs fit in full, with maximum length 93 model tokens and zero truncation. The 4K subset is nested in full train. Because both sizes receive three epochs, larger-data comparisons also receive more optimizer updates; differences across sizes do not isolate labeled-data quantity from optimization budget.

The present evidence supports continued investigation of this traditional adaptation route on promoter classification. It does not yet establish the full-data effect, multi-seed robustness, another task, few-shot performance, or zero-shot classification. A fresh fixed classifier is not a semantic zero-shot interface.

## Artifacts

- [Completed-run evidence archive](../artifacts/laya_biocpt_v1/README.md), including compressed exact predictions, traces, vocabulary metadata and recipe hashes.
- [Paired 4K calculation](../artifacts/laya_biocpt_v1/round/paired_4096_complete.json).
- [CPT continuation-state audit](../artifacts/laya_biocpt_v1/resume_state_audit.json).
- [Active plan](../refine-logs/EXPERIMENT_PLAN.md) and [pause record for the previous route](../refine-logs/ROUTE_CHANGE_20260924.md).

Local complete artifacts: `/root/autodl-tmp/jev_gene/artifacts/laya_biocpt_v1`. Raw corpora, encoded datasets and large model/optimizer files are retained locally and are not included in the Git archive. The protocol/trainer was frozen and pushed before downstream results at commit `b7add363b8b54929ec13bf3f50fc4da213f091f1`.
