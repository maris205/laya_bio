# Biological vocabulary → CPT → conventional SFT: corrected first round

**Status: CPT and the corrected 4K pair complete; full-data pair running (2026-09-24).** The primary results below use the corrected protein source and the frozen [v2 plan](../artifacts/laya_biocpt_v2/round/EXPERIMENT_PLAN.md). The [superseded old-source diagnostic](laya_biocpt_source_diagnostic.md) is retained separately.

## Fixed-final-step comparison

Promoter detection uses the existing 1,052-example selection-development split. Each arm starts from the same original encoder and expanded vocabulary, with identical fresh mean-pooling/LayerNorm/dropout/linear classifier initialization, data/order, optimizer schedule, and three SFT epochs. CPT includes 128 new-embedding/MLM-head warmup updates followed by 1,024 full-encoder MLM updates. No test inference is performed.

| Training examples | Arm | SFT updates | Train accuracy | Dev accuracy | Dev macro-F1 | Dev NLL |
|---:|---|---:|---:|---:|---:|---:|
| 4,096 | No additional CPT | 192 | 83.40% | 82.51% | 0.8244 | 0.3873 |
| 4,096 | Corrected-source CPT | 192 | 90.63% | 87.55% | 0.8755 | 0.3033 |
| 16,766 | No additional CPT | 786 planned | Pending | Pending | Pending | Pending |
| 16,766 | Corrected-source CPT | 786 planned | Pending | Pending | Pending | Pending |

The completed 4K effect is **+5.04 accuracy percentage points**. A 10,000-replicate paired group bootstrap gives a descriptive 95% interval of **[+3.14, +7.03] pp**. This is one training seed on a historically used development set; the interval does not measure training-seed variability or constitute independent test confirmation.

Class-0/class-1 recall changes from 89.44%/75.71% without CPT to 89.64%/85.50% with CPT. The corrected CPT development accuracy is 66.63%, 87.26%, and 87.55% after epochs one, two, and three. CPT trails the no-CPT arm at epoch one, then leads at epochs two and three; the corresponding no-CPT accuracies are 75.76%, 77.09%, and 82.51%. The conclusion uses the predeclared final endpoint, not a selectively chosen intermediate checkpoint.

The no-CPT 4K repeat matches all 192 old-source-run training losses/gradient traces and all six recorded train/dev prediction files exactly. DNA BPE and all promoter SFT files are unchanged by the protein-source correction. This is a same-seed reproducibility check, not an additional independent seed.

## Protein-source correction and embedding training

A whole-file scan of historical `protein_uni_16.txt` found zero N/n characters in 16,955,660,631 bytes. Since N is a canonical protein residue, that source is rejected for broad protein CPT. The cause of the upstream defect is unknown; the current preprocessing does not delete N. Its old-source 4K results remain visible as a superseded diagnostic. See the [full source-quality record](laya_protein_source_quality.md).

The replacement historical LucaOne protein source passes a new hard canonical-residue gate. The admitted training snapshot contains all 20 canonical amino acids, including 459,623 N characters. The old source fails an end-to-end negative admission test. This composition check does not certify all upstream processing or global homology independence.

Fresh DNA/protein BPE vocabularies are fitted on admitted CPT training samples only. The base vocabulary of 50,368 entries expands to 52,412, adding 1,022 rows per biological modality. New rows are initialized from the mean of their original raw-fragment embeddings; original IDs and rows are preserved. Warmup freezes encoder matrices and old embedding rows, masks old-row gradients, and gives the tied embedding parameter group zero AdamW weight decay with fresh optimizer state. Old embedding rows remain exactly invariant throughout warmup, and the encoder receives gradients after unfreezing.

In corrected CPT, 2,040 added tokens occur naturally before masking and 2,039 occur as masked prediction targets. Protein N alone has **27,991 natural input occurrences, 4,196 masked targets, and 0.32335 L2 embedding change**. DNA:N and protein:J/O/Z have no natural occurrence in the actual scheduled run (Z occurs in the larger snapshot but is not sampled); protein:B occurs without a masked target. Output-layer gradients are not treated as proof of natural input exposure. The final independent replay will also count natural inputs remaining visible after corruption.

## MLM validation and saved state

The starting checkpoint lacks an upstream MLM prediction head, so a fresh head is trained. Loss reductions include this head's learning and do not alone establish biological usefulness, language retention, or zero-shot classification.

| Checkpoint | DNA NLL | Protein NLL | Text NLL |
|---|---:|---:|---:|
| Corrected full CPT end, update 1,152 | 5.4975 | 5.4491 | 2.1957 |

The final CPT checkpoint reproduces its logits exactly after reload. The retained continuation state includes 173 optimizer parameter states at full-phase update 1,024, all RNG states, sampler order and positions. Both completed SFT fits pass exact prediction reload checks. Under the frozen storage policy, 4K model files are removed only after verification; all predictions remain. The full-data SFT checkpoints will be retained.

## Scope and pending checks

CPT has 73,728 admitted training sequences and 13,611,857 snapshot input tokens, of which 36,864 sequence presentations are scheduled. Sampling is length-biased random-byte sampling from local historical corpora. Whole biological lines sharing 31-mers (DNA, both strands) or 15-mers (protein) with protected downstream sequences are excluded, with analogous biological CPT train/validation guards. This does not guarantee global homology/parent independence. New tasks require their own admission checks.

Promoter inputs fit in full (maximum 93 model tokens, zero truncation). The 4K set is nested in the 16,766-example full set. Three epochs at each size mean different update budgets, so cross-size differences do not isolate label quantity from optimization. This experiment uses a conventional fixed classification head; it does not evaluate the old semantic candidate interface, a withheld task, few-shot learning, or zero-shot learning.

The full-data pair, independent metric/count/RNG replay audit, and final learning-curve figures remain pending. Raw sequences, complete encoded data, and large weights/optimizer states remain locally at `/root/autodl-tmp/jev_gene/artifacts/laya_biocpt_v2`. The [evidence archive](../artifacts/laya_biocpt_v2/README.md) contains exact compressed predictions, traces, vocabularies and hashes. The source-correction protocol was frozen and published at commit `e8d28fc053ae990b5a8c8ee3d6a36fc74bdfe375` before any corrected downstream results.
