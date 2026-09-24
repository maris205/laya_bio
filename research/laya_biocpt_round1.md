# Biological vocabulary → CPT → conventional SFT: corrected first round

**Status: corrected source admitted; complete production round running.** The first protein source failed a canonical-residue composition audit: a complete scan of `protein_uni_16.txt` found zero N characters. The [superseded diagnostic](laya_biocpt_source_diagnostic.md) retains its 4K comparison transparently. It is not the primary corrected result.

The corrected run uses `protein_lucaone_15g.txt`, whose admitted training sample contains all 20 canonical amino acids, including 459,623 N characters. A new hard admission gate rejects the previous source in an end-to-end negative test. DNA BPE and all promoter SFT input hashes are unchanged. All training hyperparameters and the full four-fit comparison are repeated under the [active plan](../refine-logs/EXPERIMENT_PLAN.md).

No corrected downstream result is available yet. This file will receive the full fixed-step table, independent metrics audit and plots after completion.
