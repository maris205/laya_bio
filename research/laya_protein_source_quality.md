# Historical protein corpus composition defect and corrective action

Detected: 2026-09-24, during the first biological CPT adaptation experiment.

## Verified finding

A complete scan of the local historical `data/01_raw_cpt/protein_uni_16.txt` file found **zero uppercase N and zero lowercase n**:

- Bytes scanned: 16,955,660,631.
- Newline-delimited records: 16,466,361.
- Whole-file SHA-256: `fc8de807efdfc94b51d54bfd73a66403b76c8a49bd0b056184356736bc3d5876`.
- Scan completed: 2026-09-24T15:27:25.038148+00:00.

This SHA-256 and byte count exactly match the source entry in the existing local historical-corpus release manifest. The archived file is a faithful copy of the historical source; byte-level integrity does not establish biological completeness.

`N` encodes asparagine in the standard amino-acid alphabet. It is a canonical protein residue, unlike the ambiguous-base meaning of N in DNA. See the [NCBI amino-acid code table](https://www.ncbi.nlm.nih.gov/books/NBK44863/table/sequencesquickstart.Te/). A broad protein corpus of this size without any N cannot be admitted as an ordinary complete-amino-acid source.

The upstream processing that produced this file is unknown. We do **not** assert that the current preprocessing deleted N: the current sampler only uppercases and slices sequences, and independent raw-byte scans show N was absent before it ran. The source is left unchanged to preserve provenance, and rejected for continued broad protein CPT.

## Correction in the active experiment

The corrected experiment uses the local historical `protein_lucaone_15g.txt`. Its admitted training sample contains all 20 canonical amino acids, including **459,623 N characters**. A new hard composition gate executes before vocabulary fitting/model training and rejects the old source in an end-to-end regression check with the exact missing-residue list `['N']`. The production controller independently requires a passing composition audit.

This gate establishes canonical-residue presence in the admitted sample; it does not certify all upstream preprocessing, organism balance, homology independence, or provenance. Existing long-kmer held-out exclusion guards remain in place.

DNA BPE and the 4,096/full/development promoter input files retain identical hashes. Protein samples and BPE are regenerated from the replacement corpus. All training seeds, budgets, optimizer schedules, classifier construction and endpoint rules remain unchanged. The complete four-fit SFT matrix is repeated after corrected CPT.

## Treatment of already observed results

The old-source CPT and its 4K SFT pair are preserved as a [superseded diagnostic](laya_biocpt_source_diagnostic.md). Its +5.23 pp promoter development difference is not used as the primary corrected-source conclusion. The partial full-data no-CPT fit was stopped and the old-source full-data CPT fit was never started. The [corrected round](laya_biocpt_round1.md) provides the active result record.

This discovery does not change already measured numerical predictions, and it does not by itself establish that every previous model result is invalid. It does prohibit treating this particular historical protein corpus as a clean, complete canonical-amino-acid source. Historical tokenizers fitted on it need their own coverage assessment before reuse; the active experiment fits fresh protein BPE on the replacement source.

## Evidence

- [Whole-file N scan](../artifacts/laya_biocpt_v1/protein_source_N_audit.json).
- [Corrected residue-composition audit](../artifacts/laya_biocpt_v2/data/residue_composition_audit.json).
- [Old-source rejection test](../artifacts/laya_biocpt_v2/negative_admission_test.json).
- [Unchanged DNA/SFT inputs](../artifacts/laya_biocpt_v2/preserved_promoter_inputs_check.json).

Large original and corrected raw snapshots and the scan implementation remain local in the experiment workspace. No historical corpus bytes were silently repaired or replaced.
