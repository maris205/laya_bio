# Local manuscript review

Reviewed 2026-09-23 against completed experiment artifacts and the rendered nine-page PDF. This is a local evidence/presentation review, not an external reviewer score or peer review.

## Changes from the pre-experiment draft

1. Replaced planned B0/compact-M1/M2 main comparisons with the actual four-condition, three-seed fixed test evaluation. Earlier frozen and compact-vocabulary experiments remain development history.
2. Corrected the vocabulary description to externally fitted historical BPE with alphabet repair and direct ID mapping. Removed the train-only vocabulary claim from the final method.
3. Described the implemented CLS B1 accurately, including new class-output layers, omitted candidates, and the resulting limits on causal attribution to candidate semantics.
4. Separated fixed test metrics from dev sequence/order interventions. Shuffled-sequence scores are labeled original-label matching, not verified biological accuracy.
5. Removed unperformed paraphrase, inference-latency, unseen-task and public-preregistration claims. Documented the inherited eligibility filter and iterative development history.
6. Kept the negative BPE test result and lack of measured training acceleration alongside the favorable B1 comparison. Reported all seeds and the large B1 protein variability.
7. Added the distinction between low ECE and good classification: the weak text-only predictor can have low ECE.

## Evidence checks

- Main test table regenerated from `artifacts/laya_locked_test/results.json`; every accuracy/F1 mean and SD matches the machine-readable result.
- Additional tables read original summaries, diagnostic JSON, and the frozen eligibility list. Table-source hashes are recorded in `table_provenance.json`.
- B1/candidate parameter counts, deterministic training permutations, optimizer schedule, gradient clipping, and frozen action head checked against the executed code.
- DNA/protein split sizes and inherited exclusion counts checked against formal-data and input-audit reports.
- All nine cited entries have verified source metadata: official ACL/PMLR/ICLR records, Crossref, arXiv, or explicit software/dataset documentation. BioPAWS-2 is cited as a dataset card, not an invented journal article. The local data snapshot is identified by hashes.
- Reverse-outline check: interface question → related sequence/text approaches → exact candidate/BPE/control definitions → fixed evaluation → positive control comparison and negative representation result → dependence/order/calibration diagnoses → restricted conclusion.

## Build and visual checks

`python paper/build.py` regenerates tables and figures and builds with `pdflatex`, `bibtex`, then two final LaTeX passes. It does not launch training or inference.

- 6 main-body pages, 1 reference page, 2 appendix pages.
- 2 vector figures, 9 generated tables, 9 cited references.
- 0 missing references/citations, 0 overfull/underfull boxes, 0 final LaTeX/BibTeX warnings.
- All fonts embedded; all nine pages visually inspected. Hash lines now wrap; candidate-order legend and figure labels are readable; the sequence-removal SD is displayed without clipping its lower endpoint.
- The initial version used an anonymous author placeholder. The author update now uses the user-supplied Liang Wang affiliation and superscript markers; no email or marker explanation was supplied or invented. No remaining result placeholders.
- Machine-readable build checks and PDF SHA-256: `validation_report.json`.

## Remaining scientific limits

The manuscript is suitable for reviewing the completed study, not yet a claim of submission readiness. Outstanding limits include unverified homology separation and historical-corpus overlap, a single task template per task, one fixed-head architecture, only three seeds, no matched specialist-model comparison, and candidate-order sensitivity. These are stated in the manuscript, not concealed by the completed build. The test set has been used and must not become a selection set for subsequent methods.

The [paper-write skill](/root/autodl-tmp/.codex/skills/paper-write/SKILL.md) specifies: “Send the complete draft to GPT-5.4 xhigh.” Its designated Codex MCP reviewer was not available in this session. That external review was not performed or replaced with a fabricated score. The current record covers local source, numerical, citation and visual checks only.

The six-page research-note scope follows the existing project plan rather than the skill's default ICLR template. No venue-specific submission or publication has been attempted.

## Public availability update, 2026-09-24

Updated the English reproducibility appendix, conclusion, Chinese manuscript, and README documentation to describe the three public Hugging Face repositories, their fixed revisions, checkpoint scope, and lossless corpus restoration. Removed obsolete local-only availability statements. The experimental results and evaluation protocol are unchanged. Rebuilt with the existing `paper/build.py` multi-pass pdflatex/BibTeX workflow. Validation passed: 6 main-body pages, 1 reference page, 3 appendix pages; no undefined references/citations, LaTeX warnings, overfull boxes, or BibTeX warnings. All fonts are embedded. The three pinned public links were verified as PDF link annotations, and the availability and supplementary-table pages were visually checked. The fixed Hugging Face benchmark snapshot retains the earlier manuscript; the working PDF contains this subsequent documentation update.
