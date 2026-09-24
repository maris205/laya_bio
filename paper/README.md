# Jev-Style Decision Models for Biological Sequences

This is a working research manuscript based on the completed three-seed experiments, not a venue submission or a claim of external peer review.

- `main.pdf`: English paper, main body, references, supplementary tables, and a clearly marked prospective experiment design.
- `main.tex`, `sections/`, `references.bib`: editable LaTeX and verified references.
- `figures/`: two vector diagnostic figures, PNG previews and reproducible generators.
- `tables/`: nine tables generated from experimental JSON, with no model reruns.
- `table_provenance.json`: result and test-manifest fingerprints used for the tables.
- `citation_sources/`: source metadata acquired from official proceedings, Crossref, arXiv and project/dataset documentation.
- `REVIEW.md`: local evidence and presentation review, including unresolved scientific limits.

Rebuild from the project root with `python paper/build.py`. Requires Python with matplotlib, NumPy and the completed local artifacts, plus `pdflatex` and `bibtex`. The build does not train, access a model checkpoint, or launch test inference.

The Chinese result manuscript is `research/laya_bio_paper_draft.md`. The 24-view task catalog is [here](../research/biological_decision_task_catalog.md), with 12 priority tasks covering Choice/Score/Noul and explicit source/admission status. Appendix B describes these as prospective work; existing released weights only establish the two-task Choice result.

The empirical main comparison is raw candidate / biological-BPE candidate / implemented B1 / trained text-only. Early wrapped-input and compact-vocabulary development runs are not pooled with the final test results. No claims of general superiority, statistical significance, or homology independence are made.

## Public data, code, weights, and historical corpora

The paper's reproducibility appendix records the following public releases and fixed revisions:

| Release | Contents | Fixed revision |
|---|---|---|
| [dnagpt/laya-bio](https://huggingface.co/datasets/dnagpt/laya-bio) | Original task snapshots, cleaned and paper-eligible splits, predictions/logits, manifests, code, BPE assets, and manuscript materials | [`0307d5910b0a37f54caecd444241db6b56331701`](https://huggingface.co/datasets/dnagpt/laya-bio/tree/0307d5910b0a37f54caecd444241db6b56331701) |
| [dnagpt/laya-bio-models](https://huggingface.co/dnagpt/laya-bio-models) | All 12 final checkpoints and original initialization, with tokenizers, configurations, and loading code (22.46 GB) | [`6d727dd4125d783ab719a35c6b55dd7a69276a8d`](https://huggingface.co/dnagpt/laya-bio-models/tree/6d727dd4125d783ab719a35c6b55dd7a69276a8d) |
| [dnagpt/laya-bio-historical-corpora](https://huggingface.co/datasets/dnagpt/laya-bio-historical-corpora) | Seven historical corpus files and two actual BPE-training samples; lossless shards, hashes, and restoration script (42.55 GB; 106.98 GB restored) | [`bd70a7d157741f8d19ad150adfa350a9b02ba417`](https://huggingface.co/datasets/dnagpt/laya-bio-historical-corpora/tree/bd70a7d157741f8d19ad150adfa350a9b02ba417) |

Use the default dataset configuration for the paper's exact subset: 32,050 train / 1,978 selection-dev / 1,986 calibration / 4,097 test examples. The separate `cleaned_all` configuration includes examples outside that subset. Historical corpus availability does not imply additional neural continual pretraining in this study. Source and component-specific license terms apply; homology independence and absence of pretraining overlap are not established.

The fixed benchmark release contains the manuscript as it stood at first publication of the artifacts; this working source and PDF include the subsequent availability update. Download and restoration examples and verification details are in [the release record](../research/laya_large_data_release.md).

## Decision-interface framing, 2026-09-24

The central question is now whether a shared biological candidate-decision interface combines task reuse with explicit label identity. BPE is secondary. The title, abstract, introduction, method contract, evidence organization, and conclusion follow this question. Appendix B and `../refine-logs/EXPERIMENT_PLAN.md` describe unrun comparisons with free/constrained generation and same-modality task/label transfer; all existing experimental numbers are retained. See `../research/jev_style_reframing.md` for the claim map. The fixed Hugging Face snapshots retain the earlier title and manuscript; the GitHub working paper reflects this revision.
