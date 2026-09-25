# Laya-Bio

**Jev-Style Decision Models for Biological Sequences**

Laya-Bio is an open biological sequence decision model: **sequence + natural-language question + candidate labels → candidate probabilities and an exact label selection**. One shared scorer handles the trained DNA and protein tasks without task-specific output matrices. “Jev-style” describes the typed candidate interface, inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev); this project adapts the open Laya encoder.

The paper now centers on a shared biological decision interface, with BPE as a secondary representation comparison. Candidate selection keeps the output inside the supplied label set; it does not guarantee the correct biological class, order invariance, or unseen-task understanding. Shared fixed heads and constrained generation can also enforce valid outputs.

**Active route (2026-09-25):** One model, multiple biological tasks, unified JEV-style outputs, with task quality and inference cost as the decision criteria. The bounded five-run Laya matrix and independent audit are complete: the NO-CPT joint model reaches promoter Accuracy 88.12%, structure Accuracy 55.48%, and GFP RMSE/MAE 0.6993/0.4921 with one shared scorer and no task-specific output heads. It passes all three frozen usefulness gates and is the selected checkpoint from this round. CPT joint slightly raises classification Accuracy but worsens GFP RMSE/MAE/Spearman, so CPT is not a universal gain. Both portable joint exports pass homogeneous and mixed-task raw-input checks; figures and the compact evidence archive are complete. No new test inference or alternative-backbone training was performed. Few-shot/zero-shot are deferred. See the [completed plan](refine-logs/EXPERIMENT_PLAN.md), [run tracker](refine-logs/EXPERIMENT_TRACKER.md), and [result record](research/laya_jev_multitask_round1.md). The old GFP microfit extension remains paused.

**Evidence status:** the completed two-task, four-condition, three-seed study is unchanged. Raw candidate scoring reaches 91.10% DNA and 59.97% protein test accuracy. The new design catalogs 12 priority tasks plus 12 extension views, mapping Choice, Score, and Noul to multiclass, ordered-score, binary, multilabel, and paired-sequence problems. It adds free/constrained generation and separate held-out-task tests. The formal follow-up suite remains pending. A separate [six-task engineering pilot](research/laya_multitask_pilot.md) now has two completed 100-update train/dev runs; the longer three-pass development round has also completed. Five matched single-task diagnostic runs have also completed; they found persistent weak fitting on splice/GFP and prompted a versioned splice class-name correction. These are not new held-out paper results. Those earlier interface experiments did not use additional neural continual pretraining; the new CPT results are reported separately above.

- [English paper (PDF)](paper/main.pdf) · [LaTeX source](paper/main.tex)
- [Six-task joint-training pilot and completed development round](research/laya_multitask_pilot.md)
- [Splice/GFP diagnostics, matched single-task controls, and splice label-name correction](research/laya_task_diagnostics.md)
- [Completed 32-example fitting checks and candidate-order augmentation](research/laya_microfit.md)
- [GFP readout/loss diagnostics: controlled 128-update runs](research/laya_readout_diagnostics.md)
- [Completed 512-update GFP convergence check: fitting criteria passed with exact prefix replay](research/laya_convergence.md)
- [Completed 1,024-example GFP development validation: near-constant predictions persist](research/laya_gfp_development.md)
- [Training-only GFP scale/LR controls: lower LR improves tiny fit; longer training partly recovers large-subset fit](research/laya_gfp_scale_probe.md)
- [Chinese manuscript](research/laya_bio_paper_draft.md)
- [Release and verification record](research/laya_large_data_release.md)

- [Biological decision task catalog: 12 priority + 12 extension views](research/biological_decision_task_catalog.md)
- [Reframing and claim boundaries](research/jev_style_reframing.md)
- [Follow-up experiment plan](refine-logs/EXPERIMENT_PLAN.md) · [Run tracker](refine-logs/EXPERIMENT_TRACKER.md)

## Public releases

The paper's reproducibility appendix records the following public releases and fixed revisions:

| Release | Contents | Fixed revision |
|---|---|---|
| [dnagpt/laya-bio](https://huggingface.co/datasets/dnagpt/laya-bio) | Original task snapshots, cleaned and paper-eligible splits, predictions/logits, manifests, code, BPE assets, and manuscript materials | [`0307d5910b0a37f54caecd444241db6b56331701`](https://huggingface.co/datasets/dnagpt/laya-bio/tree/0307d5910b0a37f54caecd444241db6b56331701) |
| [dnagpt/laya-bio-models](https://huggingface.co/dnagpt/laya-bio-models) | All 12 final checkpoints and original initialization, with tokenizers, configurations, and loading code (22.46 GB) | [`6d727dd4125d783ab719a35c6b55dd7a69276a8d`](https://huggingface.co/dnagpt/laya-bio-models/tree/6d727dd4125d783ab719a35c6b55dd7a69276a8d) |
| [dnagpt/laya-bio-historical-corpora](https://huggingface.co/datasets/dnagpt/laya-bio-historical-corpora) | Seven historical corpus files and two actual BPE-training samples; lossless shards, hashes, and restoration script (42.55 GB; 106.98 GB restored) | [`bd70a7d157741f8d19ad150adfa350a9b02ba417`](https://huggingface.co/datasets/dnagpt/laya-bio-historical-corpora/tree/bd70a7d157741f8d19ad150adfa350a9b02ba417) |

Use the default dataset configuration for the paper's exact subset: 32,050 train / 1,978 selection-dev / 1,986 calibration / 4,097 test examples. The separate `cleaned_all` configuration includes examples outside that subset. Historical corpus availability does not imply additional neural continual pretraining in this study. Source and component-specific license terms apply; homology independence and absence of pretraining overlap are not established.

## Load the paper's data

```python
from datasets import load_dataset

data = load_dataset(
    "dnagpt/laya-bio",
    revision="0307d5910b0a37f54caecd444241db6b56331701",
    token=False,
)
```

The test set has already been used for the reported results and should not be reused for model selection.

## Download weights and restore historical corpora

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "dnagpt/laya-bio-models",
    revision="6d727dd4125d783ab719a35c6b55dd7a69276a8d",
    local_dir="laya_models",
    token=False,
)
snapshot_download(
    "dnagpt/laya-bio-historical-corpora",
    repo_type="dataset",
    revision="bd70a7d157741f8d19ad150adfa350a9b02ba417",
    local_dir="corpus_archive",
    token=False,
)
```

The model repository contains custom Laya checkpoints; follow its model card and included loading code. Its `checkpoint_index.json` identifies all twelve final checkpoints and the initialization model. To download a single checkpoint, use the model card's `allow_patterns` example.

Install `zstandard` and reconstruct the original corpus files with:

```bash
python corpus_archive/restore_corpora.py --archive corpus_archive --output restored_corpora
```

The restoration script verifies compressed shards, decompressed bytes, and each complete original file using SHA-256. Restore a whole file before parsing it: shard boundaries can split lines, FASTA records, or UTF-8 characters. See the corpus card for `--source` selection.

All release files were checked against remote sizes and hashes. Anonymous access checks covered all thirteen weight files; the public PDB 3Di shard was downloaded and fully restored with its original hash. Every corpus shard also passed a complete local decompression round trip before upload.

## Rebuild the manuscript

The small result summaries required to rebuild the paper are included in `artifacts/`. Run `python paper/build.py` from the project root with the dependencies listed in [paper/README.md](paper/README.md). This regenerates figures, tables, and the PDF without model training or test inference. The fixed Hugging Face benchmark release preserves its original manuscript snapshot; the working manuscript includes the later public-availability and Jev-style framing updates.

## Repository contents and full experiment assets

This GitHub repository contains the manuscript, training/evaluation and figure scripts, vendored Laya code, and the small result summaries needed to rebuild the paper. The exact datasets, per-example predictions, BPE assets, and initialization metadata are in the pinned benchmark release above. Download them into a separate directory, then copy the required `data/` and `artifacts/` paths into your working checkout before running experiment scripts. The model card describes checkpoint paths and loading requirements.

```python
snapshot_download(
    "dnagpt/laya-bio",
    repo_type="dataset",
    revision="0307d5910b0a37f54caecd444241db6b56331701",
    local_dir="benchmark_release",
    token=False,
)
```

Component-specific licensing and provenance are described in [LICENSE.md](LICENSE.md), the vendored license, and the linked dataset/model cards.
