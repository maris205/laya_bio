# Laya-Bio

**Candidate Scoring and Reliability on Short Biological Sequences**

Laya-Bio studies adaptation of the open Laya typed-decision encoder to DNA promoter detection and seven-class coarse protein structure classification. The study compares raw candidate scoring, historical biological-BPE candidate scoring, a fixed-class B1 head, and a trained text-only control across three seeds. Raw candidate input gives higher test accuracy on both tasks; biological BPE reduces input length without improving measured training time. No additional neural continual pretraining was performed.

- [English paper (PDF)](paper/main.pdf) · [LaTeX source](paper/main.tex)
- [Chinese manuscript](research/laya_bio_paper_draft.md)
- [Release and verification record](research/laya_large_data_release.md)

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

The small result summaries required to rebuild the paper are included in `artifacts/`. Run `python paper/build.py` from the project root with the dependencies listed in [paper/README.md](paper/README.md). This regenerates figures, tables, and the PDF without model training or test inference. The fixed Hugging Face benchmark release preserves its original manuscript snapshot; the working manuscript includes the later public-availability update.

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
