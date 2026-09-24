# Laya-Bio 权重与历史语料公开发布记录

完成时间：2026-09-23T18:02:28Z。三个仓库均为公开仓库，可匿名访问。

| 内容 | 公开仓库 | 本次发布文件 | 大小 |
|---|---|---:|---:|
| 模型权重、配置、分词器和加载代码 | [dnagpt/laya-bio-models](https://huggingface.co/dnagpt/laya-bio-models) | 279 | 22,459,114,888 bytes（22.46 GB） |
| 历史语料、BPE 采样文件、词表与恢复脚本 | [dnagpt/laya-bio-historical-corpora](https://huggingface.co/datasets/dnagpt/laya-bio-historical-corpora) | 420 | 42,547,838,540 bytes（42.55 GB） |
| 论文数据、预测、代码和论文材料 | [dnagpt/laya-bio](https://huggingface.co/datasets/dnagpt/laya-bio) | 248 | 约 162 MB |

文件数不含各仓库自动生成的 `.gitattributes`。原数据集主页现已链接两个补充仓库及其固定版本。

## 固定版本

- 模型：[`6d727dd4125d783ab719a35c6b55dd7a69276a8d`](https://huggingface.co/dnagpt/laya-bio-models/tree/6d727dd4125d783ab719a35c6b55dd7a69276a8d)。
- 历史语料：[`bd70a7d157741f8d19ad150adfa350a9b02ba417`](https://huggingface.co/datasets/dnagpt/laya-bio-historical-corpora/tree/bd70a7d157741f8d19ad150adfa350a9b02ba417)。
- 数据集主页及发布清单更新：[`0307d5910b0a37f54caecd444241db6b56331701`](https://huggingface.co/datasets/dnagpt/laya-bio/tree/0307d5910b0a37f54caecd444241db6b56331701)。
- 首次数据发布版本 `3ff01111b721c14a07c541dcdb719523db575c09` 保留；此次更新仅修改 README、发布清单及 SHA256SUMS，数据划分与预测结果未改动。

## 权重范围

包含论文实际测试的 12 份最终权重：raw、full_bpe、b1、text_only，各 3 个固定种子（20260922、20260923、20260924），另附原始 Laya 初始化权重。每个最终 checkpoint 的全部文件均逐一匹配正式测试冻结清单中的哈希，权重保留原始 Safetensors 字节和精度，未重新训练、转换或量化。分词器、完整 BPE 表示、encoder 配置、训练摘要及校准参数随权重提供。

`checkpoint_index.json` 提供各模型路径、哈希、张量元数据和温度参数。这是自定义 Laya 模型集合；加载方式见模型卡及随附脚本。早期 smoke、pilot 和被替代实验的中间权重不属于论文最终模型集合。

## 历史语料范围

包含全部 7 份本地原始历史语料及实际拟合继承 BPE 的 2 份采样文件。原始数据总计 **106,976,664,607 bytes（106.98 GB）**，无损压缩为 404 个独立 Zstandard 分片；压缩数据本体为 42,508,143,679 bytes，另附词表和说明材料。

| 原始文件 | 原始字节 | 压缩字节 | 分片数 |
|---|---:|---:|---:|
| `data/01_raw_cpt/dna_32g.txt` | 32,420,152,883 | 9,675,125,354 | 121 |
| `data/01_raw_cpt/openwebtext.txt` | 39,587,458,433 | 14,866,321,712 | 148 |
| `data/01_raw_cpt/pdb_3di.fasta` | 103,863,984 | 34,455,671 | 1 |
| `data/01_raw_cpt/pdb_aa.fasta` | 103,863,984 | 26,118,767 | 1 |
| `data/01_raw_cpt/protein_lucaone_15g.txt` | 15,388,772,546 | 8,264,977,230 | 58 |
| `data/01_raw_cpt/protein_uni_16.txt` | 16,955,660,631 | 8,725,364,344 | 64 |
| `data/01_raw_cpt/ss.txt` | 275,671,162 | 36,985,538 | 2 |
| `bpe_training_samples/dna_1g.txt` | 1,067,478,842 | 318,603,490 | 4 |
| `bpe_training_samples/protein_1g.txt` | 1,073,742,142 | 560,191,573 | 5 |

每个分片最多对应原始文件连续 256 MiB 的字节。分片边界可以跨越行、FASTA 记录或 UTF-8 字符，恢复完整文件后再解析。`corpus_manifest.json` 记录分片顺序、偏移、压缩哈希、解压哈希和原始文件总哈希。两份 BPE 采样文件与既有历史语料清单的 SHA-256 一致。

## 下载与恢复

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "dnagpt/laya-bio-models",
    revision="6d727dd4125d783ab719a35c6b55dd7a69276a8d",
    local_dir="laya_models",
)
snapshot_download(
    "dnagpt/laya-bio-historical-corpora",
    repo_type="dataset",
    revision="bd70a7d157741f8d19ad150adfa350a9b02ba417",
    local_dir="corpus_archive",
)
```

```bash
python corpus_archive/restore_corpora.py --archive corpus_archive --output restored_corpora
# 只恢复一个原始文件：
python corpus_archive/restore_corpora.py --archive corpus_archive --output restored_corpora --source data/01_raw_cpt/dna_32g.txt
```

恢复脚本依赖 `zstandard`，逐个核验压缩分片、解压字节及最终原始文件的 SHA-256；不同内容的已有输出不会被覆盖。下载单个模型可按模型卡的 `allow_patterns` 示例缩小范围。

## 已完成核验

1. 模型包 279 个文件、语料包 420 个文件的远端大小与内容哈希全部匹配本地封存包。LFS 文件核对 SHA-256，普通 Git 文件核对 Git blob SHA-1。
2. 两个新仓库的公开状态及文件元数据均经匿名访问验证。
3. 实际匿名读取全部 13 份权重的前 65,536 字节，逐份与本地一致，Safetensors 头部可解析。此步骤是公开文件访问检查，不是重新运行模型评测。
4. 全部 404 个压缩分片在发布前均做过完整逐字节解压回读检查。
5. 从公开固定版本匿名下载 `data/01_raw_cpt/pdb_3di.fasta` 的压缩分片，使用发布恢复脚本还原 103,863,984 字节，完整 SHA-256 为 `2c8db7ea908e96e96f6a8097f152ef873c6f222acc87d750a801ae96749e137f`，与原文件完全一致。
6. 数据集主页、更新后的发布清单及 SHA256SUMS 均已匿名下载并核验内容。

机器可读凭据位于 `artifacts/laya_hf_release/`：`laya-bio-models_receipt.json`、`laya-bio-historical-corpora_receipt.json`、`model_public_loading_check.json`、`large_public_loading_check.json` 和 `benchmark_companion_links_receipt.json`。

## 来源边界

仓库保留来源记录及各组件许可说明，不将历史语料统一声称为 Apache-2.0。历史来源的精确上游版本、全库同源性及评测重叠仍有未完整确认之处。发布这些语料不改变论文中“本轮没有额外神经持续预训练”的实验事实。令牌未写入发布包或登录缓存。
