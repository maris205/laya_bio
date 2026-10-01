# Benchmark v2 公开数据集收集记录

收集日期：2026-10-01。目标：为 MVP2 统一决策评测基准收集公开生物序列数据集，覆盖分类（Choice）、有序回归（Score）、命题判断（Noul）三类接口，DNA / 蛋白 / RNA 三模态，单序列与多序列。本轮只做**来源收集与完整性核验**，不做划分、准入或训练。

数据落地目录：`data/05_benchmark_v2_raw/`（本轮新下载）与 `data/05_task_extension_sources/`（此前已下载）。下载凭证与逐文件 SHA-256 见 [`download_receipt.json`](../data/05_benchmark_v2_raw/download_receipt.json)。所有 HF 快照固定到具体 revision（commit SHA），GitHub 仓库固定到 commit。

## 一、本轮新下载

代理：clash `http://127.0.0.1:7890`（autodl `network_turbo` 亦可）。下载器 `data/05_benchmark_v2_raw/hf_snapshot_download.py` 用 `snapshot_download` 固定 revision，逐文件记录字节数与 SHA-256。

| 来源 | repo / URL | revision / sha | 模态 | 接口 | 任务粒度 | 大小 |
|---|---|---|---|---|---|---|
| GUE | `dnagpt/GUE` | `e0bc5eec01` | DNA | Choice | 28 个叶子任务（10 组蛋白标记 + mouse×5 + prom×6 + splice + tf×5 + virus） | 310.0 MB / 263 文件 |
| Genomic Benchmarks | `katarinagresova/Genomic_Benchmarks_*`（8 子集） | 各子集独立 sha | DNA | Noul | 8 子集：human_enhancers_cohn、demo_coding_vs_intergenomic、human_nontata_promoters、drosophila_enhancers_stark、demo_human_or_worm、dummy_mouse_enhancers、human_ocr_ensembl、human_ensembl_regulatory | 合计 ~121 MB |
| ProteinGym v1 | `OATML-Markslab/ProteinGym_v1` | `1ea2aa10a3` | 蛋白 | Score | DMS_substitutions **217 个 assay**、2,465,767 行；另含 DMS_indels / clinical_* | 135.9 MB / 32 文件 |
| remote_homology | `proteinea/remote_homology` | `41156f4e81` | 蛋白 | Choice | 1（TAPE 远缘同源 fold 分类） | 55.8 MB / 20 文件 |
| DeepSTARR | `GenerTeam/DeepSTARR-enhancer-activity` | `8cddef6d` | DNA | Score | 1（发育型 + housekeeping 增强子活性，两个连续输出） | 57.6 MB / 17 文件 |
| TAPE stability | `s3://songlabdata/.../stability.tar.gz` | sha256 `4287f9fe…335f6` | 蛋白 | Score | 1（train 53614 / valid 2512 / test 12851） | 1.8 MB |
| RNAcompete | github `morrislab/RNAcompete` | `2a0d8f80` | RNA | Score | 含真实数据（probe_intensity / kmer_zscore / feature.tsv） | 177 MB |

完整性核验：GUE 每叶子任务含 train/dev/test.csv；Genomic Benchmarks 8 子集 parquet 均可加载且行数正常（如 human_ensembl_regulatory train 231,348 / test 57,713）；ProteinGym 217 assay 计数来自 `DMS_id` 唯一值；TAPE stability 三 split JSON 记录数与官方一致。

## 二、此前已下载（本地在库）

`data/05_task_extension_sources/` 与 LLaMA-Gene 转换快照，对应任务目录 C01–C11 / E01–E04：

| 目录 ID | 数据 | 模态 | 接口 |
|---|---|---|---|
| C01 | dna_promoter_300（59,195 行） | DNA | Noul |
| E01 | dna_core_promoter（59,196 行） | DNA | Noul |
| C02 | dna_splice_site_prediction（45,619 行，3 类） | DNA | Choice |
| C03 | dna_transcription_factor_prediction（34,378 行） | DNA | Noul |
| C08/E03/E04 | gene_lan_transfer（8 配置：dna_protein_pair / rand / rand_v2 + dna_sim + protein_sim） | DNA+蛋白 | Noul |
| C09 | TAPE fluorescence（已用于主实验） | 蛋白 | Score |
| C11 | DeepLoc 2.0（28,303 train+valid，10 标签） | 蛋白 | 多 Noul |
| C04 | lg_fold_class（7 类结构） | 蛋白 | Choice |
| C05 | lg_signal_peptide（Sec/Tat） | 蛋白 | Choice |
| C06 | lg_subcellular_loc（6 类原核定位） | 蛋白 | Choice |
| E02 | lg_npp（神经肽前体） | 蛋白 | Noul |
| C07 | protein_homology_std / remote（双蛋白） | 蛋白 | Noul |

## 三、待补（本轮下载受阻）

| 目录 ID | 来源 | 受阻原因 | 后续 |
|---|---|---|---|
| C12 | DeepGOZero（GO 分子功能多标签） | KAUST 数据服务器 `deepgo.cbrc.kaust.edu.sa/data/deepgozero/` 返回 404；仓库仅含代码 | 另找镜像或按 SwissProt+GO 自行构建 |
| E11 | PEER / FLIP（蛋白互作） | FLIP 为 621 MB git-LFS 仓库，raw 链接不通；PEER 数据需运行其下载管线 | 按需 git-lfs 拉取或跑 PEER 下载脚本 |
| E07 | DeepSEA（919 染色质特征） | 大文件，本轮未取 | 按需获取 |

## 四、规模结论

仅 GUE(28) + Genomic Benchmarks(8) + ProteinGym(217) 三者细粒度任务已 >253，叠加本地 12 个目录任务，**远超用户设定的 ~100 题目标**。三模态（DNA / 蛋白 / RNA）、三接口（Choice / Score / Noul）、单序列与多序列（gene_lan_transfer 的 DNA-蛋白配对、protein_homology 双蛋白）均已覆盖。磁盘占用：本轮新下载约 0.66 GB（HF 快照）+ 0.21 GB（GitHub clones），当前 `/root/autodl-tmp` 余 110 GB。

下一步（未执行）：benchmark v2 规格（分类法、防泄漏划分、准入规则、任务清单）与统一 `{sequences, question, primitive, candidates_or_levels}` 转换器。多模态图像题（小规模 Qwen/Gemma VLM）为更后续阶段。
