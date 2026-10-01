# Benchmark v2 构建协议（MVP2）

更新：2026-10-01。本文只记录**已实现的构建管道与其可复核产物**，不含任何模型结果。落地数据：`data/06_benchmark_v2_unified/`；代码：`scripts/benchmark_v2/`；数据集来源与哈希见 `data/05_benchmark_v2_raw/download_receipt.json` 与 [benchmark_v2_dataset_collection.md](benchmark_v2_dataset_collection.md)。

## 1. 统一决策接口

每条样本一行 JSON，四种 primitive 共用同一外层结构 `{task_id, primitive, modality, sequences, question, split, group, provenance}`，接口字段按 primitive 追加：

| primitive | 追加字段 | 说明 |
|---|---|---|
| `choice` | `candidates`（互斥标签列表）、`answer` | 交叉熵；评价 Accuracy / Macro-F1 / NLL / Brier / 候选合法率 |
| `noul` | `candidates=["no","yes"]`、`answer`∈{yes,no} | 单命题；评价 AUROC / MCC / F1 / Brier / 校准 |
| `score` | `score_levels`、`anchors`（train 分位数，4 切点 5 级）、`gold_value`、`gold_level` | 有序等级期望值；评价原单位 MAE/RMSE + Spearman + 量化误差 |
| `multi_noul` | `candidates`（全部标签）、`labels`（逐标签 0/1）、`answer`（正标签集） | 逐标签边际概率；评价 Micro/Macro-F1、逐标签校准 |

`sequences` 为 `[{role, modality, sequence}]`。单序列 role=`query`；双序列 role=`query_1`/`query_2`，保留边界不拼接。`group` 是用于防泄漏的实体键（本轮=序列 SHA1 前 12 位）。`score` 的 `anchors` **仅由 train 计算**，test 不反灌，避免锚点泄漏。

## 2. 划分与防泄漏隔离

管道：`run_pipeline.sh` = build → isolate → check_admission → finalize_manifest。

- **优先使用原生划分**：GUE、Genomic Benchmarks、TAPE、DeepSTARR、本地 LLaMA-Gene/BioPAWS 快照自带 train/val/test（`val`→`dev`）。GB 只有 train/test，dev 按序列哈希从 train 切 10%，test 保持官方。DeepLoc 用 `Partition` 列构造（p0=test，p1-3=train，p4=dev，已在 manifest 注明为构造划分）。
- **`isolate_splits.py` 隔离规则**：
  1. 按**精确序列**（非反向互补）分组。反向互补合并会翻转链敏感任务的标签（splice donor↔acceptor、启动子方向），故推迟到标签感知的后续处理，本轮不做 RC 合并。
  2. 同一序列跨划分出现时，整组归入**最高优先级**划分（test>dev>train），低划分副本删除。
  3. 同一序列映射到**冲突标签**（同组不同 answer）时，整组**隔离删除**，不按多数投票"修正"（符合任务目录 C01 的冲突组处理原则）。

## 3. 本轮产物（slice 1：有原生/干净划分的来源）

| 统计 | 值 |
|---|---|
| 任务数 | **46** |
| 接口分布 | choice 33 · noul 9 · score 3 · multi_noul 1 |
| 模态分布 | DNA 38 · 蛋白 8（RNA 待补，见 §4） |
| train 总行数 | 2,174,790 |
| test 总行数 | 350,727 |
| 隔离冲突组 | 1,984（virus_covid 1,754、prom_core 系列 143、tape_stability 63 等） |
| **准入结果** | 任务内 train/test/dev 泄漏 **0**；标签冲突（隔离后残留）0；非法 answer 0 |

任务族：GUE 28（DNA Choice：10 组蛋白标记 + mouse×5 + prom×6 + splice + tf×5 + virus）、Genomic Benchmarks 8（DNA，7 二分类 Noul + regulatory 3 类 Choice）、TAPE stability 1（蛋白 Score，5 级）、DeepSTARR 2（DNA Score：Dev/Hk）、DeepLoc 2.0 1（蛋白 multi_Noul，11 定位）、本地快照 6（fold_class/signal_peptide/subcellular_loc/npp 为 Choice，protein_homology_std/remote 为 Noul 双蛋白）。

## 4. 待补（needs_work，未纳入本轮）

| 来源 | 目录 ID | 接口 | 受阻/未做的具体点 |
|---|---|---|---|
| dnagpt dna_promoter_300 / core / splice / tf | C01/E01/C02/C03 | Noul/Choice | 仅有名为 train 的未划分池，且含旧本地 test 成员；需实体分组 + 旧 test 隔离后才能准入，不能随机重切当盲测 |
| gene_lan_transfer（8 配置） | C08/E03/E04 | Noul 双序列 | 同上池问题 + 端点跨配置交叉；双序列按连接组件分组 |
| ProteinGym v1 | E05 | Score（217 assay） | 数据已在库；官方 train/test 需外部 `DMS_subsplits.csv`（HF 仓库不含）；且 217 assay 需亲本/家族隔离 |
| remote_homology | E12 | Choice | fold 级候选集 1,195 类，需限定每题候选子集后才可用 |
| TAPE fluorescence | C09 | Score | 已在主实验使用，尚未转成统一 v2 格式 |
| RNAcompete | E08 | Score | clone 含 probe_intensity/kmer_zscore，需组装成 RNA+RBP 的 Score 样本 |
| DeepGOZero | C12 | multi_Noul | KAUST 数据 URL 404，需另找镜像或自建 |
| PEER / FLIP | E11 | Noul/Score | FLIP 621MB LFS、raw 不通；PEER 需跑下载管线 |
| DeepSEA | E07 | multi_Noul | 919 特征大文件，按需获取 |

## 5. 主张边界

本轮完成的是**数据管道与格式准入**，非模型评测结果。所有成绩须在未来 lock-test 上、按 §3 隔离后的 test 计。跨任务序列复用（731,024 个序列键出现在 >1 任务）在多任务统一模型下不作任务内泄漏处理，但后续"按任务划分训练/测试"（leave-one-task-out）时必须升级为跨任务全局隔离。`score` 的分级期望值与标量回归对照、`multi_noul` 的 sigmoid 头对照尚未实现。
