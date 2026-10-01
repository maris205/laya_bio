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

## 3. 当前产物（slice 1 + 2 + 3）

| 统计 | 值 |
|---|---|
| 任务数 | **290**（其中 ProteinGym 217 为同题型"突变适应度"的不同 assay，按官方基准逐个计） |
| 任务族 | 12 族：GUE 28 · GenomicBenchmarks 8 · gene_lan_transfer 8 · ProteinGym 217 · RNAcompete 14 · 本地快照 7 · dnagpt DNA 池 3 · TAPE 2 · DeepSTARR 2 · DeepLoc 2.0 1 · 其余 |
| 接口分布 | choice 37 · noul 17 · score 235 · multi_noul 1 |
| 模态分布 | DNA 42 · 蛋白 226 · RNA 14 · DNA+蛋白 3 · DNA+DNA 3 · 蛋白+蛋白 2（**三模态齐全**） |
| train 总行数 | 3,119,067 |
| dev 总行数 | 440,305 |
| test 总行数 | 581,242 |
| 隔离冲突组 | 1,984（virus_covid 1,754、prom_core 系列 143、tape_stability 63、dna_core 8 等） |
| **准入结果** | 任务内 train/test/dev 泄漏 **0**；标签冲突（隔离后残留）0；非法 answer 0；flagged 任务 0 |

### slice 3 新增（231 任务）与划分要点

- **ProteinGym（E05，217 替换型 assay，Score）**：GleghornLab/ProteinGym_DMS 的 per-assay parquet 自带官方 per-mutant fold 列（`fold_random_5`/`fold_contiguous_5`/`fold_modulo_5`）。采用 **fold_contiguous_5**（MSA 连续块，同源感知，符合目录"亲本/家族隔离"）：fold0=test、fold1=dev、folds2-4=train；`fold==-100`（多突变体，官方单突变基准外）剔除；锚点仅由 train 计算；输入 `mutated_seq`，目标 `DMS_score`。217 assay 全部有有效 fold0。
- **RNAcompete（E08，14 RBP，Score，补 RNA 模态）**：本 clone 仅 14-RBP 测试子集（非完整 ~200 RBP，已在 §4 注明）。用 `probe_metadata.tsv`（探针→RNA 序列）× `probe_zscore.tsv`（探针 × 14 HybID 结合 z 分数），每 RBP 一个 Score 任务；探针随机固定子采样 25,000（记录于 manifest），按探针哈希分桶 train/dev/test。定位为相对结合偏好，不称 Kd。

任务族：GUE 28（DNA Choice：10 组蛋白标记 + mouse×5 + prom×6 + splice + tf×5 + virus）、Genomic Benchmarks 8（DNA，7 二分类 Noul + regulatory 3 类 Choice）、gene_lan_transfer 8（双序列 Noul：DNA-蛋白配对 ×3、DNA-DNA 相似 ×3、蛋白-蛋白相似 ×2）、本地快照 6（fold_class/signal_peptide/subcellular_loc/npp 为 Choice，protein_homology_std/remote 为 Noul 双蛋白）、dnagpt DNA 池 3（splice/tf/core，旧-test 隔离）、TAPE 2（fluorescence/stability，蛋白 Score）、DeepSTARR 2（DNA Score：Dev/Hk）、DeepLoc 2.0 1（蛋白 multi_Noul，11 定位）。

### slice 2 新增（12 任务）与隔离要点

- **dnagpt DNA 池（dna_splice/dna_tf/dna_core，C02/C03/E01）**：上游池只有未划分 train 且**含论文旧盲测成员**。处理：test = 旧 `lg_*` test（论文盲测，逐字保留）；train/dev = 池序列**剔除全部旧-test 成员**后按序列哈希分桶（dev 10%）；池 int 标签与 lg test 文本标签不一致者（dna_core 8 条）判为标注冲突**整组隔离**，非多数投票。**不对池随机重切伪造盲测**。
- **标签映射经 pool∩lg-test 全量重叠验证**：tf 3437/3437、core 5910/5918 一致；splice 在旧约定（0→Non-Splice,1→Acceptor,2→Donor）下 4545/4545 一致，而任务目录所载"新映射"(0→Acceptor) 为 0/4545。**采用与数据经验一致的旧约定**以保证 train/test 标签自洽；acceptor/donor 的生物学语义是否被交换仍按目录警示留待独立核实（不影响基准内部一致性）。
- **gene_lan_transfer 8 配置（C08/E03/E04）**：双序列 Noul，按**端点并查集连通分量**整组划分（80/10/10），避免同一端点跨 train/test。定位为构造相似性诊断（rand/rand_v2 可由 frame-0 标准密码表翻译判别），**非独立盲测**；跨配置端点复用仍在（rand∩rand_v2 共享 12,526 序列），leave-one-task-out 时须升级为全局隔离。
- **TAPE fluorescence（C09）**：转 Score（5 级，train 分位 anchors），与主实验同源。

## 4. 待补（needs_work）

slice 3 已纳入 ProteinGym（E05，官方 fold_contiguous_5）与 RNAcompete（E08，RNA 模态）。仍需补：

| 来源 | 目录 ID | 接口 | 受阻/未做的具体点 |
|---|---|---|---|
| remote_homology | E12 | Choice | fold 级 1,195 类，动态候选接口需每题限定候选子集（真 fold + decoy）；class_label 7 类与 lg_fold_class 近重复，不单独计。fold/superfamily/family 三级 holdout test 已在库，转换待设计 |
| RNAcompete 完整性 | E08 | Score | 已纳入 14 RBP 测试子集；完整 ~200 RBP 需另取上游矩阵 |
| DeepGOZero | C12 | multi_Noul | KAUST 数据 URL 404，需另找镜像或自建 |
| PEER / FLIP | E11 | Noul/Score | FLIP 621MB LFS、raw 不通；PEER 需跑下载管线 |
| DeepSEA | E07 | multi_Noul | 919 特征大文件，按需获取 |
| dnagpt dna_promoter_300 | C01 | Noul | lg_promoter_detection 已作原生 train/val/test 纳入（slice2）；上游池未单独纳入，避免与 lg 版本重复计任务 |

## 5. 主张边界

本轮完成的是**数据管道与格式准入**，非模型评测结果。所有成绩须在未来 lock-test 上、按 §3 隔离后的 test 计。任务计数按"评测单元"：ProteinGym 217 assay 与 RNAcompete 14 RBP 各为一个 Score 评测单元，但同属"突变适应度 / 结合偏好"一题型，题型多样性应按 12 族计，勿把 290 全数当作 290 种独立问题宣称。跨任务序列复用（925,985 个序列键出现在 >1 任务）在多任务统一模型下不作任务内泄漏处理，但后续"按任务划分训练/测试"（leave-one-task-out）时必须升级为跨任务全局隔离。`score` 的分级期望值与标量回归对照、`multi_noul` 的 sigmoid 头对照尚未实现。
