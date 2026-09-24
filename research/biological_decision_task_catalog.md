# 生物序列决策任务目录：Choice、Score、Noul

更新：2026-09-24。**这是来源核查与后续实验设计，不是 24 项已完成实验。** 目录含 12 个优先任务及 12 个扩展任务视图；同源数据、重复来源和同一任务的不同构造不能累计成独立证据。已有论文结果仍只有启动子与七类蛋白结构分类。

## 三类接口怎么对应生物问题

依据 [Jev 官方介绍](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、[Score 文档](https://docs.typesafe.ai/primitives/score)及 [Noul 文档](https://docs.typesafe.ai/primitives/noul)：

| 接口 | 生物问题 | 建议输出与监督 | 主要评价 |
|---|---|---|---|
| **Choice** | 互斥类别中选一个，如剪接位点类型、蛋白结构类别 | 候选类别概率，和为 1；交叉熵 | Accuracy、Macro-F1、NLL、Brier、候选合法率 |
| **Score** | 有序水平，如低到高的荧光、稳定性、活性 | 有序等级概率及期望等级；连续观测值需另定义训练集分箱/锚点，映射回原单位 | 原单位 MAE/RMSE、Spearman；量化误差、范围外覆盖率 |
| **Noul** | 一个命题是否成立，如启动子、结合位点、同源关系 | 一个命题的 yes 概率；二元监督 | AUROC、AUPRC、MCC、F1、Brier及校准 |
| **多标签＝逐标签 Noul** | 一个蛋白可同时有多个定位或功能 | 每个标签一个边际概率，允许同时为真；训练用掩码二元损失 | Micro/Macro-AUPRC、Micro/Macro-F1、逐标签校准 |

二分类也能写成两个候选的 Choice，主实验固定一个接口，二者转换只是等价性消融，不重复计任务。多标签不能直接套一个互斥 softmax。Score 原生是有序决策接口，不等于已经具备任意连续回归能力。现有 Laya-Bio checkpoint 只完成 Choice 适配；新增接口需实现、训练和验证。

统一输入建议为 `{sequences: [{role, modality, sequence}], question, primitive, candidates_or_levels}`。同源关系保留 protein A/B；DNA–protein 配对保留编码序列与蛋白的不同角色。不要将双序列拼接成没有边界的一条字符串。

## 优先 12 项：覆盖三类接口、单/双序列与多标签

“本地审计”仅表示格式、标签和部分重复检查完成；不表示新划分、同源隔离和 tokenizer 长度准入已通过。“来源已核实”表示核对了原始项目/论文，尚未下载并准入该任务。

| ID | 典型问题与输入 | 接口 | 数据来源 | 当前规模/状态 | 关键准入要求 |
|---|---|---|---|---|---|
| C01 | 300 bp DNA 是否为启动子 | Noul | [dnagpt/dna_promoter_300](https://huggingface.co/datasets/dnagpt/dna_promoter_300) | 59,195 行；已下载审计；旧研究视图 21,042 行 | 15 个标签冲突序列组；新源包含旧 test，须隔离 |
| C02 | DNA 是 acceptor / donor / non-splice 哪一类 | Choice，3 类 | [dnagpt/dna_splice_site_prediction](https://huggingface.co/datasets/dnagpt/dna_splice_site_prediction) | 45,619 行，400 bp；已下载审计 | 1 个冲突组；保留旧 4,562 行测试成员；簇/基因组区段隔离 |
| C03 | DNA 是否为该数据定义的 TF 结合位点 | Noul | [dnagpt/dna_transcription_factor_prediction](https://huggingface.co/datasets/dnagpt/dna_transcription_factor_prediction) | 34,378 行，101 bp；已下载审计 | 核实 TF/细胞背景；不能称为所有 TF 的通用预测 |
| C04 | 蛋白属于哪个结构类别 | Choice，7 类 | [现有固定研究快照](https://huggingface.co/datasets/dnagpt/laya-bio) `lg_fold_class` | 本地 19,468 行；旧研究任务 | 是粗结构类别，不等同于 TAPE 的远缘同源类别 |
| C05 | 蛋白信号肽属于 Sec 还是 Tat | Choice，2 类 | 本地 LLaMA-Gene 转换快照 `lg_signal_peptide` | 本地 8,304 行；已审计 | 不把 Sec/Tat 判断写成“有无信号肽”；核实选样条件 |
| C06 | 原核蛋白属于哪个亚细胞位置 | Choice，6 类 | 本地 LLaMA-Gene 转换快照 `lg_subcellular_loc` | 本地 12,993 行；已审计 | 本数据是单标签；部分序列与其他任务交叉，最长 5,627 aa |
| C07 | 两个蛋白是否同源 | Noul，双蛋白 | 本地 `protein_homology_std` / `protein_homology_remote` | 16,168 / 25,647 行；格式审计通过 | 两个数据视图不算两个独立任务；天然同源定义、家族来源及簇隔离待核实 |
| C08 | DNA 与蛋白是否形成编码配对 | Noul，DNA＋蛋白 | [dnagpt/gene_lan_transfer](https://huggingface.co/datasets/dnagpt/gene_lan_transfer) `dna_protein_pair_rand_v2` | 16,000 行；已下载审计 | 定位为一致性诊断：当前样本可由标准密码表直接翻译完全判别；必须有规则基线 |
| C09 | GFP 变体的荧光水平多高 | Score | [TAPE fluorescence](https://github.com/songlab-cal/tape) | 数据已下载；六任务工程 pilot，完整准入待做 | 保留突变距离/亲本关系；训练集定义等级；同时比较标量回归 |
| C10 | 蛋白序列的稳定性水平多高 | Score | [TAPE stability](https://github.com/songlab-cal/tape) | 来源已核实；数据准入待做 | 保留原生评估与设计家族；报告原观测尺度，不虚构物理单位 |
| C11 | 蛋白可同时定位到哪些细胞区室 | 多个 Noul，10 标签 | [DeepLoc 2.0](https://academic.oup.com/nar/article/50/W1/W228/6576357) | 28,303 条官方 train/validation 数据已下载；工程 pilot | 真多标签，与 C06 不同；冻结 2.0 数据版本及同源划分 |
| C12 | 蛋白具有哪些分子功能 | 多个 Noul，GO MF | [DeepGOZero](https://github.com/bio-ontology-research-group/deepgozero) / [论文](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i238/6617515) | 来源已核实；先规划训练集高频 32 个 MF 项的 Lite 视图 | 固定 GO/注释日期；未注释不等于实验阴性；报告注释恢复口径与掩码 |

C08 是纳入 12 项的机制诊断，不为自然生物任务广度增加一票。C12 的 32 项由训练集计数确定，不依据 test 可预测性筛选；完整 GO 评价属于后续扩展，不能将 Lite 结果称为全 GO 功能预测。

## 另外 12 个扩展视图

| ID | 任务 | 接口 / 输入 | 原始来源与选择理由 | 主要限制或定位 |
|---|---|---|---|---|
| E01 | 70 bp core promoter | Noul / DNA | [dnagpt/dna_core_promoter](https://huggingface.co/datasets/dnagpt/dna_core_promoter)，59,196 行已下载审计 | 与 C01 同类问题，作为窗口/域迁移；64 个冲突组，不当成新任务语义 |
| E02 | 神经肽前体 | Noul / 蛋白 | 本地 LLaMA-Gene 转换快照 `lg_npp`，本地 3,364 行 | 检查其原始二分类标签、来源和长序列；补充低资源蛋白任务 |
| E03 | DNA 配对相似性 | Noul / DNA＋DNA | `gene_lan_transfer` 的 50/150 bp 及 simple 配置 | 已下载；是构造相似性，不能直接称自然同源；比对/编辑距离基线 |
| E04 | 蛋白配对相似性 | Noul / 蛋白＋蛋白 | 同一来源 `protein_sim_pair_150bp` / `450bp` | 已下载；名称 bp 是源配置名，实际输入是氨基酸；与 C07 分开 |
| E05 | 突变体功能效应 / fitness | Score / 蛋白 | [ProteinGym](https://github.com/OATML-Markslab/ProteinGym)，DMS 实验 | 按 assay 评价；不同实验尺度不直接混合；亲本/蛋白家族隔离 |
| E06 | 发育型与 housekeeping 增强子活性 | 两个 Score / DNA | [DeepSTARR](https://github.com/bernardo-de-almeida/DeepSTARR) | 两个连续输出；不是二分类或两个互斥类别；序列位点隔离 |
| E07 | 染色质特征预测 | 多个 Noul / DNA | [DeepSEA 原始论文](https://www.nature.com/articles/nmeth.3547)，919 个特征 | 约 1 kb 输入和大标签集，先检查完整上下文与推理成本；坐标划分 |
| E08 | RNA 与指定 RBP 的结合偏好 | Score / RNA＋RBP 问题 | [RNAcompete](https://github.com/morrislab/RNAcompete)，[GSE41235](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41235) | 是相对结合偏好，不擅自称 Kd；引入 RNA 需新 tokenizer/字母表准入 |
| E09 | DNA enhancer / non-enhancer | Noul / DNA | [Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) 的 `human_enhancers_cohn` 子集 | 冻结具体子集与来源；排查启动子/基因组位点重叠 |
| E10 | DNA coding / non-coding | Noul / DNA | 同一 Genomic Benchmarks 的 `demo_coding_vs_intergenomic_seqs` 子集 | 保留物种和构建规则；组成捷径/长度匹配检查 |
| E11 | 两个蛋白是否发生相互作用 | Noul / 双蛋白 | [PEER Benchmark](https://github.com/DeepGraphLearning/PEER_Benchmark) | 生物相互作用与同源不是同一关系；按端点实体/家族划分，检查负例生成 |
| E12 | 远缘同源类别识别 | Choice / 单蛋白 | [TAPE remote homology](https://github.com/songlab-cal/tape) | 是单序列类别任务，不与双序列同源二分类混写；保持官方 family/superfamily/fold 难度口径 |

扩展来源已核实，不代表所有数据文件已下载、许可证已逐项准入或可直接训练。对于 E09/E10，已选具体子集，但执行前仍须冻结文件版本、标签映射及来源坐标。本地扩展转换文件未随原两任务基准发布；其文件哈希和源字段见审计 JSON，不能把原研究下载地址当作这些任务的已发布版本。

## 当前数据审计结论

可复核材料：[本地 8 任务审计](../artifacts/laya_task_expansion/admission_audit.json)、[上游池与双序列审计](../artifacts/laya_task_expansion/upstream_audit.json)、[固定版本](../artifacts/laya_task_expansion/upstream_datasets.json)、[下载 SHA-256](../artifacts/laya_task_expansion/download_receipt.json)。审计脚本见 [local](../scripts/audit_laya_task_expansion.py) 与 [upstream](../scripts/audit_laya_upstream_tasks.py)。

- 四个新 DNA 仓库及 `gene_lan_transfer` 的八个配置都只有名为 `train` 的未划分池。该名字不是安全训练许可：DNA 上游池包含旧本地 test 成员。所有旧 test 及其同源/反向互补组保持隔离，不能把全池随机重切后声称新盲测。
- DNA 数字标签与旧文本标签的匹配支持：promoter/core `1→promoter, 0→non-promoter`；splice 的旧映射 `1→acceptor, 2→donor, 0→non-splice` **仅是与旧本地文本一致，未验证语义，现已撤回该解释**；依据 SpliceFinder 原代码和训练 motif，新构建器采用 `0→acceptor, 1→donor, 2→non-splice`，详见 [后续诊断](laya_task_diagnostics.md)；TF `1→binding, 0→background`。启动子冲突组出现少量相反标注，须整组隔离并追查，不能按测试多数投票“修正”。
- 本地结构类别与定位任务有 55 条完全相同序列；结构类别与神经肽任务有 11 条；信号肽与定位任务有 3 条。定位 train 包含一条旧 fold test 和一条旧 signal test 成员。必须做跨任务的全局分组。
- 两套蛋白同源配对数据格式解析全部通过；各自原生 train/val/test 未发现完全相同的端点序列或无序配对交叉。**这仍未证明家族/同源簇独立。** 最大合计长度为 500/499 字符。
- `dna_protein_pair` 的 2,000 正、2,000 负及 `rand_v2` 的 8,000 正、8,000 负可由 frame-0 标准密码表精确翻译匹配完全区分；这是全池构造诊断，不能报告为独立测试成绩。`rand` 中仅 4,368/8,000 正例精确匹配，不能在未核实遗传密码、起始规则等前断言其他标签错误。
- 多个 gene 配置共用端点：`rand` 与 `rand_v2` 交叉 12,526 条序列。不得一个配置作 train、另一个直接当独立 test。双序列划分基于端点/亲本簇，负配对在分区内生成；现成配对按连接组件划分，巨型组件不足以分割时需重建或退出准入。
- 长度是实际风险：DNA–protein 的 `rand_v2` 最大 DNA 长度 47,496、蛋白 15,831。完整输入适配应按各模型真实 tokenizer 取共同可容纳子集，并报告长序列排除率；不静默截断。

## 执行顺序与主张边界

先做六任务开发 pilot：C01、C02、C04、C07、C09、C11，覆盖三类接口、两种模态、双序列和多标签；通过数据与接口准入后，扩展到完整优先 12 项。若来源或独立测试集不合格，保留未准入状态，不用近重复任务凑数。

主比较为共享 typed scorer、同底座共享编码器＋任务头、同一生成器 checkpoint 的自由生成／受约束生成／完整候选似然。Score 增加真正的标量回归对照；多标签增加 sigmoid 多标签头，不能只拿不合适的 softmax 做对照。跨接口指标分别报告，不把 AUPRC、准确率和相关系数直接平均。

“未见任务”另开 leave-one-task-out 训练：C10 稳定性完全移除；以 C06 原核定位为目标时同时移除 C11 等定位监督，再测迁移。包含全部 12 项的联合 checkpoint 不允许对其中任务声称零样本。Score 的严格零样本迁移还需不依赖留出任务标签统计的外部等级与数值锚点，否则只能称适配。详细设置见[实验计划](../refine-logs/EXPERIMENT_PLAN.md)与[执行表](../refine-logs/EXPERIMENT_TRACKER.md)。
