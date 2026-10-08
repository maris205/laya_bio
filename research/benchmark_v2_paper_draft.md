# BioDecisionBench：防泄漏的生物序列类型化决策基准与 seed 平均架构对照

**Liang Wang**（School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan 430070, China；通讯：wangliang.f@gmail.com）

**结果版工作稿，2026-10-06（强化实验已回填）。** 依据 16 份实验文档（`research/benchmark_v2_*.md`）与已落盘 JSON 结果撰写；venue 未绑定（ICLR 9 页 / NeurIPS D&B 均可适配）。两项强化实验已完成并入正文：noul 措辞×3 稳定 checkpoint 消融（`noul_ablation_3seed.json`）、跨族 score 臂 3-seed（`cross_family_tape_3seed.json`）。英文 LaTeX 版在本稿冻结后翻译。

## 摘要

同一个接口能回答三类问题：给序列 `GTATTCTCCCCT...` 配问题"Is this DNA sequence a promoter?"与候选 {yes, no}，模型返回 P(yes)；只换问题与候选集——折叠分类换成 {All Alpha, All Beta, ...}、突变适应度换成五个带锚点的有序档位——同一模型即答所有题：序列、问题、候选全部作为文本编码，由**一个共享打分器**对每候选输出一个概率，无任务专用头、无自由文本生成。这类类型化决策接口正被采用，但从未在匹配算力下与常规任务头对照。我们发布 **BioDecisionBench**：290 个防泄漏的生物序列决策任务，跨 DNA/蛋白/RNA 三模态与四种类型化决策接口（Noul/Choice/Score/multi-Noul），全部记录带来源、修订号、SHA-256 与逐任务准入审计（隔离后任务内 train/test/dev 泄漏为 0，1,984 个标签冲突组被整组隔离而非多数投票）。利用该基准，我们首次在**匹配算力**下对三种决策架构做 **seed 平均**对照：共享类型化决策打分器（无任务参数、动态候选文本）、同 backbone 每任务输出头、以及冻结小 LM 的全候选似然。3-seed 平均显示：**每任务头总体不劣于且部分显著优于共享打分器**（noul AUROC 0.629±0.050 vs 0.478±0.051；choice 0.489 vs 0.486、score spearman 0.018 vs 0.013 持平），且跨 seed 更稳定（noul run-std 0.030 vs 0.087）；两个训练臂均超过冻结 LM 基线。共享打分器的优势是**定性的**：它能对任意新任务零适配推理（每任务头架构上做不到），但其零样本性能 ≈ 冻结 base（同族 −0.018）、跨族迁移弱（+0.036±0.034，仅分类），且每任务头只需 **~32 条标注**即追平其零样本水平（7/11 任务）。方法论上，我们证明该 regime 下**单次 run 的结论不可信**：同配置同 seed 重跑差异达 ±0.05–0.1（score 0.107→0.015），足以推翻单次对照的胜负方向；seed 平均是此类比较的最低要求。

## 1. 引言

一个模型能否用同一接口回答三类问题，而且接口可以简单到只是文本：序列 `GTATTCTCCCCT...` + 问题*"Is this DNA sequence a promoter?"* + 候选 {yes, no} → 返回 P(yes)；只换问题与候选集——{All Alpha, All Beta, Alpha and Beta, ...} 答折叠分类、五个带值锚点的有序档位答突变适应度、逐标签 yes/no 答亚细胞定位——同一模型全答，无需任务专用输出头、无需生成自由文本。这就是**类型化决策（typed-decision）接口**。这一接口被商业系统（TypeSafe Jev）与社区复现（NanoJev、GLiClass）采用，其核心卖点是：单一小模型、动态候选、新任务零适配成本。但两个基础问题从未被受控回答：**(Q1)** 共享打分器在匹配算力下是否匹敌或优于常规的每任务输出头？**(Q2)** "动态候选/零适配"卖点能否转化为可测量的性能或成本优势？

回答这两个问题被三个障碍挡住。其一，生物序列评测碎片化：GUE、Genomic Benchmarks、TAPE、ProteinGym、DeepLoc 等各按族各自为政，接口口径（分类/回归/二元）不统一，无法在一个协议下比较决策架构。其二，泄漏审计缺失：上游数据池（如 dnagpt 系列 HF 数据集）只有名为 train 的未划分池，且**包含既往研究的盲测成员**；随机重切会伪造"新盲测"。其三——本文方法论主轴——**评测方差未被量化**：我们在小预算多任务 regime 下发现，同配置同 seed 的两次训练因 GPU 非确定性与早停选点噪声，结果差异达 ±0.05–0.1，足以把对照的胜负方向整个翻转（§5.1）。任何基于单次 run 的架构结论都不可靠。

本文的贡献：

1. **BioDecisionBench**（§3）：290 任务、10 族、三模态、四类型化接口的统一记录格式基准；防泄漏准入协议（精确序列跨划分隔离、冲突组整组隔离、旧盲测逐字保留、标签映射经全量重叠验证）；隔离后任务内泄漏 0。
2. **seed 平均评测协议与方差量化**（§4.3, §5.1）：给出单次 run 翻车的具体案例与可操作协议（≥3 seeds、每任务先 seed 内平均、报 SEM 与 run-std、按族 dev 早停且 dev≥256）。
3. **首个三方匹配对照**（§5.1–5.2）：共享打分器 vs 每任务头 vs 冻结 LM 候选似然。结论诚实：每任务头 ≥ 共享打分器（noul 显著、choice/score 持平、更稳定）；共享打分器的零样本/跨族/适配成本三个操作化检验均未显示定量优势——其价值是接口灵活性本身。
4. **训练动力学与机制诊断**（§5.3–5.4）：naive 预算放大失稳（低于 init）→ 稳定化修复但确认平台；按族早停零推理成本、全族不劣于全局单 checkpoint（族峰值 step 跨 26 倍）；noul 候选措辞受控消融（同 checkpoint +0.066 AUROC）与冻结 VLM 对比解码（消除候选坍缩）两个可复用机制修复。

**结果预览**：seed 平均主对照中每任务头 noul AUROC 0.629 vs 共享打分器 0.478（SEM 0.05 不重叠）；而单次 run 曾给出共享打分器领先的相反结论。适配成本检验中，每任务头新建一个头只需 ~32 条标注即追平共享打分器的零样本水平。

**Figure 1（hero）**：左联=基准全景（10 族×4 接口×3 模态任务数矩阵 + 统一记录 schema + 准入流水线示意）；右联=seed 平均三方条形图（按 primitive，SEM 误差棒），内嵌"单次 run 方差"小图（同配置两次 run 的 score 0.107 vs 0.015 vs seed 平均 0.013）。

## 2. 相关工作

**生物序列基准与基础模型。** DNA 侧有 GUE/DNABERT-2（28 任务表观/启动子/剪接/TF）、Genomic Benchmarks（8 子集增强子/启动子/编码区）、BEND、DeepSEA（919 染色质特征）、DeepSTARR（发育/管家增强子活性双连续输出）；蛋白侧有 TAPE（荧光/稳定性/远缘同源）、ProteinGym（217 个 DMS 突变适应度 assay，官方 per-mutant fold 划分）、FLIP、DeepLoc 2.0（10 定位多标签）、DeepGOZero（GO 功能）；RNA 侧有 RNAcompete（RBP 结合偏好）。这些基准按族割裂：各自的任务格式、划分协议与指标口径互不兼容，且以"单序列单任务型"为主。ESM/DNABERT 系基础模型在这些族上刷榜，但**没有任何基准把四类型化决策接口统一到同一记录格式与准入协议下**，也没有覆盖"双序列（DNA-蛋白配对、蛋白-蛋白同源）+ 多标签 + 有序回归"的接口多样性。BioDecisionBench 不是又一个刷榜基准：它把上述来源转成统一决策记录，以泄漏审计为一等公民（上游池含旧盲测成员的处理协议见 §3.3），并作为架构对照的评测底座。

**类型化/候选打分决策接口。** TypeSafe 的 Jev 提出 Noul/Choice/Score 结构化概率判断（服务接口，权重未开放）；社区复现包括 NanoJev（Qwen3-0.6B backbone + 共享标量头 + 候选集合 attention）、GLiClass（ModernBERT uni-encoder 动态标签分类）、SemIf（冻结 LLM 直接读选项 logits）。约束解码（如 PICARD 类方法）是生成式路线的对照面。既有工作各自验证接口可行性，但**共享打分器 vs 每任务头**这一架构选择从未在匹配算力、同一数据、多 seed 下被对照过——已有报告多为单任务、单 run、无方差量化。本文补上这一对照，并给出与直觉相反的诚实结果。

**评测方法论与方差。** Picard & Torchia（2020）从统计机制上论证深度模型评测须多次重复；Miller（2024）给出 evals 误差棒的统计处理；REVAL 等工作报告 LLM 评测对表述/顺序敏感。生物序列方向另有候选顺序敏感性证据（我们此前的 Laya-Bio 工作发现候选重排改变 ~9% 预测）。本文以一个**具体的结论翻转案例**加入这一脉络：同一对照、同一配置、同一 seed，两次 run 的 score spearman 分别为 0.107 与 0.015，胜负方向相反；3-seed 平均后单次"#1≥#2"结论被推翻。据我们所知，这是决策架构对照中 run 方差推翻结论的首个完整记录案例。

**探测与适配成本。** linear probing 与 few-shot head fitting 是表征评测的标准工具。本文把"零适配接口"的卖点操作化为**追平成本**：冻结编码器、为新任务新建一个头、用 N∈{32,128,512} 条标注拟合，测追平共享打分器零样本所需的最小 N。这一口径把"接口灵活性"从定性主张变成可比较的数字。

## 3. BioDecisionBench：设计与构建

### 3.1 统一接口与记录格式

每条样本一行 JSON：`{task_id, primitive, modality, sequences:[{role, modality, sequence}], question, split, group, provenance}` 加接口字段：

| primitive | 追加字段 | 监督/评测 |
|---|---|---|
| choice | candidates（互斥标签表）、answer | CE；accuracy、macro-F1、NLL、候选合法率 |
| noul | candidates=["no","yes"]、answer | 二元 CE；AUROC、MCC、acc、校准 |
| score | score_levels(5)、anchors(train 分位派生 4 内切点→5 代表锚值)、gold_value、gold_level | 软目标 CE+期望值 MSE；spearman（主）、按 train-std 归一化 RMSE |
| multi_noul | candidates(全标签)、labels(逐标签 0/1)、answer(正标签集) | 掩码二元；micro/macro-F1、逐标签校准 |

双序列任务保留角色边界（`query_1`/`query_2`，DNA+蛋白、DNA+DNA、蛋白+蛋白），不拼接为单串。`provenance` 记录源 repo、修订号/commit、license；下载凭证含逐文件 SHA-256。

### 3.2 来源、规模与模态

10 族 290 任务（按训练调度器的 task-id 前缀映射；上游数据源可细分至 12 个）：**GUE** 28（DNA choice：10 组蛋白标记+mouse×5+prom×6+splice+tf×5+virus）；**Genomic Benchmarks** 8（DNA：7 noul+regulatory 3 类 choice）；**ProteinGym** 217（蛋白 score，官方 fold_contiguous_5 per-mutant 划分：fold0=test/fold1=dev/fold2-4=train，−100 多突变体剔除）；**RNAcompete** 14（RNA score，14-RBP 子集，探针 25k 固定子采样）；**gene_lan_transfer** 8（双序列 noul：DNA-蛋白配对×3、DNA-DNA×3、蛋白-蛋白×2，端点并查集连通分量分组划分）；**dnagpt DNA 池** 3+1（splice/tf/core+promoter）；**TAPE** 2（荧光/稳定性 score）；**DeepSTARR** 2（DNA score 双输出拆两任务）；**DeepLoc 2.0** 1（蛋白 multi_noul 11 定位）；**本地快照** 7（fold/signal/subcell/npp choice + protein_homology std/remote 双蛋白 noul）。

接口分布 choice 37 / noul 17 / score 235 / multi_noul 1；模态 DNA 42 / 蛋白 226 / RNA 14 / 双序列 8。行数 train 3,119,067 / dev 440,305 / test 581,242。**任务计数口径**：290 按评测单元计；题型多样性按 10 族计（ProteinGym 217 assay 同属"突变适应度"一题型）——正文与表格均双口径报告，不以 290 宣称 290 种独立问题。

### 3.3 防泄漏准入协议

1. **原生划分优先**：有官方 train/dev(/test) 的来源逐字沿用（GUE、GB、TAPE、DeepSTARR、ProteinGym fold、本地快照 train/val/test）。
2. **上游池的旧盲测隔离**：dnagpt 池（splice/tf/core）只有未划分 train 池且含旧研究盲测成员。处理：test=旧盲测逐字保留；train/dev=池剔除**全部**旧盲测成员后按序列哈希分桶；池 int 标签与盲测文本标签不一致者（core 8 条）判标注冲突整组隔离。标签映射不靠猜：经池∩盲测**全量重叠**验证（tf 3437/3437、core 5910/5918；splice 在旧约定 0→Non-Splice/1→Acceptor/2→Donor 下 4545/4545 一致，而文献新构建器约定 0/4545——采用与数据自洽的约定并显式记录 acceptor/donor 语义存疑，供后续独立核实）。
3. **跨划分精确序列隔离**：同序列出现在多个划分时整组归最高优先级（test>dev>train），低划分副本删除；同序列冲突标签整组隔离（共 1,984 组，virus_covid 占 1,754）。**不做反向互补合并**：RC 会翻转链敏感任务（splice 供体/受体、启动子方向）的标签，label-aware RC 分组留作未来工作。
4. **双序列按端点分组**：gene_lan 配对按端点并查集连通分量整组划分，避免同一端点跨 train/test；构造相似性诊断定位（rand/rand_v2 可由 frame-0 密码表翻译判别）在任务 note 与正文双重声明，不作独立盲测宣称。
5. **Score 锚点纪律**：5 级锚点仅由 train 派生，test 不反灌；跨 assay 不平均原始 RMSE（原生尺度差异巨大），主指标 spearman。
6. **准入检查器**：逐任务验证划分重叠/标签冲突/answer∈candidates/长度分布，全局跨任务重复统计（925,985 个序列键出现在 >1 任务，多任务统一模型下不构成任务内泄漏，leave-one-task-out 时须升级为全局隔离）。最终：**任务内泄漏 0、flagged 任务 0**。

### 3.4 边界

当前架构约束（单序列、DNA/蛋白、512-token 预算）下 241/290 任务可被本文的编码器评测路径消费；49 个任务（RNA 14、双序列 10、multi_noul 1、超预算 24）已入基准但由相应模态/接口的评测路径覆盖或报排除率，不静默截断。基准的价值在协议与数据，不绑定单一被评架构。

## 4. 评测框架

### 4.1 三个被比架构

| | #1 共享 typed-decision scorer | #2 编码器+每任务头 | #3 冻结生成式候选似然 |
|---|---|---|---|
| backbone | Laya ModernBERT-large（423M，生物 BPE 扩表 52,412） | **同 #1** | Qwen3-0.6B（596M 因果 LM） |
| 决策机制 | 2 层 typed transformer + 共享标量 scorer 对候选 marker 打分；choice 过候选集合 attention；score 输出 5 级分布+期望值 | 每任务一个 Linear 头：choice/noul=固定类 softmax，score=标量回归（MSE） | 每候选 length-normalized logprob，argmax |
| 任务参数 | **无**（候选是输入文本） | 每任务 |N| 维头 | 无（frozen） |
| 新任务 | 零适配直接推理 | 须新建头+标注训练 | 零适配直接推理 |
| 训练 | 匹配：同数据/更新数/优化器/seed | 同 #1 | 不训练 |

#1/#2 是 matched compute（唯一变量=决策机制）；#3 是 frozen 异 backbone、序列截断 120 字符、score 口径不同（5 锚点等级 acc），**只作"现成小 LM 基线"**，全部表格显式标注。

### 4.2 训练协议

family-balanced 交错（每族每 epoch 等量 batch，族内轮转——防 ProteinGym 217 assay 淹没多样族）；AdamW，encoder lr 2e-5 / head-scorer lr 1e-4，5%→15% warmup + cosine 至 10%；grad clip 0.5；batch 64、micro-batch 梯度累积；512-token 全输入不截断、超预算排除并报率。稳定化参数（warmup 0.15、clip 0.5）来自 §5.3 的失稳诊断，非事后挑选。checkpoint 选择：**按族 dev 早停**（dev≥256/任务，每族保存各自 best-dev checkpoint；同一训练轨迹多 checkpoint，推理期零成本），动机=族峰值 step 跨 26 倍（§5.3）。

### 4.3 seed 平均协议（方法论贡献）

动机案例（§5.1）：同配置同 seed 两次 run，score spearman 0.107 vs 0.015，胜负方向翻转。协议：**≥3 seeds**（20261001/02/03）；每任务先 seed 内平均再跨任务聚合；报 mean±SEM（跨任务）与 run-std（跨 seed）双口径；早停 dev 样本 ≥256 以压选点噪声；单次 run 数字一律标注"单次抽样"。GPU 非确定性（同 seed 轨迹在 ~137 步后分叉）是该方差的物理来源，不可通过固定 seed 消除。

### 4.4 指标口径

choice：accuracy（主）+macro-F1；noul：AUROC（主，排序指标不受阈值/校准影响）+MCC+acc；score：spearman（主，尺度无关）+按 train-std 归一化 RMSE——**跨 assay 不平均原始 RMSE**（ProteinGym 各 assay 原生尺度差数个量级，原始均值被单一 assay 主导）。multi_noul：逐标签 sigmoid 口径（本文编码器路径未消费，基准已备好）。

## 5. 结果

### 5.1 主对照：seed 平均三方（Table 2 = 论文主表）

37 任务 matched 切片（cap ProteinGym=12/GUE=12；37 任务=choice 21+noul 7+score 9），2520 updates，3 seeds，按族早停：

| primitive | metric | n | #1 共享打分器 | #2 每任务头 | #3 冻结 LM |
|---|---|---|---|---|---|
| choice | accuracy | 21 | 0.486±0.031 | **0.489±0.031** | 0.424 |
| noul | AUROC | 7 | 0.478±0.051 | **0.629±0.050** | 0.522 |
| score | spearman | 9 | 0.013±0.009 | 0.018±0.033 | n/a(异口径) |

逐任务（seed-mean）：#1 胜 4 / 平 23 / #2 胜 10。跨 seed run-std：choice 0.040(#1)/0.030(#2)、noul **0.087/0.030**、score 0.064/0.054——#1 全面更不稳。

**稳健性：编码器初始化不变性（cpt 臂 3-seed 复跑）**：全协议在生物 CPT 初始化上重复，
排序不变——noul 0.651±0.042 vs 0.519±0.020、choice 0.500 vs 0.485、score 0.056 vs −0.001，
逐任务胜负 3/22/12；CPT 把两臂 noul 各抬 +0.041/+0.022 但不 closing gap。架构结论不依赖
编码器初始化（附录 TABLE_9）。

**方差案例（C2 实证）**：单次 run（bv2_pf，旧措辞）曾给出 #1 score 0.107 ≫ #2 −0.007、"#1 总体≥#2"的结论；同配置重跑（bv2_pf2）score 跌到 0.015——而 score 任务根本不涉及被改动的 noul 措辞，证明差异是纯 run 方差（per-family best step 漂移：GUE 200→5000、GB 1200→200）。3-seed 平均后 score 两臂都收敛到 ~0.01–0.02（持平），"#1 score 明显胜"不成立。

**判读**：(i) #2 ≥ #1：noul 显著（SEM 不重叠）、choice/score 持平、稳定性全面占优；(ii) 两个训练臂 choice 均 > #3（0.49/0.49 vs 0.42）、noul #2 > #3（0.629 vs 0.522）而 **#1 < #3**（0.478 vs 0.522）——训练过的共享打分器在二元判断上竟不如冻结小 LM 的候选似然；(iii) 绝对值温和（choice ~0.49 近该类任务多数类基线），这是 37 任务 joint、2520 updates、512 窗口的 stress regime 量级标定，专用训练锚点见 §6。

### 5.2 #1 价值的三个操作化检验

**(a) 同族零样本**（held-out GUE 16 + ProteinGym 20，训练不含）：trained ≈ frozen base（GUE acc 0.496 vs 0.515，Δ−0.018；逐任务 11 胜/9 平/13 负=wash）。联合训练 37 任务**没有**学到强迁移的通用决策函数。

**(b) 跨族零样本**（训练整族留出 GenomicBenchmarks，评其 8 任务；3 seeds）：mean Δ=**+0.036±0.034**（s1 +0.038/s2 −0.006/s3 +0.077），≈ in-domain 直接训练 GB 的 +0.035——弱正、量级与 in-domain 相当，暗示收益来自通用决策改进而非族特定拟合；但 std 与 mean 同量级、逐任务 −0.19～+0.31 不一致。score 族（TAPE-holdout，训练含 DeepSTARR/ProteinGym 两个 score 族，**3 seeds**）：fluorescence Δ=**−0.056±0.042**（三 seed 一致负）；stability Δ=+0.159±0.086 但 trained 绝对值 mean 仍为 **−0.059**——表面增益全部来自 frozen 病态基线（−0.218）向 ~0 回归，**score 无真实跨族迁移**（3-seed 坐实）。

**(c) 适配成本**（冻结 #2 训过 37 任务的 encoder，为 held-out 任务新建头，N 条标注线性拟合；11 任务=6 GUE+5 PG）：#2 新头 **N=32 即追平或超过 #1 零样本（7/11 任务）**；#1 零样本仍 stronger 的仅 3/11（且 #2 到 N=512 未追上）；1 任务需 N=512。GUE 族：#1 零样本 0.476 vs #2@32 = 0.511；PG 族：−0.024 vs 0.095。32 条标注的成本极低——**"零适配成本"卖点未转化为有意义的性能/成本优势**。

**(d) noul 措辞×稳定 checkpoint（强化完成）**：受控消融（同一 checkpoint、唯一变量=候选文本）在 3 个重训 seed-avg 稳定 checkpoint（s1r_2026100×`best_GenomicBenchmarks`，raw no/yes 训练）上重复，6 措辞×3 seed：

| 措辞 | mean±std（3 ckpt） | vs #2=0.629 |
|---|---|---|
| **false: no / true: yes（旧前缀式）** | **0.553±0.023** | 未达 |
| false / true | 0.543±0.018 | 未达 |
| no / yes（训练措辞） | 0.508±0.024 | 未达 |
| proposition（"yes, it does"） | 0.526±0.016 | 未达 |
| descriptive / neg-pos | 0.488 / 0.484 | 未达 |

三个发现：**(i) 无措辞反超 #2**——最优 0.553 仍低 0.076，C3 的 noul 结论不是措辞 artifact，反而加固；**(ii) 措辞效应方向依赖训练 run**——旧 checkpoint（bv2_pf，旧措辞训练）上 raw 胜 +0.066，新 checkpoint（raw 训练）上旧措辞胜 +0.045（3/3 seed 一致）：单 checkpoint 的"干净受控证据"不跨训练 run 迁移，**推理级消融同样需要多 checkpoint 确认**（方差主张在消融层面的延伸）；**(iii)** s1 重训的 raw 评测 0.477 与原 seed-avg 报告 0.478 几乎一致，重训复现性良好。

### 5.3 训练动力学

**预算缩放负结果**：同 43 任务集，2520→5600 updates（+数据 768→2048/任务）后分类 acc 0.485→**0.421（跌破 init 0.463）**、TAPE spearman 0.237→−0.020。诊断：train loss ~900 步后平台（1.06）、grad 中位 0.25 持续 4700 步、首段 grad_max 547；fold train loss 全程 ≈ln7（根本没学）——**联合优化失稳/早停于差 basin，非过拟合**（train loss 未→0）。

**稳定化修复+平台确认**：warmup 0.05→0.15、clip 1.0→0.5、dev 早停（每 200 步、best-dev promote）后，同 5600 预算恢复到 0.481（≈小预算 0.485，gain vs init +0.049≈+0.050）。结论：**~2500 updates 后算力非瓶颈**；天花板是 joint 多任务的容量/干扰。

**按族早停（零推理成本改进）**：同一训练轨迹，族峰值 best step 跨 **26 倍**（GUE 200 → local_snapshots 5200），全局单 checkpoint（step 2200）对 GUE 偏晚、对 local/DeepSTARR 偏早。改为每族各自 best-dev checkpoint 后**全族 ≥ 全局**：local +0.106、dnagpt +0.105、GB +0.082、TAPE +0.073、GUE +0.053、DeepSTARR +0.028、PG +0.000（其 best step 恰近全局）；逐任务 21 胜/9 平/7 负。平台部分来自"异质任务共享单一 checkpoint 选择"的折中，而非纯容量上限。

### 5.5 全基准专项基线（CPT 臂，排行榜）

joint-37 是压力测试，不是基准上限。实用配方口径：每族单独训练、生物 CPT 编码器初始化、
同稳定化协议+按族早停、双臂（#1/#2）、单 seed（方差 caveat 同 §4.3）。已出三族：

| 族/任务 | #1 | #2 | 文献天花板 |
|---|---|---|---|
| TAPE fluorescence | 0.568 | **0.627** | 0.68 |
| TAPE stability | 0.049 | **0.433** | 0.73 |
| dna_core | 0.475 | **0.625** | — |
| dna_splice / dna_tf | 0.560 / 0.545 | 0.560 / 0.455 | 双臂坍缩如实报 |
| DeepSTARR dev/hk | 0.069 / 0.086 | 0.230 / 0.205 | PCC 0.68/0.74（512 窗口截断 ~2kb） |

三个发现：(i) 专项+CPT 恢复近 SOTA（荧光达文献线 92%）——§5.1 的低绝对值是 joint
小预算+no_cpt 基座的性质，不是任务不可学；(ii) **score 接口复现"头>共享打分器"**
（stability 0.433 vs 0.049），与 noul 结论同构，排除 joint 偏袒解释；(iii) 基准未饱和：
DeepSTARR、旧盲测 splice/tf、部分 PG assay 距文献远，有区分度与余量。
local_snapshots 已出（#2）：npp **0.860**、signal_peptide **0.855**、promoter **0.785**
（锚点 0.88–0.91 为单任务专用配方；此处 5 任务联合+单 seed）、fold 0.505（锚点 0.56–0.58）、
subcellular 0.415（10 类，弱，如实报）。GB 已出（#2）：noul AUROC coding **0.897** / worm **0.839** / nontata-promoter **0.834** /
dummy_mouse 0.783 / cohn 0.669 / ocr 0.557 / drosophila 0.426；regulatory(choice) 0.515。
GUE 已出（#2）：28 任务宏 acc **0.497**（最高 prom_core_notata 0.570）；virus_covid
0.115（1,754 隔离冲突组+对抗类平衡，如实报并脚注）。PG 已出（#2）：185/217 assay 可评
（512 窗口），宏 Spearman **0.012**（#1 为 −0.022；中位 0.017、最高 0.399、85 负）——专项+cpt 也推不动
单序列突变适应度（MSA 口径天花板 ~0.55），作为基准最大余量如实报告。7/7 族全入表 TABLE_7/8。

**单模型全基准（joint-241，一个 checkpoint 吃 233 任务，cpt，全局早停）**：#2 臂按族——
GB noul **0.698**（专项 0.715，代价 −0.02）、dnagpt **0.562**（+0.02 反升）、GUE 0.494（≈0）、
local **0.603**（−0.08）、TAPE 0.299（**−0.23 最大代价**）、DeepSTARR 0.176、PG 0.008。
判读：统一代价族依赖——分类/命题接口单模型几乎无损（GB noul 0.698 甚至高于 37 切片
0.629，任务多样性反助），score 小族被联合训练牺牲最多；"一个模型"答案=分类行、score 不行。
#1 单模型臂已出：GB noul 0.513、TAPE 0.067、local 0.516、dnagpt 0.402、GUE 0.485、
PG −0.007、DS 0.072——#2≥#1 在单模型 regime 全族成立（第四 regime 复现）。统一代价族依赖：
分类/命题近无损、score 小族牺牲最大。§5.5 增"统一代价"段；TABLE_7 四列（single×2+spec×2）。
**单任务档（5 任务，同预算量级，cpt）**：#2：promoter 0.845 / npp 0.835 / fold 0.440 /
fluor **0.671（文献线 99%）** / coding AUROC 0.892；#1：0.790 / 0.819 / 0.440 / 0.268 / 0.901。
梯度非单调：族内正迁移（npp/fold/coding 族≥单）、跨族干扰逐层递减（单≈族>全241>37切片，
promoter/fluor）——family-balanced+按族早停的直接证据。Fig5（附录）+ §5.5 梯度段。
摘要好数升级：0.671 荧光（99%）/0.845 promoter（单任务）+0.860 npp（单模型）。
摘要已加实用配方句：0.627 荧光（文献线 92%）/0.860 npp/0.785 promoter。

### 5.4 受控机制消融

**noul 措辞**（同 checkpoint 推理级，干净受控；3-seed 版见 §5.2(d)）：单 checkpoint（bv2_pf）上 raw no/yes 比冗余式 +0.066 AUROC，但 3 个 seed 平均稳定 checkpoint 上方向**反转**（冗余 `false: no/true: yes` 反而 +0.045、3/3 seed 一致），最优措辞 0.553±0.023 仍不达 #2 的 0.629。含义：共享打分器对候选**文本表面形式**敏感——既是弱点（须调措辞、措辞收益方向依赖训练 run）也是动态候选接口的独特自由度（固定头无此维度）；同时说明**推理级消融也必须多 checkpoint 确认**，单 run 的"干净受控证据"不跨训练 run 迁移。

**冻结 VLM 对比解码**（多模态试点，详见附录/另文）：frozen Qwen2.5-VL 对合成蛋白性质图的 plain 候选似然**全坍缩**到单一候选（acc 0.040<随机 0.143，1/7 类）——文本先验淹没图像；对比解码 `logprob(候选|图+问)−logprob(候选|仅问)` 打破坍缩（4/7 类、acc 0.140、3.5×），证明图像确实参与打分，但 0.140≈随机：合成渲染+frozen VLM 无真实信号。**对比解码应成为 frozen-VLM 候选似然评测的默认口径**（可复用方法论修复）；路线 A 本身否定，真实图像（HPA）与微调 VLM 留作独立工作。

## 6. 讨论与局限

**共享打分器何时有价值？** 本文证据边界内：已见任务不占优、零样本≈base、跨族弱正、适配成本被 ~32 标注追平。其**成立的优势**是接口性质本身：无任务参数、候选任意动态、新任务零适配即可推理（#2 架构上做不到）、措辞可作为推理期自由度（但 3-seed 消融坐实：最优措辞 0.553±0.023 仍 < #2 的 0.629，措辞调优**不能**反超任务头，见 §5.2(d)）。若部署场景是候选集极大/开放、任务频繁新增、真零标注冷启动，结论可能不同——这些是假设与未来工作，不是本文结果。

**绝对值与专用锚点。** joint-37 的 choice ~0.49、score ~0.01–0.02 远低于专用训练水平：同一 Laya 架构在 3 任务专用配方下 promoter 87–89%、fold 56–58%、GFP MAE 0.32–0.49（先前工作）。本文的 joint regime 是刻意的压力测试（一个 checkpoint 扛 37 异质任务、每任务 ≤2048 样本、~70 updates/任务），量级标定"共享决策接口在多任务小预算下的现状"，不代表架构上限。

**Threats to validity。** (i) 头对头为 37 任务切片（cap PG/GUE=12），非全 290——基准是基础设施贡献，对照是 matched 切片，两者口径分开声明；(ii) 3 seeds，SEM 跨任务、run-std 跨 seed，noul n=7/score n=9 任务样本小；(iii) #3 非 matched（frozen、异 backbone、截断、score 异口径）；(iv) 跨族 score 臂已补至 3 seeds（§5.2b），但 n=2 任务样本极小，stability 的表面 Δ=+0.159 依赖 frozen 病态基线口径（trained 绝对值为负），解释须带口径声明；(v) dev 选点噪声（96→256 已缓解，未消除）；(vi) GPU 非确定性不可消除，只能 seed 平均吸收；(vii) 512-token 预算排除长序列任务样本（报排除率，不静默截断）。

**数据与许可。** 全部来源的 license/修订号/SHA-256 入 receipt；RNAcompete 为 14-RBP 测试子集（非完整 ~200 RBP）；DeepGOZero/PEER/DeepSEA 因上游不可达未纳入（catalog 中保持 needs_work，不虚报覆盖）；lg_* 快照的 splice 标签约定存疑处已显式记录（§3.3）。

**伦理与 LLM 使用披露。** 基准全部来自公开数据集，无人类受试者数据；代码/数据可用性声明见补充材料。[venue 定稿时按 checklist 补 LLM 辅助写作披露]

## 7. 结论

BioDecisionBench 把 10 族 290 个生物序列决策任务统一到四类型化接口与一套防泄漏准入协议下（隔离后任务内泄漏 0），并配套可复现的构建/评测/训练代码。在其上的首个 seed 平均、匹配算力三方对照给出诚实答案：共享类型化决策打分器**不优于**每任务头（noul 显著劣、choice/score 持平、跨 seed 更不稳），其零样本与跨族迁移弱，"零适配成本"被 ~32 条标注的新头追平；它的真实价值是接口灵活性（动态候选、无任务参数、任意新任务可推理）与推理期自由度（措辞、按族 checkpoint 选择）。方法论上，本文以一个具体的结论翻转案例确立：该 regime 下单次 run 不可信（±0.05–0.1），seed 平均+run-std 报告+稳定选点是此类架构对照的最低要求。未来工作：全 290 任务分族训练、按任务早停、matched 生成式臂（微调 Qwen）、真实生物图像多模态（另文）、更大预算/数据下重测共享接口的迁移。

---

## 附录规划（LaTeX 版）

- A. 基准全统计表（族×primitive×模态×splits×隔离统计）+ 准入检查器输出摘要
- B. 按族×primitive 全数字（pf/pf2/seed1-3 各列）+ 逐任务 seed-mean 表
- C. 训练动力学扩展：预算缩放/稳定化 dev 曲线、按族 best-step 分布、loss/grad 轨迹
- D. noul 措辞消融全表（6 措辞×3 seed，`noul_ablation_3seed.json`）+ 跨族 score 3-seed 全表（`cross_family_tape_3seed.json`）——均已回填正文
- E. 多模态试点：渲染示例图、plain vs contrastive、路线 B 设计草案
- F. 复现说明：环境/包版本、每 run 的 config+trace 落盘路径、checkpoint 与评测子集 seed

## 图表清单（对应 PAPER_PLAN.md Figure Plan）

Fig1 hero（基准全景+seed 平均三方+方差内嵌）｜Fig2（迁移梯度+适配成本曲线）｜Fig3（预算-稳定性+按族早停）｜Fig4（措辞消融+VLM 对比解码，可入附录）｜Table1 基准统计｜**Table2 主表（seed 平均三方）**｜Table3 适配成本逐任务最小 N｜Table4 按族全数字（附录）
