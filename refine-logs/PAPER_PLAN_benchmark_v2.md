# Paper Plan

**Title（主推）**: BioDecisionBench: A Leak-Proof Typed-Decision Benchmark for Biological Sequences, and a Seed-Averaged Comparison of Decision Architectures
**备选标题**:
- Does a Shared Decision Scorer Beat Task Heads? A 290-Task Biological Sequence Benchmark and a Seed-Averaged Answer
- BioDecisionBench: Benchmarking Noul/Choice/Score Decision Interfaces across DNA, Protein and RNA

**One-sentence contribution**: We release BioDecisionBench — a 290-task, leak-proof, three-modality benchmark with a unified typed-decision interface (Noul/Choice/Score/multi-Noul) — and use seed-averaged matched-compute comparisons to show that a shared typed-decision scorer's advantage over per-task heads is qualitative (dynamic candidates, zero-adaptation-cost new-task inference) rather than quantitative, while single-run comparisons in this regime are too noisy (±0.05–0.1) to support any architectural conclusion.

**Venue**: ICLR（默认；**强备选：NeurIPS Datasets & Benchmarks track**——本文是"基准+诊断"型，D&B 更契合。定 venue 前与用户确认）
**Type**: Empirical / diagnostic benchmark paper
**Date**: 2026-10-06
**Page budget**: 9 pages（正文到 Conclusion 结束；refs/appendix 不计）
**Section count**: 7

---

## Claims-Evidence Matrix

| # | Claim | Evidence（文档 / 数据） | Status | Section |
|---|-------|------------------------|--------|---------|
| C1 | BioDecisionBench：290 任务、12 族、三模态（DNA/蛋白/RNA）、四接口统一记录格式；防泄漏准入（精确序列隔离+冲突隔离+旧-test 保留），隔离后任务内泄漏 0 | dataset_collection.md、build_protocol.md；manifest.json（290）、admission_report.json（leak 0、flagged 0）、isolation_report.json（1,984 冲突组隔离） | **Supported** | §3 |
| C2 | 方法论：单次 run 不可靠——同配置同 seed 重跑差异 ±0.05–0.1（score 0.107→0.015），足以推翻单次结论；seed 平均（≥3）是该类对照的最低要求 | noul_fix_and_variance.md（pf vs pf2，per-family best step 漂移表）；seed_avg_main_result.md（3-seed 平均推翻单次"#1≥#2"） | **Supported**（本文方法论脊柱） | §4.3, §5.1 |
| C3 | 主对照（seed 平均、matched compute）：#2 任务头 ≥ #1 共享打分器——noul AUROC 0.629 vs 0.478（SEM 0.05 不重叠）、choice 0.489 vs 0.486 持平、score 0.018 vs 0.013 持平；#2 跨 seed 更稳（noul run-std 0.030 vs 0.087）；两个训练臂均 > frozen LM 候选似然 #3 | seed_avg_main_result.md；seed_avg_three_way.json | **Supported**（3 seeds；noul n=7、score n=9 任务，须报 SEM） | §5.1 |
| C4 | #1 的价值是定性的：(a) 零样本新任务接口（#2 架构上做不到）存在但性能 ≈ frozen base（同族 GUE −0.018）；(b) 跨族零样本弱正 +0.036±0.034 ≈ in-domain +0.035，仅分类、score 无干净迁移；(c) #2 新头仅需 ~32 标注/任务即追平 #1 零样本（7/11 任务） | zeroshot_transfer.md、cross_family_transfer.md（GB 3-seed + TAPE 单 seed）、adapt_cost{,_summary}.json | **Supported**（跨族 score 臂仅单 seed → 标注 partial） | §5.2 |
| C5 | 训练动力学：naive 加大预算失稳（5600 upd 低于 init）；稳定化（warmup 0.15+clip 0.5+dev 早停）修复至小预算水平但确认平台（~2500 upd 后算力非瓶颈）；按族早停（推理期零成本）全族 ≥ 全局单 checkpoint（族峰值 step 跨 26 倍：200–5200） | budget_scaling.md、stabilization.md、per_family_earlystop.md；per_family_vs_global.json、stab dev_curve | **Supported**（stab/pf 为单轨迹证据 → 文中标明） | §5.3 |
| C6 | 受控机制修复：noul 候选措辞消融（同一 checkpoint，唯一变量=措辞）原始 no/yes +0.066 AUROC；frozen-VLM 候选似然需对比解码（plain 坍缩 1/7 类 → contrastive 4/7 类，acc 0.040→0.140），但合成渲染图仍 ≈ 随机（路线 A 否定，多模态另文） | noul_fix_and_variance.md §1、multimodal_pilot.md（含 contrastive 补充） | **Supported**（措辞=干净受控证据；VLM 部分放 appendix+Discussion） | §5.4, §6, App |

**已知弱点（写作时须主动处理）**：
1. 头对头只训/评 37 任务切片（cap PG=12/GUE=12），非全 290——必须显著标明"benchmark=基础设施发布；头对头=matched 切片"，防"基准没被用"批评。
2. 绝对性能低（choice ~0.49–0.51 近随机、score spearman ~0.01–0.11）：小预算+512 窗口+单共享 checkpoint 的 stress regime。Discussion 用 MVP1 专用训练锚点（promoter 88%/fold 57%，旧论文）说明同架构专用时能到高水平，joint-37 是刻意压力测试。
3. #3 非 matched（frozen、异 backbone、截断 120 字符、score 异口径）——只作"现成 LM 基线"。
4. noul n=7、score n=9 任务样本少；seed 仅 3；跨族 score 臂单 seed。
5. 与旧 Laya-Bio 论文（2 任务 4 条件）的关系：本文引用其为先前工作（匿名第三人称），贡献不重叠（旧=单任务适配可靠性；新=基准+跨架构对照）。

---

## Structure（7 节，9 页）

### §0 Abstract（150–250 词）
- **What we achieve**: 发布 BioDecisionBench（290 防泄漏任务、DNA/蛋白/RNA、Noul/Choice/Score/multi-Noul 统一接口）+ 首个 seed 平均、matched-compute 的"共享 typed-decision scorer vs 每任务头 vs 生成式候选似然"三方对照。
- **Why it matters / is hard**: 生物序列模型评测碎片化（各族各口径）、泄漏风险高（上游池含旧 test）；决策架构（共享 scorer vs 专用头）之争缺受控证据；小预算下 run 方差足以伪造结论。
- **How**: 统一记录格式 + 精确序列隔离/冲突隔离/旧-test 保留 + family-balanced 训练 + per-family dev 早停 + 3-seed 平均。
- **Evidence**: 37 任务 matched 切片、3 seeds；迁移/适配成本/预算缩放/稳定化四组补充实验。
- **Most remarkable result**: seed 平均后任务头 noul AUROC 0.629 vs 共享 scorer 0.478；而单次 run 曾给出相反结论（方差 ±0.05–0.1）；#2 新头 ~32 标注即追平 #1 零样本。
- **Self-contained check**: 摘要内不出现未定义缩写（Noul/Choice/Score 用一句话解释）。

### §1 Introduction（1.5 页）
- **Opening hook**: 一个模型能否用同一接口回答"这段 DNA 是启动子吗（Noul）/这个蛋白属于哪类折叠（Choice）/这个变异体适应度多高（Score）"？typed-decision 接口（动态候选、无任务头）被提出用于此，但从未在受控预算下与每任务头对比过。
- **Gap**: (i) 生物序列基准按族割裂、无统一决策接口、泄漏审计缺失；(ii) 共享 scorer vs 任务头 vs 生成式似然的架构对照不存在；(iii) 该 regime 下评测方差未被量化，单次 run 结论不可信。
- **One-sentence contribution**: 见上。
- **Approach overview**: 建基准（§3）→ 定协议（seed 平均+matched compute+按族早停，§4）→ 主对照+四组机制实验（§5）→ 诚实边界（§6）。
- **Key questions**: Q1 共享 typed-decision scorer 在 matched compute 下是否匹敌/优于每任务头？Q2 其"动态候选/零适配"卖点能否转化为可测优势（零样本/跨族/适配成本）？Q3 该 regime 的评测需要多少 seed 才可信？
- **Contributions**（4 条，与矩阵对应）:
  1. BioDecisionBench：290 任务防泄漏统一基准 + 准入协议（C1）；
  2. seed 平均协议 + 方差量化：证明单次 run 可推翻结论（C2）；
  3. 首个三方 matched 对照：#2≥#1、皆>#3；#1 优势是定性的（C3+C4）；
  4. 训练动力学与机制诊断：失稳→稳定化→平台；按族早停零成本增益；措辞/对比解码受控消融（C5+C6）。
- **Results preview**: 主表数字（noul 0.629 vs 0.478）+ 方差教训（score 0.107→0.015 单次漂移）。
- **Hero figure**: Fig 1（见 Figure Plan）。
- **Front-loading check**: 引言结束前读者已知 What/Why/So-what 与主数字。
- **Key citations**: Jev/typed-decisions、DNABERT-2(GUE)、ProteinGym、Miller(error bars)、Picard&Torchia、旧 Laya-Bio 论文（第三人称）。

### §2 Related Work（1 页，按方法族组织，禁止逐篇罗列）
- **生物序列基准与基础模型**: GUE/DNABERT-2、Genomic Benchmarks、TAPE、ProteinGym、FLIP、DeepLoc/DeepGO、ESM 系。定位：这些是"单族/单任务型"基准；本文是跨模态、跨接口的**统一决策**基准，且以泄漏审计为一等公民（上游池含旧 test 的处理协议）。
- **typed/candidate-scoring 决策接口**: TypeSafe Jev（文档/博客，谨慎引用）、NanoJev、GLiClass、SemIf、约束解码。定位：接口已有，但"共享 scorer vs 任务头"从无 matched-compute 对照——本文补上。
- **评测方法论与方差**: Picard & Torchia 2020（统计机制）、Miller 2024（evals 误差棒）、REVAL 类工作、seed 报告实践。定位：给出一个**具体翻车案例**（单次 run 结论被 seed 平均推翻）+ 可操作协议（≥3 seeds、报 run-std、per-family 早停选点噪声）。
- **探测与适配成本**: linear probing、few-shot head fitting。定位：把"零样本接口价值"操作化为"追平所需标注数"（~32/任务）。

### §3 BioDecisionBench: Design & Construction（1.5 页）
- **统一接口 schema**: `{task_id, primitive, modality, sequences:[{role,modality,sequence}], question, candidates|score_levels+anchors|labels, answer, group, split, provenance}`；四 primitive 定义与监督/评测口径（CE、AUROC/MCC、锚点期望值+spearman、逐标签 sigmoid 口径）。
- **来源与规模**: 12 族/290 任务表（Table 1）；GUE 28、GenomicBenchmarks 8、ProteinGym 217、RNAcompete 14、gene_lan 8、dnagpt 池 3+promoter、TAPE 2、DeepSTARR 2、DeepLoc 1、local snapshots 7；三模态、双序列（DNA+蛋白/蛋白+蛋白）。
- **防泄漏准入协议**（本节重点，一图流程）: 原生划分优先；上游池→旧 lg-test 逐字保留为盲测+池剔除旧 test 成员+标签映射经 pool∩test 全量重叠验证（splice 旧约定 4545/4545）；精确序列跨划分归最高优先级、冲突组隔离（1,984 组）而非多数投票；min-valid 门槛；不做 RC 合并的理由（链敏感任务标签会翻转）。
- **Score 锚点协议**: train 分位数派生 5 级、锚点不反灌 test。
- **诚实边界**: 290 按评测单元计；题型多样性按 12 族计（ProteinGym 217 同题型）——正文显式声明，防夸大。
- **Notation**: 任务 t、族 F(t)、primitive p∈{noul,choice,score,multi}、指标 m(p)。

### §4 Evaluation Framework（1 页）
- **三个被比架构**: #1 SharedDecision（Laya ModernBERT-large 423M，共享 scorer 打分动态候选 marker，无任务参数）；#2 同 backbone+每任务头（choice/noul=固定类 softmax、score=标量回归 MSE）；#3 frozen Qwen3-0.6B 全候选 length-normalized logprob（非 matched，明确标注）。
- **训练协议**: family-balanced interleave（防 217 PG 淹没多样族）、AdamW enc 2e-5/head 1e-4、warmup 0.15、clip 0.5、稳定化由来（§5.3 预算失稳）；预算 2520–5600 updates；512-token 全输入不截断、超预算排除并报率。
- **checkpoint 选择**: per-family dev 早停（dev≥256），同轨迹多 checkpoint，推理期零成本。
- **seed 平均协议**（方法论贡献）: ≥3 seeds、每任务先 seed 均值再跨任务聚合、报 SEM + run-std；动机=§5.1 前的方差发现（pf vs pf2）。
- **评测口径**: choice=acc/macro-F1；noul=AUROC/MCC；score=spearman（主）+归一化 RMSE（跨 assay 不可直接平均原始 RMSE 的说明）。

### §5 Results（2.5 页）
- **§5.1 主对照（Table 2 + Fig 1b）**: seed 平均三方表；逐任务胜负（#1 4/平 23/#2 10）；run-std 对比（#1 noul 0.087 vs #2 0.030）；方差案例（pf 0.107 vs pf2 0.015 score，同配置同 seed）作为 C2 的实证。
- **§5.2 #1 价值的三个操作化检验（Fig 2）**: 同族零样本（wash）、跨族零样本（GB 3-seed +0.036±0.034 ≈ in-domain；TAPE score 无干净迁移）、适配成本（#2 新头 N=32 追平 7/11；Table 3）。结论：定性优势成立、定量优势不成立。
- **§5.3 训练动力学（Fig 3）**: 预算缩放负结果（5600 不稳→0.421<init 0.463）；稳定化修复（0.481）+平台确认（≈2520 的 0.485）；按族早停（全族 ≥ 全局，峰值 step 200–5200 跨 26 倍）。
- **§5.4 受控机制消融**: noul 措辞（同 checkpoint +0.066，Table/Fig 4a）；（VLM 对比解码放 appendix，正文一句带过指向 §6）。
- **数据源**: seed_avg_three_way.json、cross_family_seedavg.json、adapt_cost_summary.json、zeroshot_transfer.json、per_family_vs_global.json、stabilization_summary.json、budget_scaling_summary.json、noul_ablation.json。

### §6 Discussion & Limitations（1 页）
- **何时共享 scorer 才有优势**（假设+本文证据边界）: 候选集极大/开放、任务频繁新增、真零标注部署、按族早停+措辞修复后（noul 理论可达 ~0.64 反超，需稳定 checkpoint 验证）；更大预算/数据下未知。
- **专用锚点**: 同架构 MVP1 专用训练达 promoter 88%/fold 57%（引旧论文）——joint-37 是 stress regime，低绝对值≠架构无能。
- **多模态路线 A 负结果**（一段）: 合成性质图+frozen VLM ≈ 随机；对比解码必要不充分；真实图像（HPA）/微调 VLM 另文。
- **Threats to validity**: 37 任务切片、3 seeds、#3 非 matched、dev 选点噪声、GPU 非确定性、512 窗口排除率、跨族 score 单 seed。
- **Broader impact / 数据许可**: 各源 license 已录 receipt；RNAcompete 为 14-RBP 子集等边界。

### §7 Conclusion（0.5 页）
- 重述（非复制引言）: 基准+协议+方差方法论+有界诚实结论。
- Future work: 全 290 任务分族训练对照、按任务早停、matched 生成式臂（微调 Qwen）、真实图像多模态（另文）、更大 backbone/预算下重测 C4。

---

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | **Hero**（双联） | (a) 基准全景：12 族×4 primitive×3 模态任务数矩阵/气泡 + 统一记录 schema 示例 + 准入流水线示意（raw→unified→isolation→admission）；(b) 主结果：三架构按 primitive 的 seed 平均条形图（choice acc / noul AUROC / score spearman，SEM 误差棒），内嵌小图"单次 run 方差"（pf vs pf2 vs seed-avg 的 score 0.107/0.015/0.013） | manifest.json、seed_avg_three_way.json、three_way_pf{,2}_summary.json | **HIGH** |
| Fig 2 | 双联条形/曲线 | (a) 迁移梯度：init / 同族零样本 Δ / 跨族零样本 Δ(3-seed±) / in-domain Δ；(b) 适配成本：#1 零样本水平线 vs #2 新头 N=32/128/512 折线（按族），标注 7/11@N=32 | zeroshot_transfer.json、cross_family_seedavg.json、adapt_cost.json | HIGH |
| Fig 3 | 双联 | (a) 预算-稳定性：init/2520/5600-unstable/5600-stabilized 条形+stab dev 曲线内嵌；(b) 按族早停：族峰值 step 时间轴（200→5200，26×跨度）+ per-family Δ(按族−全局) 条形（全 ≥0） | budget_scaling_summary.json、stabilization_summary.json、stab_dev_curve.jsonl、per_family_vs_global.json | HIGH |
| Fig 4 | 消融（可入 appendix） | (a) noul 措辞 6 条件条形（同 checkpoint）；(b) VLM plain vs contrastive（acc+用到类数）+ 1 张渲染图示例 | noul_ablation.json、vlm_fold{,_contrastive}/eval_results.json | MEDIUM |
| Table 1 | 基准统计表 | 族×primitive×模态×train/dev/test×隔离统计 | manifest.json、admission_report.json | HIGH |
| Table 2 | **主表** | seed 平均三方对照（mean±SEM + run-std + 逐任务胜负） | seed_avg_three_way.json | HIGH |
| Table 3 | 适配成本 | 逐任务"#2 追平 #1 零样本的最小 N" | adapt_cost.json | MEDIUM |
| Table 4 | Appendix | 按族×primitive 全数字（pf/pf2/seed 各列） | three_way_pf*_summary.json、s1/s2 各 seed 目录 | MEDIUM |

**Hero figure 详述（Fig 1）**: 左联让 skim 读者 3 秒看懂基准规模与统一性（"290 tasks · DNA/Protein/RNA · Noul/Choice/Score/multi"大字 + 族×接口矩阵）；右联直接给出论文最强对比——按 primitive 分组的三架构条形图，#2 在 noul 上明显高出 #1（误差棒不重叠），两个训练臂都高于 frozen #3；右联角落内嵌"方差警示"小图（同配置两次 run 的 score 0.107 vs 0.015 vs seed-avg 0.013），一眼传达方法论教训。**Caption 草稿**: "BioDecisionBench (left) unifies 290 leak-audited tasks across three modalities and four decision primitives. Right: seed-averaged (3 seeds), matched-compute comparison — per-task heads (#2) match or beat the shared typed-decision scorer (#1) and both beat frozen candidate-likelihood (#3); inset shows single-run variability that reverses naive conclusions."

---

## Citation Plan

**复用已验证条目**（laya_bio/paper/references.bib + citation_sources/）: warner2025modernbert、zhou2024dnabert(GUE)、rao2019tape、thumuluri2022deeploc、kulmanov2022deepgozero、wang2024llamagene、xu2023biotranslator、almeida2025chatnt、laya2026、almeida2026jev、wang2026biopaws(旧论文,匿名第三人称引用)。⚠️ scholak2021picard 是 text-to-SQL 的 PICARD，**不是** Picard & Torchia 2020 统计力学——citation_sources/picard.txt 需核对后新造条目 [VERIFY]。

**需新增并逐条验证（禁止凭记忆生成 BibTeX，写作时用 search/官方页核对）**:
- §3: ProteinGym（Notin et al. 2023, NeurIPS D&B）[VERIFY]、Genomic Benchmarks（Grešová et al. 2023）[VERIFY]、DeepSTARR（de Almeida et al. 2022, Genome Biol）[VERIFY]、RNAcompete（Ray et al. 2013, Nat Biotech）[VERIFY]、FLIP（Dallago et al. 2024）[VERIFY]、DeepSEA（Zhou & Troyanskaya 2015）[VERIFY]、PEER（Xu et al. 2022）[VERIFY]
- §4: Guo et al. 2017 温度缩放（citation_sources/guo.txt 已有源）[VERIFY key]
- §2/§4 方法论: Picard & Torchia 2020（J. Stat. Mech.）[VERIFY]、Miller 2024 "Adding Error Bars to Evals"（arXiv）[VERIFY]、REVAL（Mazaré et al. 2024）[VERIFY]
- §2 接口: GLiClass（knowledgator, misc/URL）[VERIFY]、NanoJev（C-Tianyu, misc）[VERIFY]、SemIf（TheoLeeCJ, misc）[VERIFY]
- §5/§6: ESM-2（Lin et al. 2023）[VERIFY]、Qwen3（Yang et al. 2025）[VERIFY]
- Appendix 多模态: Qwen2.5-VL（Bai et al. 2025）[VERIFY]、VCD（Leng et al. CVPR 2024）[VERIFY]、Contrastive Decoding（Li et al. ACL 2023）[VERIFY]

**规则**: 全部条目发表版优先；不确定一律 [VERIFY] 标记；匿名投稿——旧 Laya-Bio 论文以第三人称引用。

---

## Reviewer Feedback

**PENDING** — 本会话无 Codex MCP（mcp__codex__codex 不在工具列表），GPT-5.4 xhigh 交叉评审未执行。定稿前须补：走 `/auto-review-loop`（Codex MCP 可用时）或人工外审，重点让评审打：逻辑流、claim-evidence 对齐、缺失实验、定位、9 页预算可行性、front-matter 强度。

---

## Next Steps
- [ ] 与用户确认 venue（ICLR vs **NeurIPS D&B**——后者对基准+负结果更友好）与标题
- [ ] （可选补强，写作前）: 跨族 score 臂补 2 seeds；noul 措辞修复在稳定 checkpoint 上验证是否反超 #2（C4 强化）
- [ ] /paper-figure 生成 Fig 1–4（数据源均已落盘 JSON）
- [ ] /paper-write 起草（先中文结果版工作稿 research/benchmark_v2_paper_draft.md，再英文 LaTeX paper_v2/main.tex，沿用旧论文双轨范式）
- [ ] 引用逐条验证入 references.bib（复用 12 条 + 新增 ~15 条全验证）
- [ ] /paper-compile + 匿名化检查 + venue checklist
