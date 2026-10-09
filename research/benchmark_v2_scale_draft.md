# Paper 2 工作稿：容量与因果 backbone 能否拯救共享类型化决策打分器？

**状态：实验进行中（2026-10-09 起）。** 基准/协议/方法论全部复用 Paper 1
（BioDecisionBench，防泄漏划分、seed 平均、按族早停、granularity ladder）。
本文变量 = backbone 容量与类型：423M 掩码编码器（Paper 1 数字直接引用）→
Qwen3-0.6B（LoRA）→ Qwen3-8B（LoRA bf16）→（可选）Gemma-4-31B-QAT（仅 frozen 推理点）。

## 摘要（占位，数字待回填）

[待 8B 主对照 3-seed 出数后写：一句话贡献 = 排序对容量稳健 or 反转；
锚点数字 = 8B noul AUROC 两臂 + 与 Paper1 423M 的差值。]

## 1. 引言

- 钩子：Paper 1 在 423M 掩码编码器上发现"任务头 ≥ 共享打分器"（四 regime 复现）。
  该排序是**接口属性**还是**小容量伪影**？
- 缺口：typed-decision 接口的所有公开报告都在小模型/单 run 上；容量轴从未受控扫描。
- 方法：同基准 37 任务 matched 切片、同协议（3 seeds、SEM+run-std、按族早停），
  backbone 换因果 LM + LoRA；双臂端口：#1Q = 候选尾 [MARK] 隐状态共享标量打分；
  #2Q = 池化隐状态每任务头。
- 贡献（占位）：S1 因果 backbone 三臂可实现且 matched；S2 排序对容量稳健/反转；
  S3 规模轴增益曲线（frozen 0.6B → LoRA 0.6B → LoRA 8B）；S4 ladder 在 8B 复现；
  S5 机制修复跨 backbone 迁移（措辞/对比解码）。
- 结果预览：[待回填]

## 2. 相关工作

- Paper 1（第三人称自引）：基准+协议+423M 对照。
- 生物序列上的因果 LM 与 LoRA 微调谱系（Qwen3、Gemma 线、ESM 对照）。
- 容量 vs 多任务干扰文献（scaling laws、multi-task interference）。
- 编码决策 vs 生成决策（候选似然、对比解码、约束解码）。

## 3. 设置

- 基准与协议：引用 Paper 1 §3/§4（不重复，一段带过+差异声明）。
- 本文差异：backbone 换因果 LM（Qwen3-0.6B / Qwen3-8B），LoRA r16 α32 dropout0.05，
  targets=q/k/v/o/gate/up/down；bf16（8B 用 QLoRA NF4）；**手动 torch.utils.checkpoint
  包整个 forward**（该 transformers+peft 组合下模型自带 gradient_checkpointing 实测无效：
  无张量参数的 lambda 会被 torch 静默跳过；张量参数版 0.6B mb16 forward 峰值 1.65GB）；
  梯度累积 mb×accum=16 行/update（0.6B mb8×2、8B mb2×8→实测 22GB）；512 全输入不截断；
  渲染模板：`question\nSequence: ...\nAnswer: <candidate> [MARK]`（#1Q 取 [MARK] 隐状态
  过共享标量头；#2Q 无候选，池化隐状态过每任务头）；score 臂 #1Q 五锚点候选 softmax+期望值。
- 任务集：Paper 1 的 37 任务 matched 切片（s1_20261001 run_config 逐字复用）；
  8B 预算 1680 updates（bpf20×12ep×7fam），0.6B 2520（bpf30）；dev 早停每 150 步、dev≥256。
- 成本表必报（实测）：0.6B bf16 ~0.5s/update、15GB；8B QLoRA ~9s/update、22GB；
  每跑墙钟 ~4.5h（8B）/ ~0.5h（0.6B）；总 GPU 时 ~30h。
- 工程教训（附录可写）：argparse 后值赢导致 --model 被 COMMON 覆盖（"0.6B 跑"实跑 8B）；
  scipy 不收 bf16；checkpoint lambda 必须带张量参数；**first-N 采样子集在 qwen 训练器复现**
  （test 按类排序→单子类→AUROC=nan、dev 污染早停；Paper 1 已修过的坑在新代码复活）——
  评测/早停子集一律固定 seed 随机采样（20261001），与 Paper 1 同口径；共享臂图内存按
  "行数×候选数"计，两臂 matched 须按行/update 对齐（3 行/update）。

## 4. 结果

### 4.1 主对照（37 切片，8B，3 seeds）[待回填]
表：primitive × {#1Q, #2Q} mean±SEM + run-std + 逐任务胜负；并排 Paper1 423M 列。
判读分支：若 #2Q ≥ #1Q 保持 → 接口属性结论；若 #1Q 追平/反超 → 容量假说成立。
### 4.2 规模轴（heads 臂已有三点）
- 423M enc no_cpt：choice 0.489 / noul 0.629 / score 0.018（3-seed）
- 423M enc cpt：choice 0.500 / noul 0.651 / score 0.056（3-seed）
- **Qwen3-0.6B LoRA：choice 0.547 / noul 0.681 / score −0.022（1 seed，matched 2 行/update）**
- 0.6B frozen 候选似然（Paper1 #3）：choice 0.424 / noul 0.522
- Qwen3-8B QLoRA：[待回填，3 seeds]
读法：因果 LM 即使 0.6B 也在 choice/noul 上超过 423M 编码器（+0.05/+0.05），score 反而更差
（小因果模型的回归头弱）；规模轴在分类/命题上单调向上至 0.6B，8B 待验。
### 4.3 granularity ladder @8B [待回填]
5 代表任务 × {single, family, all-241}：正迁移/干扰结构是否随容量变化。
### 4.4 机制迁移 [待回填/可选]
措辞消融 × 8B shared（3 ckpt）；对比解码 × 8B 候选似然。

## 5. 讨论

- 容量救了什么、没救什么（按接口分：noul/choice/score 分别看）。
- 因果 vs 掩码 backbone 的接口适配差异（[MARK] 位置打分 vs 池化）。
- 局限：LoRA 非全参；8B 单卡 bf16 上限；Gemma 点缺失（许可）则声明。
- 与比赛/agent 方向的接口（若 Gemma-4 拿到：agentic 评测床展望）。

## 6. 结论 [待回填]

## 附录

- A. 成本表全量；B. 逐任务 8B 数字；C. 0.6B 全表；D. 渲染模板与 LoRA 配置原文。

## 实验队列实况

- [x] qwen_decision.py 双臂 0.6B 冒烟（heads 0.86 / shared 0.68+0.768 @120 upd）
- [x] batched forward 5× 提速；auroc 正类索引 bug 修复
- [ ] Qwen3-8B 下载（12/16.4G）→ 0.6B 低点双臂 → 8B × 2 臂 × 3 seeds（~18-20h）
- [ ] ladder @8B；措辞 @8B
- [ ] （条件）Gemma-4-31B-QAT frozen 推理点（待 Kaggle 条款）
