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

### 4.1 主对照（37 切片，8B QLoRA，3 seeds，matched 1 行/update×1680）——已回填
heads 0.483±0.011 / 0.564±0.026 / −0.006±0.022；shared 0.492±0.003 / 0.509±0.012 / 0.009±0.011
（choice/noul/score）。臂差：noul **+0.055（3/3 seed 同向：+0.099/+0.007/+0.059）**、choice −0.008、
score −0.016（后两者持平噪声级）。
**判决（按 PAPER_PLAN_2 决策规则）**：排序在第四 regime 复现 → 主结论=接口属性；但 noul 差距
随容量从 +0.151(423M no_cpt)/+0.132(423M cpt) 收窄到 +0.069(0.6B)/+0.055(8B) → 容量部分调节。
两句话都写：shared scorer 的 noul 劣势是接口属性（全 backbone 全 seed 同向），幅度随容量递减
（更大容量/数据或继续收窄——外推未测，future work）。
表：primitive × {#1Q, #2Q} mean±SEM + run-std + 逐任务胜负；并排 Paper1 423M 列。
判读分支：若 #2Q ≥ #1Q 保持 → 接口属性结论；若 #1Q 追平/反超 → 容量假说成立。
### 4.2 规模轴（heads 臂）——已回填
matched 低数据口径：noul 0.6B 0.605 → 8B 0.564（平坦/略降）；choice 0.489→0.483；score −0.042→−0.006。
参照（高数据，不 matched）：423M cpt 0.651 / 423M no_cpt 0.629。
**容量平坦（matched 低数据下 0.6B≈8B）= 数据受限 regime 的直接证据**；规模轴的收益要在数据量
放开后才显现（本队列 1680 行/族 刻意低数据以 matched）。TABLE_10 已生成（四 backbone × 双臂）。
- 423M enc no_cpt：choice 0.489 / noul 0.629 / score 0.018（3-seed）
- 423M enc cpt：choice 0.500 / noul 0.651 / score 0.056（3-seed）
- **Qwen3-0.6B LoRA（正式口径 1 行/update×1680，1 seed）**：heads 0.489 / 0.605 / −0.042；
  shared 0.490 / 0.536 / −0.005。旧 2 行×3360 配置 heads 0.547/0.681/−0.022、shared 0.497/0.498/0.038
  （数据量敏感性参照）。noul/choice heads≥shared 在第三 backbone 复现；score 臂间差噪声级。
  **口径 caveat**：跨 backbone 规模轴内部 matched（0.6B vs 8B 同协议）；423M 行来自 Paper-1
  协议（64 行/update×2520），仅作参照不参与 matched 对比。
- 0.6B frozen 候选似然（Paper1 #3）：choice 0.424 / noul 0.522
- Qwen3-8B QLoRA heads（暂态单 seed，旧 2 行/update 配置，将重跑）：choice 0.580 / noul 0.712 / score 0.005
- Qwen3-8B QLoRA 正式（1 行/update×1680 matched）：heads s1 = choice 0.476 / noul 0.596 /
  score 0.019；shared s1 = 0.493 / 0.497 / −0.002。**noul heads−shared = +0.099 @8B：排序在
  第四个 backbone regime 复现**（423M no_cpt +0.151 / 423M cpt +0.132 / 0.6B +0.069(matched 低数据)
  / 8B s1 +0.099）。heads s2 = 0.510 / 0.499 / 0.021——**8B noul 种子波动 0.097**（s1 0.596），
  2-seed 均值 0.548；排序结论须等 shared s2 后以 seed 均值判（单 seed 差不可信，自洽于本文方法论）。
  shared s2 = 0.497 / 0.492 / −0.006 → seed2 臂差 noul +0.007（s1 +0.099，同向不同幅）；
  8B 2-seed 均值差：noul +0.053 / choice −0.002 / score +0.024。跨 backbone noul 差距序列：
  +0.151(423M no_cpt) / +0.132(423M cpt) / +0.069(0.6B) / +0.053(8B, 2-seed)——随容量略收窄
  但不消失。heads s3 = 0.464 / 0.596 / −0.059 → **8B heads 3-seed：noul 0.564±0.026 / choice
  0.483 / score −0.006**。choice 各 backbone 持平；score 噪声级偏 heads。**容量平坦确认**：matched 低数据下 noul = 423M(cpt,高数据)
  0.651 > 0.6B 0.605 ≈ 8B 0.596——8B 不超 0.6B，限制因子=数据量（1680 行/族），非容量。**数据受限信号：8B heads noul 0.540 < 0.6B heads 0.605**
  ——同低数据口径下容量不再抬升，限制因子是每族 1680 行的数据量而非 backbone 容量；
  与暂态高数据 s1（0.712）对照 = 数据量效应 > 容量效应（本队列内证据）。
- 内存注记：non-reentrant checkpoint 的 backward 重算=整段带梯度 forward，峰值∝段内 token 数
  （8B ≈11MB/token）；shared 臂按 1800 token 预算把候选文本分块多段 forward 再拼 logits
  （PG score 5 候选×512 token 最坏case 10.6s/update 存活）
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

- 口径事故#2：sed 批量改预算时漏掉插旗行（8B 行 --qlora 夹在中间），8B 首跑跑到 3360 updates
  与 0.6B 的 1680 不 matched——发现即杀回重跑。教训：改 driver 预算后必须 grep 全文件逐行核对。

## 口径事故#3（2026-10-11 00:40）：双 sweep 并行污染

16:40 杀旧 sweep 时漏杀其 bash（只杀了 python 与一个 bash），旧实例以 bpf40 配置继续跑并
与 16:40 新实例（bpf20）**同写 q8_* 目录**：s2 heads 的 eval 被旧进程以 3360-updates 配置
覆写（20:38 读到的 0.540 版与 23:33 的 0.607 版即两次写入），s3 时间线亦不可信。
处置：全杀、q8_* 全删、两队列脚本加 flock -n 单实例锁、00:43 干净重跑（bpf20/mb1/OOM 守卫/
1200-token 分块）。q06 双臂处于单写者窗口（10:49–16:40 仅一个实例），保留。
教训：**杀队列必须 bash+python 全杀并验证 pgrep 归零；同目录多写者 = 数据不可信，宁可全重跑。**

## 5. Laya vs Qwen 同协议对照（A 实验，2026-10-12）

A：Laya 423M(cpt) 跑 Qwen 同款 matched 低数据协议（1 行/update×1680、全局早停、37 切片）。
3-seed 终值：shared 0.473±0.005 / 0.515±0.013 / 0.007±0.015；heads 0.476±0.002 /
0.519±0.015 / 0.011±0.026 → **noul 臂差 +0.004 = 完全持平**——423M 在极低数据下头臂优势消失，
而 Qwen 0.6B/8B 同协议仍 +0.069/+0.055。
交互结论：heads≥shared 的 noul 排序依赖 (i) 每任务数据量（423M 高数据 +0.13~0.15 → 低数据 ~0）
或 (ii) 强预训练先验（Qwen 因果 LM 的 few-row 头拟合靠先验撑住）。423M 全参微调头在
~45 行/任务下过拟合毁特征，LoRA 小增量 delta 保住先验——机制假设，待 B-lite 与正文表述校验。
规模轴补行：423M shared/heads @ matched = 0.501/0.520 noul，与 0.6B 0.536/0.605、8B 0.509/0.564
同档偏下——自研编码器在同协议不输因果 LM 的 shared 臂，heads 臂弱于 Qwen（先验差距）。
