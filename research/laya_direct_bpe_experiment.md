# Laya 完整生物 BPE 扩表对照

本轮按用户确认的历史方法，复用 `/root/autodl-fs/omnigene_v2/scripts/vocab/trained_bpe/` 中的 DNA 20k / protein 8k BPE。保留原 Laya 文本 tokenizer，生物序列单独 BPE 切分后直接映射到新增模型 ID；新增 embedding 使用原始生物片段在原 Laya tokenizer 下的 embedding 均值初始化，并与编码器、候选评分相关权重共同做监督 CE。没有 CPT。

## 历史词表与修复

已找到原始三步脚本：采样约 1 GiB DNA、约 1 GiB protein，seed=42；BPE min_frequency=10；扩表时剥离模态前缀后用原始片段初始化。这比论文概述的原始语料 32/16 GB 更具体。原文件及脚本只读，快照和哈希随新表示保存；原始采样语料哈希见 `artifacts/laya_direct_legacy/historical_corpus_manifest.json`。历史词表使用外部语料，不能再称为本轮 train-only 词表；其语料与评测的重叠尚未建立。

发现旧 DNA BPE 缺 N，旧蛋白 BPE 缺 N/J，且 `unk_token=null`；旧蛋白编码 `MNA` 会丢失 N，得到 M、A，`MNNK` 甚至合并为 MK。新副本保留所有既有词条 ID 与 merge，仅追加缺失单字符 DNA N、protein J/N，不重新拟合，也不修改原文件。每条输入都检查 source piece 拼接及新 ID 逆映射能完整还原原始序列。

新增普通序列 token：DNA 19,999，protein 8,000，共 27,999；Laya 50,368→78,367。包括单字符，排除 source [UNK]/[PAD]。不添加本任务不使用的 3Di、DSSP 和生成控制符，故新增数量不等于旧 OmniGene 的 28,028。

输入实现不将 token 名称包装成文本再次分词。新增词条名称仅用于保存/解释 ID，自然语言上下文和候选一律由原 tokenizer 编码。保存 checkpoint 时同时保存完整 source BPE、映射、base/expanded tokenizer 和初始化来源，确保重新加载后能重建完全相同的输入 IDs。

## 固定对照

- `raw`：原始序列 + 原 Laya BPE。
- `full_bpe`：历史生物 BPE（补齐字母表）+ 显式新增 ID。
- 同一公开 Laya checkpoint，seed 20260922；重新独立初始化两条件。
- 沿用上一轮共同完整样本 ID：32,050 train、1,978 selection_dev、1,986 calibration。尽管新表示更短，本轮不纳入上一轮长度排除的记录，以保持样本一致。
- AdamW lr 2e-5、weight_decay 0.01、5% warmup/cosine、micro-batch8/accum4、3,005 updates/96,160样本处理次数；取固定预算最后 checkpoint。
- 训练候选排列按 seed+ID 固定哈希生成并重映射标签；评估用 canonical 顺序。
- 只评估 selection_dev/calibration，温度只在 calibration 拟合；test 性能保持未访问。
- 新增参数量和 token 长度不同，报告实际显存/时间，不能声称两者计算量相同。

## 验证与执行

6 项回归检查通过：缺失 N 修复、旧 merges/IDs 保持、裸片段初始化、原始输入与 upstream 一致、候选重排标签正确、自然语言编码隔离及超长输入拒绝。两条件分别完成两步 GPU 冒烟，均通过有限值、保存重载 IDs/logits 一致性检查；扩表模型两任务新 embedding 梯度均非零。冒烟不计为正式性能结果。

全量输入审计：`artifacts/laya_direct_legacy/audit/input_audit.json`。完整训练的日志与状态将由 `scripts/run_laya_direct_pair.py` 顺序管理；状态文件 `artifacts/laya_direct_legacy/pair_status.json`，任何有限值/重载/样本预算或冻结哈希失败都会停止队列。

旧 64-token 首轮及其 artifacts 保留，不能将其 M1 称为原始序列输入对照。本轮才直接比较 raw 与完整扩表。

全量审计完成：train 输入中位数 raw/full_bpe 为 173/96，p95 为 231/194，无截断；DNA 新 token 19,999 中有 18,448 个在本轮训练切分实际出现，protein 8,000 中有 7,972 个出现。未出现词条仍保留，不能声称每个新增 token 都获得了直接监督。

队列已于 2026-09-22 16:36:23 UTC 启动，先 raw 后 full_bpe。冻结清单见 `artifacts/laya_direct_legacy/frozen_pair/run_manifest.json`；动态状态见 `artifacts/laya_direct_legacy/pair_status.json`。正式结果尚未完成。

## 自动夜间监控与后续实验

用户已授权睡眠期间自动监控并继续。`scripts/laya_overnight.py` 已在独立 screen 会话 `laya-overnight` 启动，每 60 秒检查一次；使用排他锁避免重复排队。

固定流程：等待当前 seed20260922 的 raw/full_bpe 完成 → 检查冻结哈希、训练预算、有限值和重载 → 用逐样本 logits 复算指标并核对标签 → 顺序运行 seed20260923 与 20260924 的 raw/full_bpe（共额外 4 次训练）→ 每个完整 seed 对后更新报告 → 完成三 seed 后退出。新实验仍固定 3,005 updates/96,160 样本处理次数，不做 CPT、不解锁 test、不根据结果改变超参数。GPU 被其他进程占用时等待，任何训练或核验失败则停队列并保存原因。

监控状态：`artifacts/laya_direct_legacy/overnight_status.json`。监控日志：`artifacts/laya_direct_legacy/overnight_monitor.log`。冻结队列配置：`artifacts/laya_direct_legacy/overnight_manifest.json`。结果自动写入 `research/laya_direct_bpe_multiseed_results.md` 和 `artifacts/laya_direct_legacy/multiseed_results.json`；报告包含逐 seed 原始表、均值/样本标准差、配对差值和每类混淆矩阵/召回率。单 seed 不输出伪标准差或显著性结论。

部署前已用两份冒烟结果独立复算验证，且测试了 test_access、seed 错误、NaN logits、错误指标、重复样本 ID 五种失败情况，均正确拒绝。B1 固定头与序列依赖诊断留作三 seed 后的下一阶段，未擅自启动 test。

## 夜间实验已完成

监控于 2026-09-22 20:53:08 UTC 正常结束：3 个 seed × 2 个条件，共 6 次正式训练全部完成；每次均为 3,005 updates / 96,160 样本处理次数。再次核对冻结哈希并从所有预测 logits 复算指标，结果一致；没有 CPT 或 test 性能访问，GPU 已释放。

selection_dev 三 seed 均值±样本标准差：DNA Accuracy raw 89.73±0.48%，full_bpe 90.34±0.05%（配对平均+0.60个百分点，三个seed均正）；蛋白 Accuracy raw 62.31±1.41%，full_bpe 59.65±0.87%（配对平均−2.66个百分点，三个seed均负）；蛋白 Macro-F1 raw 53.69±4.37%，full_bpe 53.16±2.00%（配对变化方向不一致）。独立温度校准后 NLL，DNA raw/full 0.23169/0.24253，蛋白 1.00232/1.03864，完整扩表平均均较高。以上不是 test 结论或显著性检验。

训练循环平均耗时 raw 41.20 min、full_bpe 41.24 min；平均峰值 allocated 显存 raw 7.887 GiB、full_bpe 8.413 GiB。本轮虽缩短输入，但未观察到墙钟训练加速，显存反而增加约0.526 GiB。不能仅由 token 数推断实际训练效率。

完整结果见 [三 seed 自动报告](laya_direct_bpe_multiseed_results.md)。当前证据为任务相关的收益与代价：DNA 判别小幅提升，蛋白准确率下降，不能宣称全面优于原始 BPE。下一阶段为 B1 固定头和序列依赖诊断，再锁定 test 方案。当前没有自动运行新的实验。
