# 固定生物 BPE 后的分类接口与序列依赖验证

用户已确认将完整生物 BPE 保留为基础功能。当前主线固定历史 BPE（补齐 N/J、显式片段到 ID 映射）及 embedding 均值初始化，不继续搜索词表大小。保留之前 raw/full_bpe 的三 seed 结果，扩表收益仍按任务如实报告。

## B1 固定类别读出

B1 保留原 Laya 的 ModernBERT encoder、两层上下文 head、type embedding 和 scorer 的 LayerNorm/Linear/GELU，共享与候选模型相同的预训练权重。将最后的候选标量打分改为 CLS pooling 后的两个固定输出层（DNA 2类 / protein 7类）；新输出层按 seed 随机初始化。固定指类别槽位固定，分类层权重仍参与监督训练。

B1 输入为 `[CLS] Task context: ... Sequence: {直接生物BPE IDs} [SEP]`，不输入候选标签字符串。任务身份通过已知类别数选择固定输出层；目标为 canonical 类别编号。参数量与候选模型相近（共享原 scorer MLP，只替换末层），但读取位置、输入格式和末层初始化不同，因此不把结果当成只改变一个数学运算的严格因果消融。

## 训练过的无序列控制

text_only 保留完整候选接口及同样的 BPE 配置，在 train/dev/calibration 全部移除序列。它用于测量任务模板和类别分布能达到的成绩。新增序列 embedding 在该控制中不会获得直接序列监督。这与对已训练模型推理时移除序列是不同的实验。

## 共同训练预算

- seed 20260922、20260923、20260924；B1/text_only 各3次，共6次正式训练。
- 与 full_bpe 候选模型完全相同的 32,050 train、1,978 selection_dev、1,986 calibration，固定样本 ID 和行序。
- AdamW lr2e-5、weight decay0.01、5% warmup/cosine、BF16、microbatch8×accum4、3,005 optimizer updates / 96,160 样本处理次数。
- 从原始公开 Laya checkpoint 独立初始化；扩表仍是50,368→78,367，原始生物片段 embedding 均值初始化。不继承已监督训练的 candidate 权重。
- 只冻结不用的组件；encoder、表示 head、读出以及 embedding 联合监督训练，不做 CPT。
- 固定预算最后 checkpoint；校准温度仅在 calibration 拟合；不访问 test 性能。
- 每次检查有限值、无截断、保存重载后的输入 ID 与 logits 一致性，并用保存预测独立重算指标、核对标签与成员名单。

## 已完成的推理诊断

对 raw/full_bpe 的三个已训练 seed，共六个模型，执行原始预测复现、序列移除、三份确定性字符组成保持的随机重排、候选重排。候选重排后将 logits 映射回 canonical 类别顺序再比较概率和预测。温度使用原 calibration 结果，不在扰动数据上重拟合。

每条重排序列保留原长度及字符计数；重排通常改变真实生物含义。因此只称扰动后指标为“对原标签的匹配率”，不声称标签不变或把它当成生物泛化性能。原预测与扰动预测之间的分布/类别一致性是主要诊断之一。

六套诊断均已完成，原始 logits 全部复现。完整 BPE 三seed的 DNA 原始 Accuracy 90.34%，移除后原标签匹配率49.84%，重排后66.12%；蛋白对应59.65%、8.42%、53.79%。移除后崩溃包含分布扰动效应，需要训练过的text_only对照解释。完整 BPE 候选重排预测一致率 DNA98.16%、protein90.96%，提示候选位置敏感性。详细原始和校准指标/预测保存在 `artifacts/laya_controls/sequence_diagnostics/`。

## 自动执行与交付

4项 CPU 回归检查通过：混合任务固定头梯度/掩码，候选排列逆映射，字符重排完整性，text_only无序列泄漏及B1保留完整序列。GPU 冒烟通过后，`scripts/run_laya_controls.py` 将在独立 screen 会话中按相同种子顺序运行B1/text_only，每60秒记录状态，失败停止。运行哈希在 `artifacts/laya_controls/run_manifest.json`，状态在 `artifacts/laya_controls/status.json`，日志在该目录logs/。

报告自动写入 [laya_controls_results.md](laya_controls_results.md)。本轮完成后自动停止，test继续锁定。

启动前 GPU 冒烟已通过，且保存预测经独立标签/指标复算验证。B1 可训练参数449,709,065，对应候选模型449,700,865，差8,200；B1/text-only 冒烟峰值 allocated 显存约8.40 GiB。正式队列已部署到独立 screen 会话 `laya-controls`，每60秒监控；具体进度以status.json为准。

## 训练期间的独立 CPU 补充审计

确认每任务 train/selection_dev 各只有一个相同的 context+canonical候选模板。按训练集多数类选择的固定预测器在开发集上为 DNA49.52%、protein29.91%，作为无序列训练的参考，未用dev挑选类别。

完整 BPE 的蛋白候选重排：平均9.04%预测改变，其中3.49%正确→错误、3.38%错误→正确、2.16%错误→另一错误，总Accuracy仅改变−0.11个百分点。因此平均准确率接近不能掩盖逐样本不稳定。详见 [模板与候选位置审计](laya_reliability_audit.md)；本审计不修改正在运行的冻结实验。

## 正式对照完成与核验

队列于2026-09-23 08:01:20 UTC正常结束：B1/text_only各三个seed，共六次训练，均完成3,005 updates和96,160样本处理次数。再次验证冻结哈希、预测标签/成员名单、逐样本logits重算指标和重载一致性，全部通过。后台退出，GPU空闲。

开发集均值±样本标准差：候选模型/B1/text_only的DNA Accuracy分别为90.34±0.05%、89.01±0.86%、49.84±0.55%；蛋白Accuracy为59.65±0.87%、52.34±5.12%、29.91±0.00%；蛋白Macro-F1为53.16±2.00%、39.13±4.00%、6.58±0.00%。候选模型相对本版B1在三个配对seed上，DNA与蛋白Accuracy均更高；平均差值+1.33和+7.31个百分点，蛋白Macro-F1平均+14.03个百分点。该结论限定当前输入、固定预算和B1读出设计，不作显著性或普遍优越性声明。

text_only开发集预测已逐条检查：每个任务/seed均输出一个固定类别；蛋白全部选择训练集多数类别2，DNA分别选择0/1/0。结果与模板/类别比例参考一致。该证据加上序列扰动支持模型利用序列信息，但不证明因果生物机制。

候选位置敏感性仍存在（蛋白平均约9.04%预测随排列改变）；不能用总体Accuracy近似不变消解逐样本不稳定。BPE保留为功能配置，其增益和代价沿用之前raw/full_bpe比较，不将本轮候选接口优势归因于词表扩展。

当前全部成绩仍是selection_dev；同源独立性未确认，test仍锁定。下一阶段固定最终对比条件/报告协议后再执行test，保留候选位置敏感性作为可靠性限制。此时没有新训练任务自动启动。
