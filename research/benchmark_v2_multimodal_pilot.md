# Benchmark v2 多模态试点（路线 A）：frozen VLM 在合成性质图上退化

日期：2026-10-05。代码 `render_fold_images.py`（渲染）、`eval_generative_vlm.py`（frozen Qwen2.5-VL 候选似然）。结果 `artifacts/benchmark_v2_eval/vlm_fold/`、`data/07_multimodal/`。承接 `benchmark_v2_multimodal_scoping.md` 路线 A。

## 做了什么

- **渲染**：lg_fold_class 的 200 个 test 蛋白（无 3D 结构，只有序列）→ 合成"氨基酸理化性质热图"（8 性质 × 序列位置，viridis）。**明确标注为合成渲染**，非真实结构/显微图。
- **评测**：frozen `Qwen2.5-VL-3B-Instruct`，对每候选算 length-normalized logprob（图 + 问题 + 候选文本），argmax。同 #3 文本候选似然范式，只是多一张图。
- **对照**：同题 lg_fold_class 的序列版结果。

## 结果：图像版完全退化

| 版本 | 机制 | acc | 备注 |
|---|---|---|---|
| **图像 #4** | frozen Qwen2.5-VL 候选似然（合成性质图）| **0.040** | **200 张全预测同一类"Small Proteins and Peptides"**，macro_f1 0.011 |
| 序列 #3 | frozen Qwen3-0.6B 候选似然（文本）| 0.320 | ≈ 多数类基线 |
| 序列 #1 trained | 共享打分器（训过 43 任务）| 0.165 | full2 预算，fold 仅少量更新 |
| 序列 #1 frozen-init | base 未训 | 0.045 | |
| — | 多数类基线 / 随机 7 类 | 0.325 / 0.143 | |

图像版 0.040 **低于随机（0.143）、远低于多数类（0.325）**：模型完全忽略图像、坍缩到文本先验最强的候选。

## 校验：不是坏图

渲染图内容正常（mean~150、std~90、每图 244–258 唯一色，非全黑/模板）；20 张两两相关 0.786（同 8 行性质布局 + 同色图 → 视觉偏相似，但确含逐蛋白变化）。故退化是 **frozen VLM 不用图像信息 + 候选似然被文本先验主导**，非渲染失败。

## 判读（诚实的负结果）

1. **路线 A（合成性质图）+ frozen VLM 候选似然 = 死路**：对 fold 这种即便序列版也难（trained 0.165、frozen 文本 0.320≈多数类）的任务，把序列渲染成合成热图再让 frozen VLM 读，信号全失，输出坍缩到单一候选。
2. **frozen 候选似然在图像上的机制缺陷**：与 #3 文本版一样，候选 logprob 被候选文本自身的先验概率主导（"Small Proteins and Peptides"这类短语 token 先验高），图像几乎不改变排序 → argmax 恒定。这不是 VLM"看不懂图"，而是**候选似然解码对图像不敏感** + 合成图信息量低。
3. **合成渲染的固有局限**（scoping 已预警）：性质热图不是 VLM 预训练见过的自然图像分布，frozen VLM 无对应先验。

## 对多模态方向的结论与下一步

- **本试点否定了"路线 A + frozen VLM 候选似然"这条最省事的组合**。要得到有意义的多模态数据点，需至少其一：
  1. **路线 B（真实生物图像）**：HPA 亚细胞定位免疫荧光图（与 C06/C11 定位同标签体系）——真实图像分布，VLM 预训练更接近；但需从 HPA portal 下载、核对许可。
  2. **微调 VLM**（matched）：在图像任务上 fine-tune VLM 而非 frozen 候选似然——工作量大（7.5GB 模型训练）。
  3. **改解码**：不用候选 logprob，改用受约束生成 + 解析答案，或对比式打分（图 vs 纯文本基线的似然差），降低文本先验主导。
- 基础设施已就绪（渲染器、VLM 评测器、统一 image 记录格式 `modality:["image"]` + `images:[{role,format,path}]`），换路线 B 或改解码即可复用。

## 主张边界
单一任务（fold）、单一合成渲染、frozen VLM、200 样本、单次。这是**可行性试点的负结果**，用于否定一条路线，不代表"多模态生物决策不可行"——真实图像（路线 B）或微调 VLM 未测。

## 补充：对比解码（消除文本先验坍缩）——机制修复但信号仍缺

针对 plain 候选似然"全坍缩到单一候选"（文本先验主导），加**对比解码**：`score = logprob(候选|图+问) − logprob(候选|仅问)`，隔离图像的净贡献。`eval_generative_vlm.py --contrastive`，同 200 图重跑。

| 解码 | acc | macro_f1 | 用到的类 |
|---|---|---|---|
| plain | 0.040 | 0.011 | 1/7（全坍缩到"Small Proteins and Peptides"）|
| **contrastive** | **0.140** | 0.052 | 4/7（Alpha and Beta 107 / Mixed 84 / All Beta 8 / Alpha+Beta 1）|

**判读**：
1. **对比解码修复了机制**：预测不再坍缩到单类、跨 4 类分布、acc 提升 3.5×（0.040→0.140）——证明图像现在**确实影响**打分（plain 下图像被文本先验淹没）。这是可复用的方法论改进（frozen VLM 候选似然评测应默认用对比解码）。
2. **但 0.140 ≈ 随机(1/7=0.143)**，仍远低于多数类(0.325)与序列版 frozen-text Qwen3(0.320)。即 frozen VLM 对"合成性质图→fold"**无真实判别信号**。
3. **失败根因是信号/grounding，不只是解码**：合成性质热图超出 VLM 预训练自然图像分布 + fold-from-sequence 本就难（序列版 trained 也仅 0.165）。对比解码是**必要修复但不充分**。

**修正后的多模态结论**：路线 A（合成渲染）+ frozen VLM 即使加对比解码仍 ≈ 随机，确认是死路。要有意义的多模态数据点必须换 **路线 B（HPA 真实定位图，VLM 预训练分布内）** 或 **微调 VLM**。对比解码作为评测方法保留。
