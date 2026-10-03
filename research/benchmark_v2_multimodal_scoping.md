# Benchmark v2 多模态扩展可行性调研（生物图像 + 小型 VLM）

日期：2026-10-03。对应用户"用小型 qwen/gemma 多模态模型加图像题"的设想。本文是**调研与计划**，未实现（GPU 正跑 bv2_pf 训练）。结论先行：**模型不是瓶颈，数据是**——HF 上几乎没有与我们序列任务对齐的、干净成规模的生物图像决策数据集。

## 1. 候选 backbone（小型、可在本机 32G 单卡推理）

| 模型 | 大小 | 许可 | 备注 |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | 7.5 GB | Apache-2.0 | 240 万下载，动态分辨率，通用 VLM 强；首选 |
| `google/paligemma2-3b-mix-448` | 6.1 GB | 开放 | 448px，轻量 |
| `Qwen/Qwen2-VL-2B-Instruct` | ~4 GB | Apache | 更小更快 |
| gemma-3-4b multimodal | ~8 GB | Gemma 许可 | 备选 |

3B 级 VLM 的候选似然推理可复用 `eval_generative.py` 的范式（对每候选算 length-normalized logprob，argmax），只是 prompt 多一张图。显存/速度在 32G 卡上没问题。

## 2. 数据：真正的瓶颈

HF 搜索"cell microscopy / biology image / histopathology / science image QA / molecular structure image / bio VQA"命中的都是**小规模或偏医疗组织病理**的数据集（如 breast-histopathology、Single-cell Microscopy for Cancer Classification、nf-core/cellpainting），**没有**与我们的分子序列任务（启动子/结构类/定位/同源/荧光）对齐的图像决策集。直接拿这些会让"多模态 benchmark"与序列 benchmark 两张皮。

### 三条可行数据路线（按对齐度/自足性排序）

**路线 A（最推荐，自足）：从已有结构数据渲染图像。**
我们已有蛋白序列/结构类标签（lg_fold_class 7 类、DeepLoc 定位、TAPE 等）。用 `py3Dmol`/`matplotlib` 把蛋白渲染成 2D 接触图、3D 卡通图、或理化性质热图 → 图像版 Choice（"这张结构图属于哪类 fold？"）。标签直接复用序列任务，**图像与序列两模态共享同一套题**，正好做"同问题、序列 vs 图像"的跨模态对照。完全可控、无需外部大下载。

**路线 B（对齐度高，需外部下载）：Human Protein Atlas (HPA) 亚细胞定位免疫荧光图。**
HPA 的细胞染色图像带亚细胞定位标签，与我们 C06/C11（蛋白定位）同标签体系 → 图像版定位 Choice/multi-Noul。是最"生物对齐"的真实图像，但数据大、需从 HPA portal 下载（非 HF 一键）。

**路线 C（拓宽域，弱对齐）：公开细胞/组织图像。**
Cell Painting、细胞显微、组织病理等。能增加"细胞/组织"模态广度，但与分子序列任务不同域，只作 benchmark 的模态扩展，不参与跨模态对照。

## 3. 接口扩展（与现统一格式兼容）

统一记录加 `images: [{role, format, path|bytes}]`，primitive 仍是 Choice/Noul/Score：
```
{task_id, primitive, modality:["image"] 或 ["image","protein"],
 images:[{role:"query", path:"..."}], sequences:[...可选...],
 question, candidates, answer, split, group, provenance}
```
评测器加一个 VLM 候选似然路径（Qwen2.5-VL：图 + question + 候选文本 → 每候选 logprob → argmax）。跨模态对照 = 同 group 的序列版 vs 图像版准确率差。

## 4. 建议的最小可行试点（下一步，若用户确认）

1. 下载 `Qwen/Qwen2.5-VL-3B-Instruct`（7.5 GB，走 clash 代理；注意 xet 大文件用 curl 分块，见 Qwen3-0.6B 下载踩坑）。
2. **路线 A 试点**：从 lg_fold_class 取 ~200 蛋白，用 py3Dmol/matplotlib 渲染接触图 → 图像版 7 类 fold Choice（train/test 沿用现划分）。
3. 写 `eval_generative_vlm.py`：Qwen2.5-VL 候选似然评测，跑这个图像 fold 任务。
4. 对照：同题的序列版（#1 已测 fold acc ~0.54）vs 图像版 VLM 候选似然 → 第一个跨模态数据点。
5. 若试点通，再扩 HPA 定位（路线 B）做第二个图像任务族。

## 5. 风险与边界
- 渲染图（路线 A）是"合成图像"，VLM 在其上的表现不代表真实显微/结构图能力；须明确标注为合成渲染对照。
- VLM 是 frozen 候选似然基线（同 #3），非 matched-compute 训练对照；要 matched 需在图像任务上微调 VLM，工作量大。
- 生物图像数据许可/隐私（HPA、组织病理）需逐项核对，未核实前不纳入正式发布。
- 本轮只调研；实现待用户确认路线与预算。
