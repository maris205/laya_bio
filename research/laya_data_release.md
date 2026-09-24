# Laya-Bio 公开数据发布记录

补充发布：模型权重和历史语料现已公开，完整入口、固定版本及核验记录见 [权重与历史语料发布记录](laya_large_data_release.md)。下文保留首次数据包发布记录。

完成时间：2026-09-23T14:58:15Z。

- 公开数据集：[dnagpt/laya-bio](https://huggingface.co/datasets/dnagpt/laya-bio)。
- 固定版本：[3ff01111b721c14a07c541dcdb719523db575c09](https://huggingface.co/datasets/dnagpt/laya-bio/commit/3ff01111b721c14a07c541dcdb719523db575c09)。
- 数据包：248 个文件，161,962,521 字节（约 162 MB）。远端另有一个 `.gitattributes` 文件。
- 上传内容：论文两个任务的原始 JSONL、清理后数据与排除记录、论文实际样本范围的 Parquet、12 组模型预测及分析、BPE 资源与来源记录、训练和分析代码、论文 PDF/源码/图表。
- 本次范围：论文两个任务的数据和复现材料；未打包约 98 GB 的历史预训练语料、模型权重及无关任务数据。

## 数据配置

默认配置包含论文实际使用的两个任务，标签索引在每个任务内定义。

| 任务 | train | selection_dev | calibration | test |
|---|---:|---:|---:|---:|
| promoter_detection | 16,766 | 1,052 | 1,053 | 2,145 |
| fold_class | 15,284 | 926 | 933 | 1,952 |
| default 合计 | 32,050 | 1,978 | 1,986 | 4,097 |

另外提供 `cleaned_all`、`cleaned_promoter_detection`、`cleaned_fold_class` 三种完整清理视图；这些视图位于继承的长度资格筛选之前，不能与论文评测分母混用。

## 使用

```python
from datasets import load_dataset

data = load_dataset("dnagpt/laya-bio", revision="3ff01111b721c14a07c541dcdb719523db575c09")
dna = load_dataset("dnagpt/laya-bio", "promoter_detection", revision="3ff01111b721c14a07c541dcdb719523db575c09")
protein = load_dataset("dnagpt/laya-bio", "fold_class", revision="3ff01111b721c14a07c541dcdb719523db575c09")
```

## 发布核验

全部 248 个远端文件的大小和内容哈希与本地发布包一致；LFS 对象使用 SHA-256，普通 Git 文件使用 Git blob SHA-1。仓库公开状态及元数据均经匿名访问验证。

已实际执行固定版本的匿名 `load_dataset(..., token=False, split="test")`，加载 4,097 条测试记录；逐条比较全部字段，与本地 Parquet 完全一致，ID 无重复。DNA/蛋白质分别为 2,145/1,952 条。六种配置在发布前均已通过本地加载验证。

核验文件：`artifacts/laya_hf_release/upload_receipt.json`、`artifacts/laya_hf_release/public_loading_check.json`。发布内容清单：`releases/laya-bio/release_manifest.json` 与 `SHA256SUMS`。

令牌未写入发布包或登录缓存。仓库保留各来源组件的许可说明。测试标签与预测已公开，后续研究不得将其描述为未经使用的测试集。发布包中的论文 PDF 是上传前完成的工作稿；其公开发布状态以本记录及仓库说明为准。
