# Laya-Bio 词表来源与正式表示版本

更新：2026-09-22。本文记录本轮正式 M1/M2 表示使用的可验证来源，以及与历史 OmniGene 扩表产物的边界。

## 结论

`data/02_vocab/dna_bpe_20k.json` 和 `data/02_vocab/protein_bpe_8k.json` 是可读取的 BPE JSON 产物，但当前工作区没有它们的训练脚本、训练语料清单、语料哈希、训练日期或参数记录。`data/README.md` 只把它们描述为“训练好的 DNA/protein BPE 词表”，旧计划把来源写成 `biopaws/vocab/trained_bpe/`，这不足以证明它们是在本轮任务的 train 视图上拟合，也不能证明没有接触过评测数据。因此它们被视为**历史候选参考，不作为正式词表来源**。

正式表示由 [scripts/prepare_laya_representation.py](../scripts/prepare_laya_representation.py) 从当前固定的 `artifacts/laya_formal_data/*_train.jsonl` 重新拟合。脚本只读取两个 train 文件来拟合 DNA 20k 和蛋白 8k BPE、统计候选频率和选择新增 token；不读取 CPT 语料，不执行 CPT。运行产物和全部哈希记录在 [artifacts/laya_formal_representation/metadata.json](../artifacts/laya_formal_representation/metadata.json)。

## 可核查的历史产物

| 文件 | 大小 | SHA-256 | 可核查内容 |
|---|---:|---|---|
| `data/02_vocab/dna_bpe_20k.json` | 1,464,876 bytes | `7d61b94ed3023430227b8adc44f1bf2cd737e4c53fcd3252aff88d3502502425` | JSON BPE，20,000 个词条，19,994 条 merge；Whitespace pre-tokenizer |
| `data/02_vocab/protein_bpe_8k.json` | 504,546 bytes | `ae5391de0a8b824b0847c523e34214049e9121e16fe5ae143e9a8136c1f4f600` | JSON BPE，8,000 个词条，7,974 条 merge；Whitespace pre-tokenizer |
| `data/02_vocab/vocab_expansion_meta.json` | 199 bytes | `f208e0b0297f6f0895df0fe03b895557af31dbf9fa149fd54e9d4e904538bc2e` | 旧 OmniGene/Gemma 扩表汇总：262,144→290,172；不能恢复拟合语料 |

完整的 `data/02_vocab/expanded_tokenizer_reference.json` 与 `reference_checkpoint/tokenizer.json` 内容哈希相同（`7a1150e789379996f41858aed1ff5e72a2d34d8e50f16dce504c0859a007d144`），它是旧 Gemma 扩表参考，不能把其中 token ID 或 embedding 搬到 Laya。旧扩表元数据没有保存 DNA/protein BPE 的训练语料和脚本。

## 正式 train-only 版本

运行命令为：

```bash
python scripts/prepare_laya_representation.py --force
```

输入清单由 `artifacts/laya_formal_data/manifest.json` 固定。当前拟合行数为 DNA 16,766、蛋白 15,593，共 32,359 行。产生的 source BPE 文件为：

| 文件 | 训练行数 | SHA-256 |
|---|---:|---|
| `artifacts/laya_formal_representation/source_tokenizers/dna_bpe_20k.json` | 16,766 | `b23c7fd6d46868995ba0c437c66d76fb9b6f5ce7b850428485b8ff94e5e0c287` |
| `artifacts/laya_formal_representation/source_tokenizers/protein_bpe_8k.json` | 15,593 | `f9fc1e0799086462ca39e8f5f10f0a0bbde702ce9ea3ccc7156f9d56e67aad80` |

候选选择为每种模态 32 个。每个候选的排序分数是 train 中源 BPE piece 出现次数乘以“原 Laya tokenizer 对完整包装字符串的 token 数减一”；平局时按频率、节省量、piece 长度和字典序确定。频率、分数、扩展后 ID、扩展前分解 ID 全部写入 [new_tokens.json](../artifacts/laya_formal_representation/new_tokens.json) 和 `metadata.json`。新增 embedding 初始化时应使用 `metadata.json` 中的 `decompositions_before_mutation`，然后仅由监督 CE 更新。

## 精确 piece 隔离

早期 pilot 的裸 `AddedToken("▶GGC")` 存在边界问题：在输入 `▶GGCC` 时，tokenizer 会先匹配 `▶GGC`，再留下 `C`；当前 pilot 的 `◆ALA`/`▶GCC` 等 token 均能触发这种情况。它并不等价于源 BPE 的一个完整 piece。

正式版本对每个 source piece 加完整包围符号：DNA 为 `▶{piece}◀`，蛋白为 `◆{piece}◇`。M1 与 M2 使用完全相同的包装文本；M1 只没有对应的 AddedToken，M2 只对选中的完整包装串加入 AddedToken。由于候选包含关闭符号，候选不可能匹配更长 piece 的前缀。准备脚本对 9,681 个“候选是更长 source piece 前缀”的情况逐一检查，M2 没有出现错误 AddedToken 命中；所有 64 个新增 token 的 round-trip 也均为单一新 ID。

表示接口在 [laya_representation.py](../scripts/laya_representation.py) 中：

```python
from laya_representation import Representation

m1 = Representation.load("artifacts/laya_formal_representation", expanded=False)
m2 = Representation.load("artifacts/laya_formal_representation", expanded=True)
state_m1 = m1.state(record)
state_m2 = m2.state(record)
assert state_m1 == state_m2
```

`state(record)` 只改变序列的表示方式，不改写任务问题或候选文本。正式输入审计仍需报告 M1/M2 的 token 长度、截断率和完整序列覆盖；`build_sequence` 应在发现超过上下文预算时拒绝样本，而不是悄悄把完整生物序列截断后当作完整输入。

## 2026-09-23 新发现

用户提供 `/root/autodl-fs/omnigene_v2/scripts/vocab/trained_bpe/`，其父目录含三步原始脚本和采样语料。历史词表来源已补充，不再处于“找不到拟合脚本”的状态。新实验复用该词表，并在副本补齐缺失单字符；此前 64-token 实验的 train-only 词表与结果保持原样。详见 [完整 BPE 对照](laya_direct_bpe_experiment.md)。
