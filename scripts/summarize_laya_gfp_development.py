#!/usr/bin/env python3
"""Audit and summarize the completed fixed GFP development run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    a = parser.parse_args()
    run = a.source / 'run'
    summary = read(run / 'summary.json')
    launch = read(a.source / 'launch.json')
    trace = lines(run / 'training.jsonl')
    curve = lines(run / 'learning_curve.jsonl')
    assert summary['complete'] and len(trace) == 512
    assert [r['step'] for r in trace] == list(range(1, 513))
    assert [r['step'] for r in curve] == list(range(0, 513, 64))
    assert all(np.isfinite(r[k]) for r in trace for k in ['loss', 'lr', 'gradient_norm'])
    train_ids = summary['selected_ids']['train']
    dev_ids = summary['selected_ids']['dev']
    assert not set(train_ids) & set(dev_ids)
    counts = Counter(i for r in trace for i in r['batch_ids'])
    assert set(counts) == set(train_ids) and set(counts.values()) == {16}
    for epoch in range(1, 17):
        ids = [i for r in trace if r['epoch'] == epoch for i in r['batch_ids']]
        assert sorted(ids) == sorted(train_ids)
    for name, expected in launch['source_sha256'].items():
        assert sha(a.source / 'frozen_code' / name) == expected
    assert sha(run / 'checkpoint/model.safetensors') == summary['checkpoint_sha256']
    best = min(curve, key=lambda r: r['dev']['scalar_prediction']['rmse'])
    assert best == summary['best']
    for entry in curve:
        for split, ids in [('train', train_ids), ('dev', dev_ids)]:
            rr = lines(run / f'{split}_step{entry["step"]:04d}_predictions.jsonl')
            assert [r['id'] for r in rr] == ids
            y = np.array([r['value'] for r in rr])
            pred = np.array([r['scalar_prediction'] for r in rr])
            assert np.isfinite(pred).all()
            assert abs(np.sqrt(np.mean((y - pred)**2)) - entry[split]['scalar_prediction']['rmse']) < 1e-12
    cpu = read(a.repo / 'artifacts/laya_task_diagnostics/input_and_cpu_controls.json')['tasks']['fluorescence']
    comparison = {'test_access': False, 'best_step': best['step'], 'final_step': 512,
                  'controls': {k: cpu[k] for k in ['position_ridge', 'train_mean_constant_dev', 'train_median_constant_dev']},
                  'initial': curve[0], 'best': best, 'final': curve[-1],
                  'checks': {'finite_updates': len(trace), 'complete_epochs': 16, 'source_hashes': True,
                             'checkpoint_hash': True, 'prediction_rmse_recomputed': True,
                             'best_checkpoint_selection': True,
                             'reload_probability_error': summary['reload_max_probability_error'],
                             'reload_scalar_error': summary['reload_max_scalar_error']}}
    dest = a.repo / 'artifacts/laya_gfp_development'
    dest.mkdir(exist_ok=True)
    shutil.copytree(a.source / 'frozen_code', dest / 'frozen_code', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for path in run.glob('*.json*'):
        shutil.copy2(path, dest / path.name)
    shutil.copy2(a.source / 'launch.json', dest / 'launch.json')
    (dest / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    epochs = [r['epochs'] for r in curve]
    for ax, metric, label in zip(axes.flat, ['rmse', 'mae', 'spearman', 'accuracy'], ['Scalar RMSE (lower is better)', 'Scalar MAE (lower is better)', 'Scalar Spearman rho', 'Five-bin accuracy']):
        for split, color in [('train', '#167D9A'), ('dev', '#C85B35')]:
            values = [r[split][metric] if metric == 'accuracy' else r[split]['scalar_prediction'][metric] for r in curve]
            ax.plot(epochs, values, 'o-', label=split, color=color, markersize=4)
        if metric in ['rmse', 'mae', 'spearman']:
            ax.axhline(cpu['position_ridge']['dev'][metric], linestyle='--', color='#666666', label='Position ridge dev')
        ax.axvline(best['epochs'], color='#999999', linestyle=':', label='Selected checkpoint')
        ax.set(xlabel='Training epochs (32 updates each)', ylabel=label)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('GFP: 1,024 train / 128 reused development variants; one seed')
    fig.savefig(dest / 'learning_curves.png', dpi=180)
    fig.savefig(dest / 'learning_curves.pdf')
    plt.close(fig)
    def row(label, metric):
        rho = metric['spearman']
        return f'| {label} | {metric["mae"]:.5f} | {metric["rmse"]:.5f} | {rho:.5f} |' if rho is not None else f'| {label} | {metric["mae"]:.5f} | {metric["rmse"]:.5f} | 未定义（常数） |'
    table = '\n'.join([row('训练均值常数', cpu['train_mean_constant_dev']), row('训练中位数常数', cpu['train_median_constant_dev']), row('位置 one-hot 岭回归', cpu['position_ridge']['dev']), row('本轮初始化', curve[0]['dev']['scalar_prediction']), row(f'本轮开发集选定（step {best["step"]}）', best['dev']['scalar_prediction']), row('本轮最终（step 512）', curve[-1]['dev']['scalar_prediction'])])
    evolution = '\n'.join(f'| {r["step"]} | {r["epochs"]:g} | {r["train"]["scalar_prediction"]["rmse"]:.5f} | {r["dev"]["scalar_prediction"]["rmse"]:.5f} | {r["dev"]["scalar_prediction"]["mae"]:.5f} | {r["dev"]["scalar_prediction"]["spearman"]:.5f} | {r["dev"]["accuracy"]:.2%} |' for r in curve)
    conclusion = ('本轮所选模型的开发集 RMSE 未超过位置岭回归对照。' if best['dev']['scalar_prediction']['rmse'] >= cpu['position_ridge']['dev']['rmse'] else '本轮所选模型的开发集 RMSE 低于位置岭回归对照；仍需独立验证与重复 seed。')
    report = f'''# GFP 扩大训练集后的开发集验证

本轮固定使用 1,024 条训练数据与 128 条开发集数据，从原始 Laya 预训练权重重新训练 512 次更新（16 轮）。保留 typed transformer head 与 type embedding，采用 mean pooling、新建 LayerNorm、五档分类与连续标量联合损失。最佳开发集 scalar RMSE 出现在 **step {best['step']} / epoch {best['epochs']:g}**。完整预算已结束，无提前停止。{conclusion}

## 开发集结果

主要指标预先指定为连续标量 RMSE；五档分类及 anchor 期望值为辅助输出。表内历史 CPU 对照采用同一 GFP 划分，其超参数当时固定，未在本轮重新选择。

| 模型/对照 | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|
{table}

选定 checkpoint 的训练集 MAE / RMSE / Spearman 为 {best['train']['scalar_prediction']['mae']:.5f} / {best['train']['scalar_prediction']['rmse']:.5f} / {best['train']['scalar_prediction']['spearman']:.5f}。最终训练集对应为 {curve[-1]['train']['scalar_prediction']['mae']:.5f} / {curve[-1]['train']['scalar_prediction']['rmse']:.5f} / {curve[-1]['train']['scalar_prediction']['spearman']:.5f}。选定/最终模型在开发集上的预测标准差分别为 {best['dev']['scalar_prediction']['prediction_std']:.6f} / {curve[-1]['dev']['scalar_prediction']['prediction_std']:.6f}，真实开发集标准差为 {cpu['dev']['target_std']:.6f}。低方差预测的微小 RMSE 改善可能仅来自均值偏移，应结合排序相关性和训练误差解释。开发集挑选会使所报告的最佳值偏乐观；同时保留固定最终步结果。

## 完整学习曲线

| 更新 | 训练轮数 | Train scalar RMSE | Dev scalar RMSE | Dev scalar MAE | Dev scalar ρ | Dev 五档准确率 |
|---:|---:|---:|---:|---:|---:|---:|
{evolution}

![训练与开发集曲线](../artifacts/laya_gfp_development/learning_curves.png)

## 冻结的训练协议

- 单 seed 20260924；从原始预训练模型初始化，未使用前一轮 32 样本过拟合权重。全参数训练，模型可训练参数 {summary['trainable_parameters']:,}（保留全部任务头，但本轮仅 GFP 有训练样本）。
- 每轮独立打乱全部 1,024 样本，每条恰好出现 16 次；总曝光 16,384 次。前一轮 32 样本实验同为 512 次更新，但每例出现 512 次；本轮每例仅 16 次，不能把相同更新数当成相同的逐样本训练深度。有效 batch 32，micro-batch 4，BF16 autocast、FP32 权重，AdamW weight decay 0.01、梯度裁剪 1.0。
- 学习率 16 次更新线性 warmup 至 1e-4，随后余弦下降，step 512 为 1e-5。此设置在启动前冻结；与小样本轮次相比，样本规模、曝光次数和学习率日程均变化，不能单独归因到某一结构改动。
- 损失为 0.5 × 五档 CE + 0.5 × 标准化连续值 MSE；分箱、anchors、均值和标准差均沿用训练集定义。未用开发集拟合标准化参数。
- 初始及每 64 次更新分别全量评估训练集和开发集。预先按最小开发集 scalar RMSE 选权重，含 step 0、相同值优先更早一步；只保留一份选定权重，全部评估点及最终预测均保留。

## 数据边界与证据范围

读取固定 manifest 下的 train/dev 文件并校验 SHA-256。GFP 来源为 TAPE 的 train/valid，训练/开发之间 ID、完整 inputs 和 group 交集均为 0。本轮未读取 TEST。开发集在历史诊断中已使用，属于**重复使用的开发集**；GFP 变体共享母体蛋白，没有跨家族独立性的保证。单 seed、1,024/128 规模结果不是确认性测试，也不支持候选评分器与共享头的架构优劣结论。

## 验证、耗时与产物

全部 512 次 loss/梯度有限；逐轮批次核对确认无遗漏/重复，每例恰好 16 次。全部 9 个评估点的预测 RMSE 已独立重算；最佳步选择与冻结源文件哈希通过。选定权重重载后概率最大差 {summary['reload_max_probability_error']:g}，标量最大差 {summary['reload_max_scalar_error']:g}。

训练、周期评估及加载累计 {summary['training_and_evaluation_seconds']/60:.2f} 分钟，峰值 allocated 显存 {summary['peak_allocated_gib']:.2f} GiB；显卡为 {summary['gpu']}。本地保留权重（仅模型状态，无优化器状态）：`{run / 'checkpoint/model.safetensors'}`，SHA-256 `{summary['checkpoint_sha256']}`。磁盘限制下不另存最终步权重（若与最佳步不同），最终步预测仍完整保留。

- [完整汇总](../artifacts/laya_gfp_development/summary.json)、[对照与审核](../artifacts/laya_gfp_development/comparison.json)
- [训练逐步记录](../artifacts/laya_gfp_development/training.jsonl)、[学习曲线数值](../artifacts/laya_gfp_development/learning_curve.jsonl)
- [运行配置](../artifacts/laya_gfp_development/run_config.json)、[冻结代码](../artifacts/laya_gfp_development/frozen_code/)
- [训练脚本](../scripts/laya_gfp_development.py)、[汇总脚本](../scripts/summarize_laya_gfp_development.py)
- [前一轮 32 样本拟合](laya_convergence.md)、[历史开发集与 CPU 对照](laya_task_diagnostics.md)
'''
    (a.repo / 'research/laya_gfp_development.md').write_text(report)
    (dest / 'README.md').write_text('Fixed 512-update GFP train/dev development validation. See [report](../../research/laya_gfp_development.md). All small evaluation predictions and frozen code are included; large weights remain local. Test data were not accessed. The initialization provenance is historical and its dev_access=false refers to its original tiny-fit stage; this run explicitly accesses development data in run_config.json.\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
