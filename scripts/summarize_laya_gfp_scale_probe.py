#!/usr/bin/env python3
"""Audit completed training-only scale/LR probes and archive a transparent report."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from laya_gfp_scale_probe import metric, probe_lr, sha_file, write_json


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    state = read(a.source / 'status.json')
    assert state['status'] == 'complete'
    for name, expected in state['source_sha256'].items():
        assert sha_file(a.source / 'frozen_code' / name) == expected
    for name, expected in state['reference_sha256'].items():
        assert sha_file(a.source / 'reference_training_only' / name) == expected
    arms = ['n32_high', 'n32_low', 'n1024_low']
    results, curves, traces = {}, {}, {}
    for name in arms:
        root = a.source / name
        s = read(root / 'summary.json')
        trace = lines(root / 'training.jsonl')
        curve = lines(root / 'learning_curve.jsonl')
        assert s['complete'] and len(trace) == s['updates']
        assert [r['step'] for r in trace] == list(range(1, s['updates'] + 1))
        assert [r['step'] for r in curve] == s['evaluation_steps']
        assert all(np.isfinite(r[k]) for r in trace for k in ['loss', 'lr', 'gradient_norm'])
        assert all(r['lr'] == probe_lr(r['step'], s['lr_scale']) for r in trace)
        counts = Counter(i for r in trace for i in r['batch_ids'])
        assert set(counts) == set(s['selected_ids']) and set(counts.values()) == {s['per_example_presentations']}
        for epoch in range(1, s['per_example_presentations'] + 1):
            assert sorted(i for r in trace if r['epoch'] == epoch for i in r['batch_ids']) == sorted(s['selected_ids'])
        assert sha_file(root / 'checkpoint/model.safetensors') == s['checkpoint_sha256']
        assert s['reload_max_probability_error'] == s['reload_max_scalar_error'] == 0
        for point in curve:
            rr = lines(root / f'train_step{point["step"]:04d}_predictions.jsonl')
            assert [r['id'] for r in rr] == s['selected_ids']
            assert metric(rr, s['task_info']) == point['training']
            assert metric([r for r in rr if r['id'] in s['anchor_ids']], s['task_info']) == point['anchor32']
        results[name], curves[name], traces[name] = s, curve, trace
    assert lines(a.source / 'n32_high/train_step0000_predictions.jsonl') == lines(a.source / 'n32_low/train_step0000_predictions.jsonl')
    assert [r['batch_ids'] for r in traces['n32_high']] == [r['batch_ids'] for r in traces['n32_low']]
    reference_trace = lines(a.source / 'reference_training_only/training.jsonl')
    assert len(reference_trace) == 512
    assert [r['batch_ids'] for r in reference_trace] == [r['batch_ids'] for r in traces['n1024_low'][:512]]
    assert all(abs(low['lr'] / high['lr'] - .2) < 1e-14 for low, high in zip(traces['n1024_low'], reference_trace))
    info = results['n1024_low']['task_info']
    anchors = set(results['n32_high']['selected_ids'])
    ref = []
    for path in sorted((a.source / 'reference_training_only').glob('train_step*_predictions.jsonl')):
        rr = lines(path)
        step = int(path.name.split('_')[1][4:])
        ref.append({'step': step, 'per_example_exposures': step / 32, 'training': metric(rr, info),
                    'anchor32': metric([r for r in rr if r['id'] in anchors], info)})
    curves['n1024_high_reference'] = ref
    at512 = {name: next(r for r in curve if r['step'] == 512) for name, curve in curves.items()}
    extension = [r for r in curves['n1024_low'] if r['step'] >= 512]
    old32 = read(a.repo / 'artifacts/laya_convergence/native_readout_512/summary.json') if (a.repo / 'artifacts/laya_convergence/native_readout_512/summary.json').exists() else None
    if old32 is None:
        # Repository archive uses a different layout in early versions.
        choices = list((a.repo / 'artifacts/laya_convergence').rglob('summary.json'))
        old32 = next(read(x) for x in choices if read(x).get('actual_updates') == 512)
    assert old32['selected_ids'] == results['n32_high']['selected_ids']
    equal_exposure = []
    for exposure in [16, 32]:
        for lr in ['high', 'low']:
            large_name = 'n1024_high_reference' if lr == 'high' else 'n1024_low'
            small = next(r for r in curves['n32_' + lr] if r['step'] == exposure)
            large = next((r for r in curves[large_name] if r['step'] == exposure * 32), None)
            if large is not None:
                equal_exposure.append({'exposures': exposure, 'lr': lr, 'n32': small, 'n1024': large})
    comparison = {'dev_access': False, 'test_access': False, 'at512': at512, 'low_lr_extension': extension,
                  'equal_exposure': equal_exposure, 'historical_constant_lr_n32': old32['final']['canonical_panel'],
                  'checks': {'new_updates': sum(len(v) for v in traces.values()), 'all_finite': True,
                             'complete_epochs_and_exposure_counts': True, 'all_metrics_recomputed': True,
                             'initial_n32_high_low_exact': True, 'initial_n1024_low_reference_exact': True,
                             'paired_batch_orders_exact': True, 'lr_ratio_exact_within_1e-14': True,
                             'source_reference_checkpoint_hashes': True, 'all_reload_errors_zero': True}}
    dest = a.repo / 'artifacts/laya_gfp_scale_probe'
    dest.mkdir(exist_ok=True)
    for folder in ['frozen_code', 'reference_training_only']:
        shutil.copytree(a.source / folder, dest / folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['status.json', 'initialization_provenance.json']:
        shutil.copy2(a.source / name, dest / name)
    for name in arms:
        sub = dest / name
        sub.mkdir(exist_ok=True)
        for path in (a.source / name).glob('*.json*'):
            shutil.copy2(path, sub / path.name)
    write_json(dest / 'comparison.json', comparison)
    labels = {'n32_high': 'N32, peak LR 1e-4', 'n32_low': 'N32, peak LR 2e-5',
              'n1024_high_reference': 'N1024, peak LR 1e-4 (prior)', 'n1024_low': 'N1024, peak LR 2e-5'}
    colors = {'n32_high': '#177E89', 'n32_low': '#47A8BD', 'n1024_high_reference': '#B76D19', 'n1024_low': '#B43B44'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name, curve in curves.items():
        x = [r['step'] for r in curve]
        style = '--' if name == 'n1024_high_reference' else '-'
        for ax, field in [(axes[0, 0], 'normalized_rmse'), (axes[0, 1], 'spearman')]:
            ax.plot(x, [r['training']['scalar_prediction'][field] for r in curve], style, marker='o', markersize=3, label=labels[name], color=colors[name])
        axes[1, 0].plot(x, [r['anchor32']['scalar_prediction']['normalized_rmse'] for r in curve], style, marker='o', markersize=3, label=labels[name], color=colors[name])
        axes[1, 1].plot([r['per_example_exposures'] for r in curve if r['step']], [r['training']['scalar_prediction']['normalized_rmse'] for r in curve if r['step']], style, marker='o', markersize=3, label=labels[name], color=colors[name])
    axes[0, 0].axhline(.15, color='#777777', linestyle=':', label='Tiny-fit reference 0.15')
    for ax in axes.flat:
        ax.grid(alpha=.15)
    axes[0, 0].set(xlabel='Optimizer updates', ylabel='Full training normalized scalar RMSE')
    axes[0, 1].set(xlabel='Optimizer updates', ylabel='Full training scalar Spearman rho')
    axes[1, 0].set(xlabel='Optimizer updates', ylabel='Shared 32-example normalized scalar RMSE')
    axes[1, 1].set(xlabel='Presentations per example (log scale)', ylabel='Full training normalized scalar RMSE', xscale='log')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Training-only GFP scale / learning-rate probe; one seed')
    fig.savefig(dest / 'learning_curves.png', dpi=180)
    fig.savefig(dest / 'learning_curves.pdf')
    plt.close(fig)
    def table_row(label, entry):
        m = entry['training'];s = m['scalar_prediction']
        rho = f'{s["spearman"]:.5f}' if s['spearman'] is not None else 'undefined'
        return f'| {label} | {s["mae"]:.5f} | {s["rmse"]:.5f} | {s["normalized_rmse"]:.5f} | {rho} | {s["prediction_std"]:.6f} | {m["accuracy"]:.2%} | {m["nll"]:.5f} |'
    rows512 = '\n'.join(table_row(labels[name], at512[name]) for name in ['n32_high', 'n32_low', 'n1024_high_reference', 'n1024_low'])
    rows_ext = '\n'.join(table_row(str(r['step']), r) for r in extension)
    final_low = results['n1024_low']['final']['training']['scalar_prediction']
    old_metric = old32['final']['canonical_panel']['scalar_prediction']
    checkpoints = '\n'.join(f'- `{name}`：`{a.source / name / "checkpoint/model.safetensors"}`；SHA-256 `{results[name]["checkpoint_sha256"]}`。' for name in arms)
    exposure_rows = '\n'.join(f'| {x["exposures"]} | {x["lr"]} | {x["n32"]["training"]["scalar_prediction"]["rmse"]:.5f} | {x["n1024"]["training"]["scalar_prediction"]["rmse"]:.5f} | {x["n1024"]["anchor32"]["scalar_prediction"]["rmse"]:.5f} |' for x in equal_exposure)
    report = f'''# GFP 训练规模、学习率与曝光次数诊断
+
+本轮新增三次训练，只读取训练样本，复用上一轮 1,024 样本高学习率的训练预测作为第四个对照。三个新实验均完成，无提前停止；新训练共 2,048 次更新。实验比较按启动前冻结的预算报告，未按结果追加参数搜索或开发集评估。
+
+## 主要观察
+
+1. **降低学习率改善了固定小样本拟合。**同样 512 次更新，32 样本 RMSE 从 {at512['n32_high']['training']['scalar_prediction']['rmse']:.5f} 降至 {at512['n32_low']['training']['scalar_prediction']['rmse']:.5f}。高学习率组只通过标量拟合参考线，NLL {at512['n32_high']['training']['nll']:.5f} 未达到 0.15；低学习率组最终两项均通过。
+2. **在大样本、相同 512 步预算下，单独降低学习率没有恢复有效幅度的连续值拟合。**RMSE 从 {at512['n1024_high_reference']['training']['scalar_prediction']['rmse']:.5f} 变为 {at512['n1024_low']['training']['scalar_prediction']['rmse']:.5f}，仍接近均值预测；排序信号有所改善。
+3. **延长低学习率大样本训练产生了实质改善，但仍欠拟合。**第 1,024 步 RMSE {final_low['rmse']:.5f}，较第 512 步下降 {(1-final_low['rmse']/at512['n1024_low']['training']['scalar_prediction']['rmse'])*100:.2f}%，预测标准差增至 {final_low['prediction_std']:.5f}、Spearman 为 {final_low['spearman']:.5f}。第 768 步尚未出现同等改善，不能把这条稀疏评估曲线描述为全程单调收敛。
+
+本轮支持继续研究训练预算与优化轨迹，尚不能把大样本问题归结为模型容量不足。没有运行大样本高学习率的 1,024 步对照，因此不清楚其延长后是否也会改善；也没有验证跨 seed 或开发集表现。
+
+## 相同更新预算：2 × 2 对照
+
+固定第 512 次更新，比较 32／1,024 样本与学习率峰值 1e-4／2e-5。高学习率日程为 16 步 warmup + 余弦下降到 1e-5；低学习率将整个日程乘以 0.2。每例曝光数分别为 512 与 16，所以此表是相同更新数／总样本曝光量的比较，不是相同逐样本训练深度。
+
+| 条件 | Train MAE ↓ | Train RMSE ↓ | 标准化 RMSE ↓ | Spearman ↑ | 预测标准差 | 五档准确率 | NLL ↓ |
+|---|---:|---:|---:|---:|---:|---:|---:|
+{rows512}
+
+标准化采用同一份 1,024 条训练数据定义的标准差 {info['fluorescence']['std']:.8f}。0.15 是沿用的小样本标量拟合参考线，不是泛化标准。32 样本分类拟合参考为准确率 ≥31/32 且 NLL ≤0.15；大样本分类结果报告为连续诊断值。
+
+## 同一轨迹上的预算延长
+
+1,024 样本、低学习率实验在第 512 步后保持学习率 2e-6，继续至 1,024 步。第 512／768／1,024 步分别对应每例 16／24／32 次曝光；没有重启优化器或更换训练样本。
+
+| 更新 | Train MAE ↓ | Train RMSE ↓ | 标准化 RMSE ↓ | Spearman ↑ | 预测标准差 | 五档准确率 | NLL ↓ |
+|---|---:|---:|---:|---:|---:|---:|---:|
+{rows_ext}
+
+最终标准化 RMSE 为 {final_low['normalized_rmse']:.5f}，尚未达到 0.15 的拟合参考线。额外预算已经改善误差，但当前预算尚不足以实现充分拟合，不能把剩余误差唯一归因于样本规模。
+
+## 相同逐样本曝光次数的补充读数
+
+| 每例曝光 | 学习率档位 | N32 全部样本 RMSE | N1024 全部样本 RMSE | N1024 中共享 32 条 RMSE |
+|---:|---|---:|---:|---:|
+{exposure_rows}
+
+高学习率大样本历史对照只有 16 次曝光，故没有其 32 次曝光读数。低学习率小样本在 16／32 次曝光时也未达到拟合参考线，因此不能用小样本 512 次曝光的成功，要求大样本在 16／32 次曝光时达到同样效果。这些补充比较仍改变了更新数和学习率历史；完全匹配大样本每例 512 次曝光需要 16,384 次更新，未在本轮固定预算内执行。**本轮没有完全分离样本规模与曝光次数的因果作用。**
+
+## 与已成功的小样本实验的关系
+
+32 条数据与此前成功拟合的子集逐 ID、顺序核对一致。此前 8 步 warmup 后恒定 1e-4、512 更新的标量 RMSE 为 {old_metric['rmse']:.5f}；本轮 32 样本高学习率臂改用 16 步 warmup + 余弦下降，RMSE 为 {at512['n32_high']['training']['scalar_prediction']['rmse']:.5f}。这项比较检验整个学习率日程变化，不能单独区分 warmup 与衰减的作用。
+
+![完整训练曲线](../artifacts/laya_gfp_scale_probe/learning_curves.png)
+
+曲线同时展示全训练样本与所有条件共享的 32 条训练样本，以避免只比较不同大小面板的总体均值。最后一幅图按每例曝光次数绘制，但跨样本规模的相同曝光点具有不同更新数与学习率历史，不能作纯粹的样本规模因果估计。所有原始评估点和该比较的数值均保留在 comparison.json。
+
+## 协议与核验
+
+- 单 seed 20260924；每次从同一原始预训练权重初始化，保留 typed head／type embedding，mean pooling + 新建 LayerNorm；0.5 五档 CE + 0.5 标准化标量 MSE。训练参数总数 420,010,014（含未使用任务头）。
+- 有效 batch 32、micro-batch 4，AdamW weight decay 0.01，clip norm 1，BF16 autocast／FP32 权重。全部条件保持模型、损失和训练集定义的分箱／anchors／标准化一致。
+- 同规模的高／低学习率臂初始预测一致、前 512 步批次 ID 顺序一致；大样本低学习率与历史对照初始预测完全一致。两档逐步学习率比值为 0.2。每次评估前后 CPU／CUDA RNG 状态一致，评估不会扰动后续训练随机轨迹。
+- 全部 2,048 次新训练损失和梯度有限；每轮完整覆盖样本、累计曝光次数准确；全部训练和共享 32 条样本的指标从预测重算一致。三个最终 checkpoint 保存重载概率和标量最大误差均为 0；源代码、历史训练预测和权重 SHA-256 均通过。
+- 没有读取新的 dev 或 test 数据。历史大样本对照仅复制 training.jsonl 与 train_step*_predictions.jsonl，没有复用其开发集结果进行本轮选择。该配置此前受开发诊断启发，不能称为从未接触开发信息的研究。
+- 三次新训练合计 {state['total_wall_seconds']/60:.2f} 分钟，固定完成后只保留各自最终模型状态，无优化器状态。单 seed 与有限超参数点限制结论，不作为模型架构优劣或泛化证据。
+
+## 后续决策
+
+下一步优先在固定低学习率轨迹上进行更长的训练收敛验证，例如预先冻结 2,048 次更新预算，再考虑新的开发集评估。当前仅保留模型权重，没有优化器状态；若要求严格的预算延长对照，需要从原始初始化重放并核验前 1,024 步，再延长，不能将重新初始化 AdamW 的继续训练称为同一轨迹。此后续尚未执行，当前不扩大多任务评测。
+
+## 产物
+
+- [比较与检查结果](../artifacts/laya_gfp_scale_probe/comparison.json)、[冻结协议与运行状态](../artifacts/laya_gfp_scale_probe/status.json)
+- [全部轻量记录与预测](../artifacts/laya_gfp_scale_probe/)、[训练脚本](../scripts/laya_gfp_scale_probe.py)、[队列脚本](../scripts/run_laya_gfp_scale_probe.py)
+- [上一轮 1,024 样本开发验证](laya_gfp_development.md)、[32 样本拟合成功记录](laya_convergence.md)
+
+本地保留的最终权重（未上传 Git）：
+
+{checkpoints}
+'''.replace('\n+', '\n')
    (a.repo / 'research/laya_gfp_scale_probe.md').write_text(report)
    (dest / 'README.md').write_text('Training-only GFP sample-scale / learning-rate probes. See [report](../../research/laya_gfp_scale_probe.md). Final weights remain local. New train-only predictions, traces, frozen code, and the historical training-only reference are included.\n')
    print(json.dumps(comparison['checks'], indent=2))


if __name__ == '__main__':
    main()
