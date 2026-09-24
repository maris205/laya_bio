"""Monitored B1/text-only follow-up, three seeds, automatic dev-only reports."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from laya_metrics import classification_metrics
from laya_overnight import expected_eval

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'artifacts/laya_controls'
PREVIOUS=ROOT/'artifacts/laya_direct_legacy'
SEEDS=(20260922,20260923,20260924)
TASKS=('promoter_detection','fold_class')
KINDS=('b1','text_only')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def status(state,**values):
    data={'state':state,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'no_cpt':True,'test_access':False,'poll_seconds':60,**values}
    write_json(RUNS/'status.json',data);print(json.dumps(data),flush=True)


def validate(path,kind,seed,smoke=False):
    s=json.loads((path/'summary.json').read_text())
    def require(value,reason):
        if not value:raise RuntimeError(f'{path.name}: {reason}')
    require(s['kind']==kind and s['seed']==seed,'wrong model/seed')
    require(s['formal']!=smoke and s['smoke_only']==smoke,'wrong result type')
    require(s['no_cpt'] and not s['test_access'],'CPT/test gate failed')
    require(s['training']['finite'] and s['checkpoint_reload_logits_match'] and s['checkpoint_reload_input_ids_match'],'training/reload failure')
    require(s['evaluated_splits']==['selection_dev','calibration'],'split changed')
    require(all(v['truncated_count']==0 for v in s['item_stats'].values()),'truncation')
    require(s['representation_sha256']==sha(ROOT/'artifacts/laya_direct_bpe_legacy_representation/metadata.json'),'representation mismatch')
    require(s['expansion']['new_vocab_size']==78367 and s['expansion']['maximum_initialization_error']==0,'initialization mismatch')
    if kind=='b1':
        require(set(s['new_embedding_gradient_probe'])==set(TASKS),'missing task gradient')
        require(all(v['gradient_norm']>0 and math.isfinite(v['gradient_norm']) for v in s['new_embedding_gradient_probe'].values()),'invalid gradient')
    else:require(not s['embedding_receives_sequence_supervision'],'text-only received sequence')
    if not smoke:
        require(s['training']['updates']==3005 and s['training']['examples_consumed']==96160,'budget mismatch')
        require(s['n_rows']=={'train':32050,'selection_dev':1978,'calibration':1986},'data count mismatch')
        reference=json.loads((PREVIOUS/f'full_bpe_seed{seed}/summary.json').read_text())
        require(all(s['item_stats'][split]['membership_sha256']==reference['item_stats'][split]['membership_sha256']
                    for split in s['item_stats']),'candidate/control membership differs')
    for split in ('selection_dev','calibration'):
        records=[json.loads(line) for line in (path/f'{split}_predictions.jsonl').read_text().splitlines()]
        truth=expected_eval(split)
        require(len(records)==s['n_rows'][split] and len({r['id'] for r in records})==len(records),'duplicate/missing predictions')
        if not smoke:require({r['id'] for r in records}==set(truth),'prediction membership mismatch')
        for r in records:
            require(all(r[key]==truth[r['id']][key] for key in ('label','task','n_classes')),'label mismatch')
        for task in TASKS:
            selected=[r for r in records if r['task']==task]
            for mode,temp in [('raw',1.),('calibrated',s['calibration_temperature'][task]['temperature'])]:
                metrics=classification_metrics([r['logits'] for r in selected],[r['label'] for r in selected],
                                                n_classes=selected[0]['n_classes'],temperature=temp)
                for key in ('accuracy','balanced_accuracy','macro_f1','mcc','nll','brier','ece15'):
                    reported=s['evaluation'][split][mode][task][key]
                    require(math.isfinite(reported) and abs(reported-metrics[key])<1e-9,'metric recomputation failed')
    return s


def diagnose_complete():
    for seed in SEEDS:
        for condition in ('raw','full_bpe'):
            p=RUNS/f'sequence_diagnostics/{condition}_seed{seed}/summary.json'
            if not p.exists():return False
            s=json.loads(p.read_text())
            if s['seed']!=seed or s['condition']!=condition or not s['baseline_logits_reproduced'] or s['test_access']:
                raise RuntimeError('Invalid sequence diagnostic')
            if set(s['variants'])!={'sequence_removed','residue_shuffle_0','residue_shuffle_1','residue_shuffle_2','candidate_permutation'}:
                raise RuntimeError('Incomplete diagnostic variants')
    return True


def moments(values):
    return {'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else None,'n':len(values)}


def fmt(values,scale=100):
    m=moments(values)
    return f"{scale*m['mean']:.2f}"+(f" ± {scale*m['std']:.2f}" if m['std'] is not None else '')


def report():
    summaries={'candidate':{},'b1':{},'text_only':{}}
    for seed in SEEDS:
        summaries['candidate'][seed]=json.loads((PREVIOUS/f'full_bpe_seed{seed}/summary.json').read_text())
        for kind in KINDS:
            path=RUNS/f'{kind}_seed{seed}'
            if (path/'summary.json').exists():summaries[kind][seed]=validate(path,kind,seed)
    lines=['# Laya BPE 基础功能：分类接口与序列依赖对照','',
           '仅 selection_dev；无 CPT；test 未访问。完整生物 BPE 固定为功能配置，不再搜索词表规模。',
           '均值±标准差为跨训练 seed 的样本标准差，单个 seed 不报告标准差。','',
           '## 模型对照','',
           '| 模型 | 已完成 seed 数 | 任务 | Accuracy (%) | Macro-F1 (%) | MCC | 校准 NLL |',
           '|---|---:|---|---:|---:|---:|---:|']
    for kind,runs in summaries.items():
        if not runs:continue
        for task in TASKS:
            metrics=[s['evaluation']['selection_dev'] for s in runs.values()]
            lines.append(f"| {kind} | {len(runs)} | {task} | {fmt([m['raw'][task]['accuracy'] for m in metrics])} | {fmt([m['raw'][task]['macro_f1'] for m in metrics])} | {fmt([m['raw'][task]['mcc'] for m in metrics],1)} | {fmt([m['calibrated'][task]['nll'] for m in metrics],1)} |")
    lines+=['','candidate 为动态候选评分接口；b1 为相同初始化、近似相同参数量的固定类别输出头；text_only 为相同候选模型在训练与评估时均移除序列。部分 seed 完成时，不把其均值与三 seed 基线当作完整配对比较。','',
            '## 推理时扰动诊断','',
            '序列重排不保证保持生物标签。下表移除/重排后的数值是相对原始标签的匹配率，用于依赖诊断，不是扰动序列的生物分类准确率。每个模型先平均三个固定重排副本，再报告跨三个训练 seed 的均值±标准差。','',
            '| 表示 | 任务 | 原始 Accuracy (%) | 移除序列：原标签匹配率 (%) | 重排序列：原标签匹配率 (%) | 候选重排预测一致率 (%) |',
            '|---|---|---:|---:|---:|---:|']
    diagnostics={}
    for condition in ('full_bpe','raw'):
        values=[]
        for seed in SEEDS:
            path=RUNS/f'sequence_diagnostics/{condition}_seed{seed}/summary.json'
            if path.exists():values.append(json.loads(path.read_text()))
        diagnostics[condition]=values
        if not values:continue
        for task in TASKS:
            original=[v['original']['raw'][task]['accuracy'] for v in values]
            removed=[v['variants']['sequence_removed']['reference_label_metrics']['raw'][task]['accuracy'] for v in values]
            shuffled=[statistics.mean(v['variants'][f'residue_shuffle_{i}']['reference_label_metrics']['raw'][task]['accuracy'] for i in range(3)) for v in values]
            agreement=[v['variants']['candidate_permutation']['comparison_to_original'][task]['prediction_agreement'] for v in values]
            lines.append(f'| {condition} | {task} | {fmt(original)} | {fmt(removed)} | {fmt(shuffled)} | {fmt(agreement)} |')
    paired={}
    lines+=['','## 配对结果与解释边界','']
    for kind in KINDS:
        paired[kind]={}
        for task in TASKS:
            seeds=sorted(summaries[kind])
            if not seeds:continue
            delta=[summaries[kind][seed]['evaluation']['selection_dev']['raw'][task]['accuracy']-
                   summaries['candidate'][seed]['evaluation']['selection_dev']['raw'][task]['accuracy'] for seed in seeds]
            paired[kind][task]=moments(delta)
            lines.append(f'- {kind} 相对相同 seed 的 candidate，{task} Accuracy 平均差值 {100*statistics.mean(delta):+.2f} 个百分点（{len(seeds)} 个配对 seed）。')
    lines+=['','输入扰动只检查依赖性；不能证明模型学到因果机制。候选重排按类别语义还原 logits 后比较，若一致率低于 100% 则说明位置敏感性。B1 不读取候选字符串，以已知任务选择固定 2/7 类输出；它保留原 Laya 上下文表示层和读出 MLP，但最后类别层重新初始化，且输入长度不同。','',
            '本轮完成后队列停止。固定 test 评估协议及尚未验证的同源独立性仍需单独处理，不因开发集结果自动解锁 test。','']
    (ROOT/'research/laya_controls_results.md').write_text('\n'.join(lines))
    write_json(RUNS/'results.json',{'no_cpt':True,'test_access':False,
         'completed_seeds':{k:list(v) for k,v in summaries.items()},'paired_accuracy_delta_control_minus_candidate':paired,
         'report':str(ROOT/'research/laya_controls_results.md')})


def freeze():
    previous=json.loads((PREVIOUS/'frozen_pair/run_manifest.json').read_text())['hashes']
    own=[ROOT/'scripts'/name for name in ('laya_control_experiment.py','laya_sequence_diagnostics.py',
          'run_laya_controls.py','laya_overnight.py','test_laya_controls.py')]
    own += [ROOT/'artifacts/laya_model/model.safetensors',ROOT/'artifacts/laya_model/rl_agent_config.json']
    for seed in SEEDS:
        for condition in ('raw','full_bpe'):
            own += [PREVIOUS/f'{condition}_seed{seed}/summary.json']
    hashes={**previous,**{str(p.relative_to(ROOT)):sha(p) for p in own}}
    data={'seeds':list(SEEDS),'models':list(KINDS),'updates':3005,'micro_batch':8,'grad_accum':4,
          'no_cpt':True,'test_access':False,'hashes':hashes}
    path=RUNS/'run_manifest.json'
    if path.exists() and json.loads(path.read_text())!=data:raise RuntimeError('Control manifest changed')
    write_json(path,data)


def verify():
    for name,expected in json.loads((RUNS/'run_manifest.json').read_text())['hashes'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError(f'Frozen file changed: {name}')


def launch(kind,seed,smoke=False):
    verify()
    output=RUNS/(f'smoke_{kind}' if smoke else f'{kind}_seed{seed}')
    if (output/'summary.json').exists():validate(output,kind,seed,smoke);return
    if output.exists():raise RuntimeError(f'Partial output exists: {output}')
    while True:
        mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(mem.strip().splitlines()[0])<500:break
        status('waiting_for_gpu',kind=kind,seed=seed,smoke=smoke);time.sleep(60)
    command=[sys.executable,'-u',str(ROOT/'scripts/laya_control_experiment.py'),'--kind',kind,
             '--seed',str(seed),'--output-dir',str(output)]
    if smoke:command.append('--smoke')
    log=RUNS/'logs'/f'{output.name}.log'
    with log.open('w') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        while process.poll() is None:
            status('running',kind=kind,seed=seed,smoke=smoke,pid=process.pid,log=str(log))
            time.sleep(60)
        if process.returncode:raise RuntimeError(f'{output.name} failed: {process.returncode}; see {log}')
    validate(output,kind,seed,smoke)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-pid',type=int)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if a.report_only:report();return
    with (RUNS/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            freeze();verify()
            while not diagnose_complete():
                if not a.diagnostic_pid or not Path(f'/proc/{a.diagnostic_pid}').exists():
                    raise RuntimeError('Diagnostics ended without all six completed results')
                status('waiting_for_diagnostics',diagnostic_pid=a.diagnostic_pid);time.sleep(60)
            report()
            for kind in KINDS:launch(kind,SEEDS[0],smoke=True)
            for seed in SEEDS:
                for kind in KINDS:
                    launch(kind,seed)
                    report()
            status('complete',completed_runs=6,completed_seeds=list(SEEDS),report=str(ROOT/'research/laya_controls_results.md'))
        except Exception as exc:
            status('failed',error=str(exc));raise


if __name__=='__main__':main()
