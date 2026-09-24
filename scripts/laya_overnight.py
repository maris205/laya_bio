"""Bounded overnight monitor: finish first pair, run two more seeds, report dev.

Run under screen. No language-model/API call, CPT, test evaluation, or messaging.
Polling is every 60 seconds; one GPU worker at a time. Failures stop the queue.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'artifacts/laya_direct_legacy'
STATUS=RUNS/'overnight_status.json'
SEEDS=(20260922,20260923,20260924)
CONDITIONS=('raw','full_bpe')
TASKS=('promoter_detection','fold_class')
METRICS=('accuracy','balanced_accuracy','macro_f1','mcc','nll','brier','ece15')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def status(state,**values):
    value={'state':state,'monitor_pid':__import__('os').getpid(),
           'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'poll_seconds':60,'no_cpt':True,'test_access':False,**values}
    atomic_json(STATUS,value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def process_alive(pid):
    if not pid:return False
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return False
    try:return path.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def verify_frozen():
    paths=[RUNS/'frozen_pair/run_manifest.json',RUNS/'overnight_manifest.json']
    for path in paths:
        manifest=json.loads(path.read_text())
        for name,expected in manifest['hashes'].items():
            if sha(ROOT/name)!=expected:raise RuntimeError(f'Frozen input/code changed: {name}')


def expected_eval(split):
    eligible=json.loads((ROOT/'artifacts/laya_formal_data/eligible_ids.json').read_text())
    allowed={rid for task in eligible['tasks'].values() for part in task['splits'].values() for rid in part['ids']}
    result={}
    for task in TASKS:
        for line in (ROOT/f'artifacts/laya_formal_data/{task}_{split}.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['id'] not in allowed:continue
            label=r['label'] if isinstance(r['label'],int) else r['choices'].index(r['label'])
            result[r['id']]={'task':task,'label':label,'n_classes':len(r['choices'])}
    return result


def validate(run_dir,seed,condition,smoke=False):
    s=json.loads((run_dir/'summary.json').read_text())
    def require(test,message):
        if not test:raise RuntimeError(f'{run_dir.name}: {message}')
    require(s['seed']==seed and s['condition']==condition,'wrong seed/condition')
    require(s['smoke_only']==smoke and s['formal']!=smoke,'wrong formal/smoke status')
    require(s['no_cpt'] and not s['test_access'],'CPT/test boundary changed')
    require(s['evaluated_splits']==['selection_dev','calibration'],'evaluation splits changed')
    require(s['training']['finite'] and s['checkpoint_reload_input_ids_match'] and s['checkpoint_reload_logits_match'],
            'training/reload verification failed')
    if not smoke:
        require(s['n_rows']=={'train':32050,'selection_dev':1978,'calibration':1986},'sample count changed')
        require(s['training']['updates']==3005 and s['training']['examples_consumed']==96160,'update budget changed')
    require(s['embedding_trainable'],'embedding frozen')
    require(all(v['truncated_count']==0 for v in s['item_stats'].values()),'input was truncated')
    require(s['representation_sha256']==sha(ROOT/'artifacts/laya_direct_bpe_legacy_representation/metadata.json'),
            'representation mismatch')
    if condition=='full_bpe':
        require(s['expansion']['new_vocab_size']==78367 and s['expansion']['maximum_initialization_error']==0,
                'full-vocabulary initialization failed')
        require(set(s['new_embedding_gradient_probe'])==set(TASKS),'missing modality gradient check')
        require(all(math.isfinite(v['gradient_norm']) and v['gradient_norm']>0 for v in s['new_embedding_gradient_probe'].values()),
                'invalid new embedding gradient')
    diagnostics={}
    for split in ('selection_dev','calibration'):
        predictions=[json.loads(line) for line in (run_dir/f'{split}_predictions.jsonl').read_text().splitlines()]
        require(len(predictions)==s['n_rows'][split],'prediction count differs from summary')
        require(len({r['id'] for r in predictions})==len(predictions),'duplicate prediction IDs')
        expected=expected_eval(split)
        if not smoke:require({r['id'] for r in predictions}==set(expected),'prediction ID membership changed')
        by_task=defaultdict(list)
        for r in predictions:
            truth=expected[r['id']]
            require(all(r[k]==truth[k] for k in ('task','label','n_classes')),'prediction label/task mismatch')
            require(len(r['logits'])==r['n_classes'] and all(math.isfinite(x) for x in r['logits']),'invalid prediction logits')
            by_task[r['task']].append(r)
        diagnostics[split]={}
        for task,values in by_task.items():
            for mode,temp in [('raw',1.0),('calibrated',s['calibration_temperature'][task]['temperature'])]:
                recomputed=classification_metrics([r['logits'] for r in values],[r['label'] for r in values],
                                                  n_classes=values[0]['n_classes'],temperature=temp)
                for metric in METRICS:
                    reported=s['evaluation'][split][mode][task][metric]
                    require(math.isfinite(reported) and abs(recomputed[metric]-reported)<1e-9,
                            f'{split}/{task}/{mode}/{metric} cannot be reproduced')
            k=values[0]['n_classes'];confusion=[[0]*k for _ in range(k)]
            for r in values:
                predicted=max(range(k),key=lambda i:r['logits'][i])
                confusion[r['label']][predicted]+=1
            diagnostics[split][task]={'confusion':confusion,'class_counts':[sum(row) for row in confusion],
                'class_recall':[row[i]/sum(row) if sum(row) else None for i,row in enumerate(confusion)]}
    atomic_json(run_dir/'verified_diagnostics.json',diagnostics)
    return s


def mean_std(values):
    return {'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else None,'n':len(values)}


def summarize():
    data={}
    for seed in SEEDS:
        if not all((RUNS/f'{c}_seed{seed}/summary.json').exists() for c in CONDITIONS):continue
        data[seed]={c:validate(RUNS/f'{c}_seed{seed}',seed,c) for c in CONDITIONS}
        for split in data[seed]['raw']['item_stats']:
            if data[seed]['raw']['item_stats'][split]['membership_sha256']!=data[seed]['full_bpe']['item_stats'][split]['membership_sha256']:
                raise RuntimeError('Pair membership hashes differ')
    aggregate={}
    for task in TASKS:
        aggregate[task]={}
        for mode in ('raw','calibrated'):
            aggregate[task][mode]={}
            for metric in METRICS:
                values={c:[pair[c]['evaluation']['selection_dev'][mode][task][metric] for pair in data.values()] for c in CONDITIONS}
                if not data:continue
                delta=[b-a for a,b in zip(values['raw'],values['full_bpe'])]
                relative=[100*(b-a)/abs(a) for a,b in zip(values['raw'],values['full_bpe']) if a!=0]
                aggregate[task][mode][metric]={**{c:mean_std(v) for c,v in values.items()},
                     'paired_delta_full_minus_raw':mean_std(delta),
                     'relative_change_percent':mean_std(relative) if relative else None}
    result={'complete_seeds':list(data),'no_cpt':True,'test_access':False,'evaluation_split':'selection_dev',
            'aggregate':aggregate,'runs':{str(seed):{c:str(RUNS/f'{c}_seed{seed}/summary.json') for c in CONDITIONS} for seed in data},
            'uncertainty':'sample standard deviation across seeds, not confidence intervals; no homology independence claim'}
    atomic_json(RUNS/'multiseed_results.json',result)
    lines=['# Laya 原始输入与完整生物 BPE：自动汇总','',
           f'完整配对 seed：{list(data)}。仅 selection_dev 评价；test 未访问；无 CPT。',
           '以下均从保存的逐样本 logits 重新计算并核对。历史 BPE 使用外部序列语料，不能称为 train-only 词表。','',
           '## 逐 seed 原始结果','',
           '| seed | 条件 | 任务 | Accuracy (%) | Macro-F1 (%) | MCC | NLL | ECE15 | 训练分钟 |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for seed,pair in data.items():
        for c,s in pair.items():
            for task in TASKS:
                m=s['evaluation']['selection_dev']['raw'][task]
                lines.append(f"| {seed} | {c} | {task} | {100*m['accuracy']:.2f} | {100*m['macro_f1']:.2f} | {m['mcc']:.4f} | {m['nll']:.4f} | {m['ece15']:.4f} | {s['training']['seconds']/60:.2f} |")
    lines += ['','## 均值与跨 seed 标准差','',
              '| 任务 | 指标 | raw | full_bpe | 配对差值 full−raw |','|---|---|---:|---:|---:|']
    for task in TASKS:
        for metric,scale in [('accuracy',100),('macro_f1',100),('nll',1),('ece15',1)]:
            if not data:continue
            value=aggregate[task]['raw'][metric]
            def fmt(v):
                return f"{scale*v['mean']:.4f}"+(f" ± {scale*v['std']:.4f}" if v['std'] is not None else '（单 seed）')
            lines.append(f"| {task} | {metric}{' (%) / 差值为百分点' if scale==100 else ''} | {fmt(value['raw'])} | {fmt(value['full_bpe'])} | {fmt(value['paired_delta_full_minus_raw'])} |")
    lines += ['','## 当前发现','']
    for index,task in enumerate(TASKS,1):
        if not data:continue
        acc=aggregate[task]['raw']['accuracy']['paired_delta_full_minus_raw']['mean']*100
        f1=aggregate[task]['raw']['macro_f1']['paired_delta_full_minus_raw']['mean']*100
        nll=aggregate[task]['calibrated']['nll']['paired_delta_full_minus_raw']['mean']
        lines.append(f'{index}. {task}：完整扩表相对原始输入的平均 Accuracy 差值 {acc:+.2f} 个百分点，Macro-F1 差值 {f1:+.2f} 个百分点；校准后 NLL 差值 {nll:+.5f}（负值更好）。这些是当前开发集观察，不直接证明生物机制或统计显著性。')
    lines += ['','## 后续','',
              '三 seed 完成后本队列停止。下一阶段保留固定分类头 B1 和序列移除/重排诊断；模型/协议锁定后再做 test。',
              '不要根据本表自动挑选最有利 seed 或修改本轮超参数。每类混淆矩阵和召回率见各 run 的 verified_diagnostics.json。','']
    (ROOT/'research/laya_direct_bpe_multiseed_results.md').write_text('\n'.join(lines))
    return result


def setup_manifest():
    paths=[Path(__file__),RUNS/'frozen_pair/run_manifest.json']
    manifest={'seeds':list(SEEDS),'conditions':list(CONDITIONS),'additional_jobs':4,'poll_seconds':60,
              'no_cpt':True,'test_access':False,'stop_after':'three complete matched seed pairs',
              'hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    path=RUNS/'overnight_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Overnight manifest changed')
    atomic_json(path,manifest)


def run_monitor():
    setup_manifest()
    while True:
        verify_frozen()
        first=json.loads((RUNS/'pair_status.json').read_text())
        if first['state']=='failed':raise RuntimeError('First pair failed: '+first.get('error','unknown'))
        if first['state']=='complete':
            summarize()
            break
        if not process_alive(first.get('pid')) and time.time()-(RUNS/'pair_status.json').stat().st_mtime>180:
            raise RuntimeError('First-pair worker disappeared without successful completion')
        status('waiting_for_first_pair',first_pair=first,remaining_additional_jobs=4)
        time.sleep(60)
    for seed in SEEDS[1:]:
        for condition in CONDITIONS:
            verify_frozen()
            output=RUNS/f'{condition}_seed{seed}'
            if (output/'summary.json').exists():
                validate(output,seed,condition)
                continue
            # Never overwrite a partial run and silently call it a fresh result.
            if output.exists():raise RuntimeError(f'Partial run exists: {output}; review before retry')
            while True:
                usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                if int(usage.strip().splitlines()[0])<500:break
                status('waiting_for_free_gpu',seed=seed,condition=condition)
                time.sleep(60)
            command=[sys.executable,'-u',str(ROOT/'scripts/laya_direct_experiment.py'),
                     '--condition',condition,'--seed',str(seed),'--output-dir',str(output)]
            logpath=RUNS/'logs'/f'{condition}_seed{seed}.log'
            with logpath.open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    status('running_additional_seed',seed=seed,condition=condition,worker_pid=process.pid,
                           log=str(logpath),command=command)
                    time.sleep(60)
                if process.returncode:raise RuntimeError(f'{condition} seed {seed} failed ({process.returncode}); {logpath}')
            validate(output,seed,condition)
        summarize()
    report=summarize()
    if report['complete_seeds']!=list(SEEDS):raise RuntimeError('Final result is missing a seed')
    status('complete',completed_seeds=list(SEEDS),completed_runs=6,
           report=str(ROOT/'research/laya_direct_bpe_multiseed_results.md'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='validate both smoke runs; no training')
    p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args()
    if a.check:
        for c in CONDITIONS:validate(RUNS/f'smoke_{c}',SEEDS[0],c,smoke=True)
        print('Both smoke runs passed independent prediction/metric/label/initialization validation.')
        return
    if a.summarize_only:
        print(json.dumps(summarize(),indent=2));return
    # Kernel lock prevents two screen sessions from scheduling duplicate jobs.
    with (RUNS/'overnight.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run_monitor()
        except Exception as exc:
            status('failed',error=str(exc))
            raise


if __name__=='__main__':main()
