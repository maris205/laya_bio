#!/usr/bin/env python3
"""Plot per-task quality, continuous Score calibration and measured cost."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from laya_biocpt_data import read_lines


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();s=json.loads(a.summary.read_text());a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(2,3,figsize=(15.4,8.3),layout='constrained')
    tasks=['promoter','structural_class','fluorescence'];titles=['Promoter · Noul','Protein structure · Choice','GFP · Score']
    for i,(task,title) in enumerate(zip(tasks,titles)):
        metric='rmse' if task=='fluorescence' else 'accuracy';scale=1 if task=='fluorescence' else 100
        for name,color,style,label in [('cpt_joint','#2474b5','-','CPT joint'),('no_cpt_joint','#777777','-','No-CPT joint'),('cpt_'+task,'#e07829','--','CPT single')]:
            pts=[r for r in s['runs'][name]['curve'] if r['split']=='dev']
            axes[0,i].plot([r['epoch'] for r in pts],[scale*r['metrics'][task][metric] for r in pts],marker='o',color=color,linestyle=style,label=label)
        baseline=s['baselines'][task][metric]*scale;axes[0,i].axhline(baseline,color='#aaaaaa',linestyle=':',label='Train-fitted constant')
        axes[0,i].set(title=title,xlabel='SFT epoch',ylabel='RMSE (native units; lower is better)' if task=='fluorescence' else 'Development accuracy (%)',xticks=[0,1,2,3])
        axes[0,i].legend(frameon=False,fontsize=8)
    ax=axes[1,0]
    for i,task in enumerate(tasks[:2]):
        c=s['comparisons'][task]['cpt_joint_minus_cpt_single'];value=100*c['right_minus_left'];lo,hi=np.array(c['paired_entity_bootstrap_95_CI'])*100
        ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color='#2474b5',capsize=4)
        ax.annotate(f'{value:+.2f} pp',(value,i),xytext=(0,10),textcoords='offset points',ha='center')
    ax.axvline(0,color='#888888',linestyle=':');ax.axvline(-3,color='#bb7777',linestyle='--',label='Predeclared −3 pp flag')
    ax.set(yticks=[0,1],yticklabels=['Promoter','Structure'],ylim=(-.6,1.6),xlabel='CPT joint − CPT single accuracy (pp)',title='Joint retention · descriptive 95% intervals');ax.legend(frameon=False,fontsize=8)
    ax=axes[1,1];rr=[r for r in read_lines(a.root/'round/cpt_joint/dev_epoch3_predictions.jsonl') if r['task']=='fluorescence']
    true=np.array([r['value'] for r in rr]);pred=np.array([r['score_prediction'] for r in rr])
    ax.hexbin(true,pred,gridsize=35,mincnt=1,cmap='Blues');lo=min(true.min(),pred.min());hi=max(true.max(),pred.max());ax.plot([lo,hi],[lo,hi],color='#888888',linestyle=':')
    ax.set(xlabel='Observed log fluorescence',ylabel='Predicted expected log fluorescence',title='CPT joint · full GFP development set')
    ax=axes[1,2];x=np.arange(3);width=.34
    for shift,name,color,label in [(-width/2,'no_cpt_joint','#777777','No-CPT joint'),(width/2,'cpt_joint','#2474b5','CPT joint')]:
        bench=s['runs'][name].get('end_to_end_benchmark',s['runs'][name]['inference_benchmark'])
        values=[bench[t]['single_request_p50_ms'] for t in tasks]
        ax.bar(x+shift,values,width,color=color,label=label)
    ax.set(xticks=x,xticklabels=['Promoter','Structure','GFP'],ylabel='Single-request p50 latency (ms)',title=('Raw sequence → typed answer' if all('end_to_end_benchmark' in s['runs'][n] for n in ['cpt_joint','no_cpt_joint']) else 'Full panel · cached tokenization'));ax.legend(frameon=False,fontsize=8)
    fig.suptitle('One Laya model, three tasks, shared JEV-style output\n8,192 labels/task · one seed · fixed final epoch · reused development sets',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(a.output/('decision_round.'+ext),dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
