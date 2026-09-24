#!/usr/bin/env python3
"""Plot the fixed-protocol CPT and downstream learning curves from saved results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    r=json.loads(a.summary.read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.2),layout='constrained')
    ax=axes[0,0]
    curves=r['cpt_curve']
    for modality,color in [('dna','#2474b5'),('protein','#e07829'),('text','#777777')]:
        ax.plot([v['step'] for v in curves],[v['validation'][modality]['nll'] for v in curves],marker='o',label=('DNA' if modality=='dna' else modality.capitalize()),color=color)
    ax.axvline(128,color='#999999',linestyle=':',linewidth=1)
    ax.set(xlabel='CPT update (128-step embedding/head warmup)',ylabel='Fixed-mask validation NLL',title='A. MLM adaptation includes a fresh prediction head')
    ax.legend(frameon=False)
    styles={'no_cpt_4096':('#777777','--','No CPT · 4,096'),'cpt_4096':('#2474b5','--','CPT · 4,096'),'no_cpt_full':('#777777','-','No CPT · 16,766'),'cpt_full':('#2474b5','-','CPT · 16,766')}
    for axis,split,title in [(axes[0,1],'dev','B. Selection-development learning curves'),(axes[1,0],'train','C. Training accuracy before and after SFT')]:
        for name,run in r['runs'].items():
            c,s,label=styles[name]
            points=[v for v in run['curve'] if v['split']==split]
            axis.plot([v['epoch'] for v in points],[100*v['accuracy'] for v in points],color=c,linestyle=s,marker='o',label=label)
        axis.set(xlabel='SFT epoch',ylabel='Accuracy (%)',title=title,xticks=[0,1,2,3])
        axis.legend(frameon=False,fontsize=9)
    ax=axes[1,1]
    for i,size in enumerate(['4096','full']):
        comp=r['comparisons'][size]
        value=100*comp['accuracy_delta_CPT_minus_no_CPT']
        lo,hi=np.array(comp['paired_group_bootstrap_95_percentile_CI'])*100
        ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color='#2474b5',capsize=4)
        ax.annotate(f'{value:+.2f} pp',xy=(value,i),xytext=(0,12),textcoords='offset points',ha='center')
    ax.axvline(0,color='#999999',linestyle=':',linewidth=1)
    ax.set(yticks=[0,1],yticklabels=['4,096 train','16,766 train'],ylim=(-.55,1.55),xlabel='CPT − no CPT final dev accuracy (percentage points)',title='D. Paired difference with descriptive 95% intervals')
    fig.suptitle('Biological vocabulary → CPT → promoter SFT\nOne seed · fixed final step · reused development set · no test inference',fontsize=14)
    for ext in ['png','pdf']:
        fig.savefig(a.output/('learning_curves.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
