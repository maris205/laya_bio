"""Dev reference-label matching; these are not new biological-label evaluations."""
from paper_plot_style import *
import numpy as np

fig, axes = plt.subplots(1,2,figsize=(6.8,2.65),sharey=True)
for ax,task,label in zip(axes,TASKS,('DNA promoter','Protein structural class')):
    for j,condition in enumerate(('raw','full_bpe')):
        seed_values=[]
        for seed in SEEDS:
            s=read(f'artifacts/laya_controls/sequence_diagnostics/{condition}_seed{seed}/summary.json')
            val=[s['original']['raw'][task]['accuracy'],
                 s['variants']['sequence_removed']['reference_label_metrics']['raw'][task]['accuracy'],
                 np.mean([s['variants'][f'residue_shuffle_{r}']['reference_label_metrics']['raw'][task]['accuracy'] for r in range(3)])]
            seed_values.append(val)
        values=np.asarray(seed_values)*100
        ax.errorbar(np.arange(3)+(j-.5)*.12,values.mean(0),yerr=values.std(0,ddof=1),
                    color=COLORS[j],marker=('o','s')[j],linestyle=('-','--')[j],capsize=3,
                    label=('Raw candidate','BioBPE candidate')[j],linewidth=1.3,markersize=5)
    ax.set_xticks(range(3),['Original','Removed','Shuffled'])
    ax.set_xlabel(label)
    ax.set_ylim(-6,100)
    ax.set_yticks(range(0,101,20))
    ax.grid(axis='y',alpha=.18)
axes[0].set_ylabel('Original-label match (%)')
axes[0].legend(frameon=False,loc='lower left')
fig.tight_layout(w_pad=1.5)
save(fig,'sequence_diagnostics')
