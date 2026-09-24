"""Show prediction flips whose effects partially cancel in aggregate accuracy."""
from paper_plot_style import *
import numpy as np

records=read('artifacts/laya_controls/reliability_audit.json')['candidate_order_details']
fig,axes=plt.subplots(1,2,figsize=(6.8,2.7),sharey=True)
for ax,task,label in zip(axes,TASKS,('DNA promoter','Protein structural class')):
    bottom=np.zeros(2)
    for j,(key,legend,hatch) in enumerate((('correct_to_wrong','Correct to wrong',''),
                                         ('wrong_to_correct','Wrong to correct','///'),
                                         ('wrong_to_other_wrong','Wrong to other wrong','...'))):
        vals=[np.mean([100*r[key]/r['n'] for r in records if r['task']==task and r['condition']==c])
              for c in ('raw','full_bpe')]
        ax.bar(range(2),vals,bottom=bottom,color=COLORS[j],hatch=hatch,edgecolor='white',width=.52,label=legend)
        bottom+=vals
    for x,v in enumerate(bottom):ax.text(x,v+.15,f'{v:.2f}%',ha='center',fontsize=9)
    ax.set_xticks(range(2),['Raw','BioBPE'])
    ax.set_xlabel(label)
    ax.set_ylim(0,10.5)
axes[0].set_ylabel('Predictions changed (%)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,
           bbox_to_anchor=(.5,1.04),columnspacing=1.1)
fig.tight_layout(rect=(0,0,1,.9),w_pad=1.5)
save(fig,'candidate_order')
