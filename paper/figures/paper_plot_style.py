from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent
SEEDS = (20260922, 20260923, 20260924)
TASKS = ('promoter_detection', 'fold_class')
COLORS = ('#0072B2', '#D55E00', '#009E73')
plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'], 'font.size':10,
                     'axes.labelsize':10, 'xtick.labelsize':9, 'ytick.labelsize':9,
                     'legend.fontsize':9, 'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.bbox':'tight', 'savefig.pad_inches':.06})


def read(path): return json.loads((ROOT/path).read_text())
def save(fig, name):
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.png', dpi=180)
    plt.close(fig)
