"""Shared publication style for the BioDecisionBench paper figures.

All figures are generated from the on-disk result JSONs (artifacts/benchmark_v2_eval,
artifacts/benchmark_v2_train, data/06_benchmark_v2_unified) - no hardcoded metrics.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JEV = Path("/root/autodl-tmp/jev_gene")
AE = JEV / "artifacts/benchmark_v2_eval"
TR = JEV / "artifacts/benchmark_v2_train"
UNI = JEV / "data/06_benchmark_v2_unified"
FIG_DIR = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "stix",
})

# architecture colors (grayscale-distinguishable: blue / red / gray)
C_SHARED = "#1f77b4"      # #1 shared typed-decision scorer
C_HEAD = "#d62728"        # #2 per-task heads
C_GEN = "#7f7f7f"         # #3 frozen LM candidate likelihood
C_INIT = "#bcbd22"        # untrained init / frozen baseline in transfer plots
C_ACC = "#2ca02c"
C_NEG = "#9467bd"

def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=160)
    print(f"Saved: {FIG_DIR / (name + '.pdf')}")

def load(path):
    return json.loads(Path(path).read_text())

def qgroups_conflict():
    """Conflict-group quarantine count from the first isolate run.

    The on-disk isolation_report.json was overwritten by an idempotent re-run
    (post-isolation there are zero remaining conflicts), so the authoritative
    count lives in research/benchmark_v2_build_protocol.md's admission table.
    """
    import re
    txt = (JEV / "research/benchmark_v2_build_protocol.md").read_text()
    m = re.search(r"隔离冲突组 \| ([\d,]+)", txt)
    return int(m.group(1).replace(",", ""))

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

# Canonical family map used by the training/balancing code
# (duplicate of scripts/benchmark_v2/train_benchmark_v2.py::PREFIX_FAMILY; keep in sync)
PREFIX_FAMILY = [("pg_", "ProteinGym"), ("rnac_", "RNAcompete"), ("gue_", "GUE"),
                 ("gb_", "GenomicBenchmarks"), ("gl_", "gene_lan"), ("dna_", "dnagpt_pools"),
                 ("deepstarr", "DeepSTARR"), ("tape_", "TAPE"), ("lg_", "local_snapshots"),
                 ("protein_homology", "local_snapshots"), ("deeploc", "DeepLoc")]

def family_of(task_id):
    for pre, fam in PREFIX_FAMILY:
        if task_id.startswith(pre):
            return fam
    return "other"
