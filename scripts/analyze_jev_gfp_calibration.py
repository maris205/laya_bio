"""Post-hoc GFP output calibration (supplementary, outside the frozen no-calibration protocol).

For each NO-CPT joint run, fit a calibration of the GFP score on that run's own final train-diagnostic
predictions (1,024 fixed train rows), apply it to the final dev predictions, and re-apply the frozen success
rule. Classification results are unchanged. Dev labels are never used for fitting; no test access.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_jev_recipes import JP, PROJECT, RULE, final_dev, runs  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402


def gfp_rows(path):
    return [row for row in map(json.loads, path.read_text().splitlines()) if row["task"] == "fluorescence"]


def fit(train):
    x = [row["score_prediction"] for row in train]
    y = [row["value"] for row in train]
    mx, my = statistics.mean(x), statistics.mean(y)
    var = sum((v - mx) ** 2 for v in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / var if var > 0 else 0.0
    return {"affine": (slope, my - slope * mx), "shift": (1.0, my - mx)}


def metrics(dev, a, b):
    pred = [a * row["score_prediction"] + b for row in dev]
    true = [row["value"] for row in dev]
    err = [p - t for p, t in zip(pred, true)]
    return {"rmse": math.sqrt(sum(e * e for e in err) / len(err)), "mae": sum(map(abs, err)) / len(err),
            "sd": statistics.pstdev(pred), "spearman": float(spearmanr(true, pred).statistic), "bias": sum(err) / len(err)}


def gate(m, cls):
    gfp = m["rmse"] < RULE["gfp_rmse_lt"] and m["mae"] < RULE["gfp_mae_lt"] and m["sd"] >= RULE["gfp_sd_ge"]
    return gfp, gfp and cls["promoter_acc"] >= RULE["promoter_acc_ge"] and cls["structure_acc"] >= RULE["structure_acc_ge"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=JP / "analysis/gfp_calibration.json")
    args = parser.parse_args()
    table = {}
    for recipe, seed, path in runs():
        if not path.is_dir() or json.loads((path / "status.json").read_text())["status"] != "complete":
            continue
        train, dev = gfp_rows(path / "train_epoch3_predictions.jsonl"), gfp_rows(path / "dev_epoch3_predictions.jsonl")
        assert len(train) == 1024 and len(dev) == 5362, path
        cls = final_dev(path)
        record = {"path": str(path.relative_to(PROJECT)), "seed": seed}
        for name, (a, b) in {"raw": (1.0, 0.0), **fit(train)}.items():
            m = metrics(dev, a, b)
            gfp_ok, all_ok = gate(m, cls)
            record[name] = {"a": a, "b": b, **m, "gfp_pass": gfp_ok, "pass": all_ok}
        table.setdefault(recipe, []).append(record)
    summary = {}
    for recipe, records in table.items():
        summary[recipe] = {name: {"gfp_pass": sum(r[name]["gfp_pass"] for r in records), "pass": sum(r[name]["pass"] for r in records),
                                  "n": len(records), "mae_mean": statistics.mean(r[name]["mae"] for r in records)}
                           for name in ("raw", "shift", "affine")}
    out = {"note": "Supplementary post-hoc analysis outside the frozen no-calibration protocol; calibration fit on train diagnostics only.",
           "rule": RULE, "test_inference": False, "summary": summary, "runs": table}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for recipe, s in summary.items():
        print(f"{recipe:28s} n={s['raw']['n']}  " + "  ".join(
            f"{k}: GFP {v['gfp_pass']}/{v['n']} all {v['pass']}/{v['n']} MAE {v['mae_mean']:.3f}" for k, v in s.items()))
    print("\nper-run (seed: raw MAE -> affine MAE, affine a/b):")
    for recipe, records in table.items():
        for r in records:
            print(f"  {recipe:28s} {r['seed']}  {r['raw']['mae']:.3f} -> {r['affine']['mae']:.3f} (shift {r['shift']['mae']:.3f})  a={r['affine']['a']:.2f} b={r['affine']['b']:+.2f}  bias {r['raw']['bias']:+.3f} -> {r['affine']['bias']:+.3f}  pass {r['raw']['pass']}->{r['affine']['pass']}")


if __name__ == "__main__":
    main()
