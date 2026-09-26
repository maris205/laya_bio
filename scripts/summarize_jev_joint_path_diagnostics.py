"""Summarize frozen joint-path diagnostics against matched controls."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
RUN_ROOTS = (
    PROJECT / "artifacts/laya_jev_joint_path_diagnostics/run/round",
    PROJECT / "artifacts/laya_jev_joint_path_diagnostics/run_followup/round",
    PROJECT / "artifacts/laya_jev_joint_path_diagnostics/run_weighting_v3/round",
)
SEEDS = (20260927, 20260928)
VARIANTS = ("task_block_gfp_last", "task_block_gfp_first", "task_block_gfp_last_fluo0p5", "task_block_gfp_last_fluo2p0")
TASKS = ("promoter", "structural_class", "fluorescence")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fluorescence(records):
    return [row for row in records if row["task"] == "fluorescence"]


def score_metrics(records, spec):
    values = [row["value"] for row in records]
    predictions = [row["score_prediction"] for row in records]
    errors = [pred - value for pred, value in zip(predictions, values)]
    n = len(records)
    return {
        "rows": n,
        "rmse": math.sqrt(sum(error * error for error in errors) / n),
        "mae": sum(abs(error) for error in errors) / n,
        "spearman": float(spearmanr(values, predictions).statistic),
        "bias": sum(errors) / n,
        "prediction_sd": statistics.pstdev(predictions),
        "constant_mean_rmse": math.sqrt(sum((value - spec["train_mean"]) ** 2 for value in values) / n),
        "constant_median_mae": sum(abs(value - spec["train_median"]) for value in values) / n,
    }


def summarize_run(path, expected_task):
    status = load_json(path / "status.json")
    config = load_json(path / "run_config.json")
    assert status["status"] == "complete"
    assert status["updates"] == 384 if expected_task == "fluorescence" else status["updates"] == 1152
    assert status["checkpoint_reload_exact"] and not status["test_inference"]
    records = fluorescence(load_jsonl(path / "dev_epoch3_predictions.jsonl"))
    train_records = fluorescence(load_jsonl(path / "train_epoch3_predictions.jsonl"))
    trace = load_jsonl(path / "training_trace.jsonl")
    if expected_task == "joint":
        assert Counter(row["task"] for row in trace) == dict.fromkeys(TASKS, 384)
    else:
        assert Counter(row["task"] for row in trace) == {"fluorescence": 384}
    dev = score_metrics(records, config["score_spec"])
    train = score_metrics(train_records, config["score_spec"])
    classification_dev = {}
    if expected_task == "joint":
        final = [row for row in load_jsonl(path / "learning_curve.jsonl") if row["epoch"] == 3 and row["split"] == "dev"]
        assert len(final) == 1
        classification_dev = {
            task: {key: final[0]["metrics"][task][key] for key in ("accuracy", "macro_f1")}
            for task in ("promoter", "structural_class")
        }
    return {
        "path": str(path),
        "dev": dev,
        "train": train,
        "classification_dev": classification_dev,
        "gate": dev["rmse"] < dev["constant_mean_rmse"] and dev["mae"] < dev["constant_median_mae"],
        "near_constant_dev": dev["prediction_sd"] < 0.05,
        "near_constant_train": train["prediction_sd"] < 0.05,
        "checkpoint_sha256": status["checkpoint_sha256"],
    }


def baseline_dir(seed):
    if seed == 20260926:
        return PROJECT / "artifacts/laya_jev_multitask_v1/round/no_cpt_joint"
    return PROJECT / f"artifacts/laya_jev_multitask_seed_confirm/run/round/seed_{seed}_no_cpt_joint"


def single_dir(seed):
    return PROJECT / f"artifacts/laya_jev_gfp_single_seed_controls/run/round/seed_{seed}_no_cpt_gfp_single"


def find_variant(seed, variant):
    name = f"seed_{seed}_no_cpt_joint_{variant}"
    for root in RUN_ROOTS:
        path = root / name
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/laya_jev_joint_path_diagnostics/analysis/summary.json")
    args = parser.parse_args()
    seeds = {}
    variant_count = 0
    pending = []
    for seed in SEEDS:
        variants = {}
        for variant in VARIANTS:
            path = find_variant(seed, variant)
            if path is None:
                continue
            if load_json(path / "status.json")["status"] != "complete":
                pending.append(str(path))
                continue
            variants[variant] = summarize_run(path, "joint")
        variant_count += len(variants)
        seeds[str(seed)] = {
            "baseline_joint": summarize_run(baseline_dir(seed), "joint"),
            "matched_gfp_single": summarize_run(single_dir(seed), "fluorescence"),
            "variants": variants,
        }
    output = {
        "seeds": seeds,
        "arm": "no_cpt",
        "collapse_prediction_sd_threshold": 0.05,
        "audit": {
            "status": "pass",
            "test_inference": False,
            "variant_count": variant_count,
            "pending_runs": pending,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.output), "variant_count": variant_count, "pending": len(pending)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
