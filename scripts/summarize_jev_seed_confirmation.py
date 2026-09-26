"""Independently recompute and audit the paired seed-confirmation results."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score


PROJECT = Path(__file__).resolve().parents[1]
PRIOR = PROJECT / "artifacts/laya_jev_multitask_v1/round"
CURRENT = PROJECT / "artifacts/laya_jev_multitask_seed_confirm/run/round"
TASKS = ("promoter", "structural_class", "fluorescence")
SEEDS = (20260926, 20260927, 20260928)
ARMS = ("no_cpt", "cpt")
DATA_MANIFEST = json.loads(
    (PROJECT / "artifacts/laya_jev_multitask_v1/data/manifest.json").read_text(encoding="utf-8")
)


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(seed, arm):
    if seed == SEEDS[0]:
        return PRIOR / f"{arm}_joint"
    return CURRENT / f"seed_{seed}_{arm}_joint"


def by_task(records, task):
    result = {record["id"]: record for record in records if record["task"] == task}
    assert len(result) == sum(record["task"] == task for record in records)
    return result


def score_metrics(records, spec):
    values = [row["value"] for row in records]
    predictions = [row["score_prediction"] for row in records]
    errors = [pred - value for pred, value in zip(predictions, values)]
    n = len(records)
    return {
        "rmse": math.sqrt(sum(error * error for error in errors) / n),
        "mae": sum(abs(error) for error in errors) / n,
        "spearman": float(spearmanr(values, predictions).statistic),
        "bias": sum(errors) / n,
        "prediction_sd": statistics.pstdev(predictions),
        "constant_mean_rmse": math.sqrt(sum((value - spec["train_mean"]) ** 2 for value in values) / n),
        "constant_median_mae": sum(abs(value - spec["train_median"]) for value in values) / n,
    }


def class_metrics(records, task):
    labels = [row["label"] for row in records]
    predictions = [row["prediction"] for row in records]
    count = len(records[0]["probs"])
    train_counts = DATA_MANIFEST["outputs"]["train"]["classes"][task]
    majority = max(range(count), key=lambda label: train_counts[str(label)])
    constant = [majority] * len(labels)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(count)), average="macro", zero_division=0)),
        "constant_accuracy": float(accuracy_score(labels, constant)),
        "constant_macro_f1": float(f1_score(labels, constant, labels=list(range(count)), average="macro", zero_division=0)),
        "correct_by_class": {str(label): sum(row["label"] == label and row["prediction"] == label for row in records) for label in range(count)},
        "support_by_class": {str(label): sum(row["label"] == label for row in records) for label in range(count)},
    }


def summarize_run(path):
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and status["updates"] == 1152
    assert status["checkpoint_reload_exact"] and status["model_retained"]
    assert not status["test_inference"] and not config["test_inference"]
    assert status["per_entity_exposure_histogram"] == {"3": 24576}
    assert sha(path / "model.safetensors") == status["checkpoint_sha256"]
    trace = load_lines(path / "training_trace.jsonl")
    assert len(trace) == 1152 and Counter(row["task"] for row in trace) == dict.fromkeys(TASKS, 384)
    assert all(
        all(math.isfinite(row[key]) for key in ("loss", "gradient_norm", "encoder_gradient_norm", "scorer_gradient_norm", "encoder_lr"))
        and row["encoder_gradient_norm"] > 0 and row["scorer_gradient_norm"] > 0
        for row in trace
    )
    records = load_lines(path / "dev_epoch3_predictions.jsonl")
    assert len(records) == 7353 and all(row["output_valid"] for row in records)
    assert all(abs(sum(row["probs"]) - 1) < 1e-5 for row in records)
    grouped = {task: by_task(records, task) for task in TASKS}
    assert [len(grouped[task]) for task in TASKS] == [1052, 939, 5362]
    metrics = {
        "promoter": class_metrics(list(grouped["promoter"].values()), "promoter"),
        "structural_class": class_metrics(list(grouped["structural_class"].values()), "structural_class"),
        "fluorescence": score_metrics(list(grouped["fluorescence"].values()), config["score_spec"]),
    }
    reverse = by_task(load_lines(path / "dev_reversed_choice_predictions.jsonl"), "structural_class")
    assert set(reverse) == set(grouped["structural_class"])
    metrics["structural_class"]["reverse_accuracy"] = float(accuracy_score(
        [reverse[key]["label"] for key in reverse], [reverse[key]["prediction"] for key in reverse]
    ))
    metrics["structural_class"]["order_agreement"] = sum(
        grouped["structural_class"][key]["prediction"] == reverse[key]["prediction"] for key in reverse
    ) / len(reverse)
    frozen = [row for row in load_lines(path / "learning_curve.jsonl") if row["epoch"] == 3 and row["split"] == "dev"]
    assert len(frozen) == 1
    for task in TASKS:
        for key in ("accuracy", "macro_f1") if task != "fluorescence" else ("rmse", "mae", "spearman"):
            assert math.isclose(metrics[task][key], frozen[0]["metrics"][task][key], abs_tol=1e-10)
    gates = {
        task: metrics[task]["accuracy"] > metrics[task]["constant_accuracy"]
        and metrics[task]["macro_f1"] > metrics[task]["constant_macro_f1"]
        for task in TASKS[:2]
    }
    gates["fluorescence"] = (
        metrics["fluorescence"]["rmse"] < metrics["fluorescence"]["constant_mean_rmse"]
        and metrics["fluorescence"]["mae"] < metrics["fluorescence"]["constant_median_mae"]
    )
    return {"config": config, "metrics": metrics, "gates": gates, "ids": grouped,
            "audit": {"checkpoint_sha256": status["checkpoint_sha256"], "updates": len(trace),
                      "checkpoint_reload_exact": True, "all_gradients_finite_nonzero": True,
                      "output_valid_fraction": 1.0, "test_inference": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write machine-readable summary JSON")
    args = parser.parse_args()
    runs = {(seed, arm): summarize_run(run_dir(seed, arm)) for seed in SEEDS for arm in ARMS}
    reference = runs[(SEEDS[0], ARMS[0])]["config"]
    for seed in SEEDS:
        no = runs[(seed, "no_cpt")]
        cpt = runs[(seed, "cpt")]
        for key in ("initial_shared_head_sha256", "batch_order_sha256_by_task", "training_panel_order_sha256", "data_manifest_sha256", "initial_model_sha256", "score_spec"):
            assert no["config"][key] == cpt["config"][key]
        for arm in ARMS:
            run = runs[(seed, arm)]
            assert run["config"]["seed"] == seed
            for key in ("initial_shared_head_sha256", "data_manifest_sha256", "initial_model_sha256", "score_spec"):
                assert run["config"][key] == reference[key]
            assert sha(run_dir(seed, arm) / "dev_epoch0_predictions.jsonl") == sha(run_dir(SEEDS[0], arm) / "dev_epoch0_predictions.jsonl")
    for task in TASKS:
        reference_ids = runs[(SEEDS[0], ARMS[0])]["ids"][task]
        for run in runs.values():
            current = run["ids"][task]
            assert set(current) == set(reference_ids)
            assert all(current[key]["label"] == reference_ids[key]["label"] for key in current)
            if task == "fluorescence":
                assert all(current[key]["value"] == reference_ids[key]["value"] for key in current)
    output = {"seeds": {}, "aggregate": {}, "paired_cpt_minus_no_cpt": {},
              "audit": {"status": "pass", "all_six_weights_hashed": True,
                        "all_traces_finite_and_nonzero": True, "all_outputs_valid": True,
                        "paired_initial_head_schedule_panel_hashes_match": True,
                        "initial_dev_predictions_identical_across_seeds_per_arm": True,
                        "test_inference": False}}
    for seed in SEEDS:
        output["seeds"][str(seed)] = {}
        for arm in ARMS:
            run = runs[(seed, arm)]
            output["seeds"][str(seed)][arm] = {"metrics": run["metrics"], "gates": run["gates"], "audit": run["audit"]}
    selections = {
        "promoter_accuracy": lambda run: run["metrics"]["promoter"]["accuracy"],
        "promoter_macro_f1": lambda run: run["metrics"]["promoter"]["macro_f1"],
        "structure_accuracy": lambda run: run["metrics"]["structural_class"]["accuracy"],
        "structure_macro_f1": lambda run: run["metrics"]["structural_class"]["macro_f1"],
        "structure_order_agreement": lambda run: run["metrics"]["structural_class"]["order_agreement"],
        "gfp_rmse": lambda run: run["metrics"]["fluorescence"]["rmse"],
        "gfp_mae": lambda run: run["metrics"]["fluorescence"]["mae"],
        "gfp_spearman": lambda run: run["metrics"]["fluorescence"]["spearman"],
    }
    for metric, select in selections.items():
        output["aggregate"][metric] = {
            arm: {"mean": statistics.mean(select(runs[(seed, arm)]) for seed in SEEDS),
                  "sample_sd": statistics.stdev(select(runs[(seed, arm)]) for seed in SEEDS)}
            for arm in ARMS
        }
        output["paired_cpt_minus_no_cpt"][metric] = {
            "by_seed": {str(seed): select(runs[(seed, "cpt")]) - select(runs[(seed, "no_cpt")]) for seed in SEEDS},
            "mean": statistics.mean(select(runs[(seed, "cpt")]) - select(runs[(seed, "no_cpt")]) for seed in SEEDS),
        }
    result = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        assert not args.output.exists(), f"Refusing to overwrite: {args.output}"
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
