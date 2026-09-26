"""Audit and summarize JEV GFP-only single-task seed controls."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
PRIOR = PROJECT / "artifacts/laya_jev_multitask_v1/round"
SINGLE_CURRENT = PROJECT / "artifacts/laya_jev_gfp_single_seed_controls/run/round"
JOINT_CURRENT = PROJECT / "artifacts/laya_jev_multitask_seed_confirm/run/round"
SEEDS = (20260926, 20260927, 20260928)
ARMS = ("no_cpt", "cpt")
TASKS = ("promoter", "structural_class", "fluorescence")
COLLAPSE_SD = 0.05


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single_dir(seed, arm):
    if seed == 20260926 and arm == "cpt":
        return PRIOR / "cpt_fluorescence"
    return SINGLE_CURRENT / f"seed_{seed}_{arm}_gfp_single"


def joint_dir(seed, arm):
    if seed == 20260926:
        return PRIOR / f"{arm}_joint"
    return JOINT_CURRENT / f"seed_{seed}_{arm}_joint"


def fluorescence(records):
    selected = [row for row in records if row["task"] == "fluorescence"]
    assert len(selected) == sum(row["task"] == "fluorescence" for row in records)
    return selected


def by_id(records):
    out = {row["id"]: row for row in records}
    assert len(out) == len(records)
    return out


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


def assert_trace(path, expected_updates):
    trace = load_jsonl(path / "training_trace.jsonl")
    assert len(trace) == expected_updates
    assert Counter(row["task"] for row in trace) == {"fluorescence": expected_updates}
    for row in trace:
        for key in ("loss", "gradient_norm", "encoder_gradient_norm", "scorer_gradient_norm", "encoder_lr"):
            assert math.isfinite(row[key]), (path, key, row)
        assert row["encoder_gradient_norm"] > 0, (path, row)
        assert row["scorer_gradient_norm"] > 0, (path, row)
    return trace


def final_learning_curve(path, split):
    rows = [row for row in load_jsonl(path / "learning_curve.jsonl") if row["epoch"] == 3 and row["split"] == split]
    assert len(rows) == 1
    return rows[0]["metrics"]["fluorescence"]


def summarize_single(seed, arm):
    path = single_dir(seed, arm)
    status = load_json(path / "status.json")
    config = load_json(path / "run_config.json")
    assert status["status"] == "complete" and status["updates"] == 384
    assert status["checkpoint_reload_exact"] and not status["test_inference"]
    assert config["seed"] == seed and config["arm"] == arm and config["task"] == "fluorescence"
    assert config["test_inference"] is False
    assert status["presentations_by_task"] == {"promoter": 0, "structural_class": 0, "fluorescence": 24576}
    assert status["per_entity_exposure_histogram"] == {"3": 8192}
    if status["model_retained"]:
        assert sha(path / "model.safetensors") == status["checkpoint_sha256"]
    assert_trace(path, 384)
    dev_records = fluorescence(load_jsonl(path / "dev_epoch3_predictions.jsonl"))
    train_records = fluorescence(load_jsonl(path / "train_epoch3_predictions.jsonl"))
    assert len(dev_records) == 5362 and len(train_records) == 1024
    assert all(row["output_valid"] for row in dev_records + train_records)
    dev_metrics = score_metrics(dev_records, config["score_spec"])
    train_metrics = score_metrics(train_records, config["score_spec"])
    frozen_dev = final_learning_curve(path, "dev")
    for key in ("rmse", "mae", "spearman"):
        assert math.isclose(dev_metrics[key], frozen_dev[key], abs_tol=1e-10)
    gate = dev_metrics["rmse"] < dev_metrics["constant_mean_rmse"] and dev_metrics["mae"] < dev_metrics["constant_median_mae"]
    return {
        "path": str(path),
        "config": config,
        "dev": dev_metrics,
        "train": train_metrics,
        "gate": gate,
        "near_constant_dev": dev_metrics["prediction_sd"] < COLLAPSE_SD,
        "near_constant_train": train_metrics["prediction_sd"] < COLLAPSE_SD,
        "audit": {
            "checkpoint_sha256": status["checkpoint_sha256"],
            "model_retained": status["model_retained"],
            "updates": 384,
            "checkpoint_reload_exact": True,
            "all_gradients_finite_nonzero": True,
            "test_inference": False,
        },
        "epoch0_dev_by_id": by_id(fluorescence(load_jsonl(path / "dev_epoch0_predictions.jsonl"))),
        "dev_by_id": by_id(dev_records),
    }


def summarize_joint(seed, arm):
    path = joint_dir(seed, arm)
    status = load_json(path / "status.json")
    config = load_json(path / "run_config.json")
    assert status["status"] == "complete" and status["updates"] == 1152
    assert status["checkpoint_reload_exact"] and status["model_retained"]
    assert not status["test_inference"] and not config["test_inference"]
    assert config["seed"] == seed and config["arm"] == arm and config["task"] == "joint"
    trace = load_jsonl(path / "training_trace.jsonl")
    assert len(trace) == 1152 and Counter(row["task"] for row in trace) == dict.fromkeys(TASKS, 384)
    dev_records = fluorescence(load_jsonl(path / "dev_epoch3_predictions.jsonl"))
    train_records = fluorescence(load_jsonl(path / "train_epoch3_predictions.jsonl"))
    dev_metrics = score_metrics(dev_records, config["score_spec"])
    train_metrics = score_metrics(train_records, config["score_spec"])
    gate = dev_metrics["rmse"] < dev_metrics["constant_mean_rmse"] and dev_metrics["mae"] < dev_metrics["constant_median_mae"]
    return {
        "path": str(path),
        "config": config,
        "dev": dev_metrics,
        "train": train_metrics,
        "gate": gate,
        "near_constant_dev": dev_metrics["prediction_sd"] < COLLAPSE_SD,
        "near_constant_train": train_metrics["prediction_sd"] < COLLAPSE_SD,
        "epoch0_dev_by_id": by_id(fluorescence(load_jsonl(path / "dev_epoch0_predictions.jsonl"))),
        "dev_by_id": by_id(dev_records),
    }


def compare_epoch0(single, joint):
    s0 = single["epoch0_dev_by_id"]
    j0 = joint["epoch0_dev_by_id"]
    assert set(s0) == set(j0)
    for key in s0:
        left = s0[key]
        right = j0[key]
        assert left["value"] == right["value"]
        assert left["label"] == right["label"]
        assert left["probs"] == right["probs"]
        assert left["score_prediction"] == right["score_prediction"]


def compact_run(run):
    return {key: run[key] for key in ("path", "dev", "train", "gate", "near_constant_dev", "near_constant_train", "audit") if key in run}


def diff(single, joint):
    return {
        "rmse": single["dev"]["rmse"] - joint["dev"]["rmse"],
        "mae": single["dev"]["mae"] - joint["dev"]["mae"],
        "spearman": single["dev"]["spearman"] - joint["dev"]["spearman"],
        "prediction_sd": single["dev"]["prediction_sd"] - joint["dev"]["prediction_sd"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write machine-readable JSON summary")
    args = parser.parse_args()
    singles = {(seed, arm): summarize_single(seed, arm) for seed in SEEDS for arm in ARMS}
    joints = {(seed, arm): summarize_joint(seed, arm) for seed in SEEDS for arm in ARMS}

    reference_score = singles[(20260926, "no_cpt")]["config"]["score_spec"]
    output = {
        "collapse_prediction_sd_threshold": COLLAPSE_SD,
        "seeds": {},
        "aggregate": {},
        "audit": {
            "status": "pass",
            "single_runs": 6,
            "new_single_runs": 5,
            "reused_single_runs": 1,
            "matched_epoch0_fluorescence_predictions": True,
            "matched_score_spec": True,
            "matched_fluorescence_batch_order": True,
            "test_inference": False,
        },
    }

    for seed in SEEDS:
        output["seeds"][str(seed)] = {}
        for arm in ARMS:
            single = singles[(seed, arm)]
            joint = joints[(seed, arm)]
            assert single["config"]["score_spec"] == joint["config"]["score_spec"] == reference_score
            assert single["config"]["initial_shared_head_sha256"] == joint["config"]["initial_shared_head_sha256"]
            assert single["config"]["data_manifest_sha256"] == joint["config"]["data_manifest_sha256"]
            assert single["config"]["initial_model_sha256"] == joint["config"]["initial_model_sha256"]
            assert single["config"]["batch_order_sha256_by_task"]["fluorescence"] == joint["config"]["batch_order_sha256_by_task"]["fluorescence"]
            if arm == "cpt":
                assert single["config"]["cpt_checkpoint_sha256"] == joint["config"]["cpt_checkpoint_sha256"]
            compare_epoch0(single, joint)
            assert set(single["dev_by_id"]) == set(joint["dev_by_id"])
            assert all(single["dev_by_id"][key]["value"] == joint["dev_by_id"][key]["value"] for key in single["dev_by_id"])
            output["seeds"][str(seed)][arm] = {
                "single": compact_run(single),
                "joint": compact_run(joint),
                "single_minus_joint": diff(single, joint),
            }

    for metric in ("rmse", "mae", "spearman", "prediction_sd"):
        output["aggregate"][metric] = {}
        for arm in ARMS:
            single_values = [singles[(seed, arm)]["dev"][metric] for seed in SEEDS]
            joint_values = [joints[(seed, arm)]["dev"][metric] for seed in SEEDS]
            output["aggregate"][metric][arm] = {
                "single_mean": statistics.mean(single_values),
                "single_sample_sd": statistics.stdev(single_values),
                "joint_mean": statistics.mean(joint_values),
                "joint_sample_sd": statistics.stdev(joint_values),
                "single_minus_joint_mean": statistics.mean(s - j for s, j in zip(single_values, joint_values)),
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
