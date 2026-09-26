"""Read-only follow-up diagnostics for the frozen three-task JEV decision round.

Prints aggregate JSON. It reads saved development predictions only and does not
load raw biological sequences, test labels, or model checkpoints.
"""

import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "artifacts/laya_jev_multitask_v1/round"
ARMS = ("no_cpt_joint", "cpt_joint")
STRUCTURE = "structural_class"
GFP = "fluorescence"


def load(arm, filename, task):
    path = ROOT / arm / filename
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["task"] != task:
                continue
            key = record["id"]
            assert key not in records, (path, key)
            records[key] = record
    return records


def same_ids_and_targets(*sets):
    ids = set(sets[0])
    assert all(set(records) == ids for records in sets[1:])
    for key in ids:
        assert len({records[key]["label"] for records in sets}) == 1
        if "value" in sets[0][key]:
            assert len({records[key]["value"] for records in sets}) == 1
    return sorted(ids)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def class_report(canonical, reversed_order, paired, ids):
    output = {}
    for label in range(7):
        subset = [key for key in ids if canonical[key]["label"] == label]
        support = len(subset)
        correct = sum(canonical[key]["prediction"] == label for key in subset)
        reverse_correct = sum(reversed_order[key]["prediction"] == label for key in subset)
        flips = sum(canonical[key]["prediction"] != reversed_order[key]["prediction"] for key in subset)
        only_this = sum(canonical[key]["prediction"] == label and paired[key]["prediction"] != label for key in subset)
        only_pair = sum(canonical[key]["prediction"] != label and paired[key]["prediction"] == label for key in subset)
        output[str(label)] = {
            "support": support,
            "canonical_correct": correct,
            "canonical_recall": ratio(correct, support),
            "canonical_precision": ratio(correct, sum(canonical[key]["prediction"] == label for key in ids)),
            "canonical_f1": ratio(
                2 * correct,
                support + sum(canonical[key]["prediction"] == label for key in ids),
            ),
            "reverse_correct": reverse_correct,
            "reverse_recall": ratio(reverse_correct, support),
            "order_flips": flips,
            "order_flip_fraction": ratio(flips, support),
            "this_only_correct_vs_other_arm": only_this,
            "other_arm_only_correct": only_pair,
        }
    return output


def regression_report(records, ids):
    errors = [records[key]["score_prediction"] - records[key]["value"] for key in ids]
    predictions = [records[key]["score_prediction"] for key in ids]
    targets = [records[key]["value"] for key in ids]
    n = len(ids)
    mean_prediction = sum(predictions) / n
    return {
        "n": n,
        "rmse": math.sqrt(sum(error * error for error in errors) / n),
        "mae": sum(abs(error) for error in errors) / n,
        "bias_pred_minus_true": sum(errors) / n,
        "true_mean": sum(targets) / n,
        "prediction_mean": mean_prediction,
        "prediction_sd": math.sqrt(sum((value - mean_prediction) ** 2 for value in predictions) / n),
        "prediction_min": min(predictions),
        "prediction_max": max(predictions),
    }


def main():
    canonical = {arm: load(arm, "dev_epoch3_predictions.jsonl", STRUCTURE) for arm in ARMS}
    reverse = {arm: load(arm, "dev_reversed_choice_predictions.jsonl", STRUCTURE) for arm in ARMS}
    structure_ids = same_ids_and_targets(*(canonical.values()), *(reverse.values()))
    assert len(structure_ids) == 939
    classes = {}
    for arm, other in ((ARMS[0], ARMS[1]), (ARMS[1], ARMS[0])):
        classes[arm] = class_report(canonical[arm], reverse[arm], canonical[other], structure_ids)
    structure = {}
    for arm in ARMS:
        own, rev = canonical[arm], reverse[arm]
        predictions = Counter(own[key]["prediction"] for key in structure_ids)
        flipped = sum(own[key]["prediction"] != rev[key]["prediction"] for key in structure_ids)
        canonical_correct = sum(own[key]["prediction"] == own[key]["label"] for key in structure_ids)
        reverse_correct = sum(rev[key]["prediction"] == rev[key]["label"] for key in structure_ids)
        structure[arm] = {
            "canonical_accuracy": canonical_correct / len(structure_ids),
            "reverse_accuracy": reverse_correct / len(structure_ids),
            "order_flips": flipped,
            "order_flip_fraction": flipped / len(structure_ids),
            "predicted_class_counts": {str(i): predictions[i] for i in range(7)},
            "by_true_class": classes[arm],
        }

    gfp = {arm: load(arm, "dev_epoch3_predictions.jsonl", GFP) for arm in ARMS}
    gfp_ids = same_ids_and_targets(*(gfp.values()))
    assert len(gfp_ids) == 5362
    spec = json.loads((ROOT / ARMS[0] / "run_config.json").read_text(encoding="utf-8"))["score_spec"]
    anchors = spec["anchors"]
    bins = [
        ("below_anchor_min", lambda value: value < anchors[0]),
        ("anchor_0_to_1", lambda value: anchors[0] <= value < anchors[1]),
        ("anchor_1_to_2", lambda value: anchors[1] <= value < anchors[2]),
        ("anchor_2_to_3", lambda value: anchors[2] <= value < anchors[3]),
        ("anchor_3_to_4", lambda value: anchors[3] <= value <= anchors[4]),
        ("above_anchor_max", lambda value: value > anchors[4]),
    ]
    gfp_report = {
        "anchors": anchors,
        "all": {arm: regression_report(gfp[arm], gfp_ids) for arm in ARMS},
        "by_true_value_bin": {},
    }
    for name, predicate in bins:
        subset = [key for key in gfp_ids if predicate(gfp[ARMS[0]][key]["value"])]
        if not subset:
            continue
        bin_result = {arm: regression_report(gfp[arm], subset) for arm in ARMS}
        bin_result["mae_delta_cpt_minus_no_cpt"] = (
            bin_result["cpt_joint"]["mae"] - bin_result["no_cpt_joint"]["mae"]
        )
        bin_result["contribution_to_total_mae_delta"] = (
            len(subset) / len(gfp_ids) * bin_result["mae_delta_cpt_minus_no_cpt"]
        )
        gfp_report["by_true_value_bin"][name] = bin_result
    no = gfp[ARMS[0]]
    cpt = gfp[ARMS[1]]
    gfp_report["paired"] = {
        "no_cpt_lower_absolute_error": sum(
            abs(no[key]["score_prediction"] - no[key]["value"])
            < abs(cpt[key]["score_prediction"] - cpt[key]["value"])
            for key in gfp_ids
        ),
        "cpt_lower_absolute_error": sum(
            abs(cpt[key]["score_prediction"] - cpt[key]["value"])
            < abs(no[key]["score_prediction"] - no[key]["value"])
            for key in gfp_ids
        ),
    }
    print(json.dumps({"structure": structure, "gfp": gfp_report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
