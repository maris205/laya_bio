"""Run GFP-only single-task controls for the JEV seed-confirmation study."""

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PRIOR = PROJECT / "artifacts/laya_jev_multitask_v1"
SOURCE = PRIOR / "round/frozen_code/laya_jev_multitask_train.py"
MODEL = PROJECT / "artifacts/laya_model"
CPT = PROJECT / "artifacts/laya_biocpt_v2"
DATA = PRIOR / "data"
PLAN = PROJECT / "artifacts/laya_jev_gfp_single_seed_controls/EXPERIMENT_PLAN.md"
TRACKER = PROJECT / "artifacts/laya_jev_gfp_single_seed_controls/EXPERIMENT_TRACKER.md"

ALL_SEEDS = (20260926, 20260927, 20260928)
ARMS = ("no_cpt", "cpt")
NEW_RUNS = (
    (20260926, "no_cpt"),
    (20260927, "no_cpt"),
    (20260927, "cpt"),
    (20260928, "no_cpt"),
    (20260928, "cpt"),
)
SMOKES = ((20260927, "no_cpt"), (20260927, "cpt"))
REUSED = {(20260926, "cpt"): PRIOR / "round/cpt_fluorescence"}
TASKS = ("promoter", "structural_class", "fluorescence")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_name(seed, arm):
    return f"seed_{seed}_{arm}_gfp_single"


def injected_train(seed, args):
    assert seed in ALL_SEEDS
    sys.path.insert(0, str(SOURCE.parent))
    spec = importlib.util.spec_from_file_location("frozen_jev_train", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEED = seed
    sys.argv = [str(SOURCE), *args]
    module.main()


def assert_finite_positive_trace(trace):
    assert trace, "empty trace"
    for row in trace:
        for key in ("loss", "gradient_norm", "encoder_gradient_norm", "scorer_gradient_norm", "encoder_lr"):
            assert math.isfinite(row[key]), (key, row)
        assert row["encoder_gradient_norm"] > 0, row
        assert row["scorer_gradient_norm"] > 0, row


def verify_reused_single(path):
    status = load_json(path / "status.json")
    config = load_json(path / "run_config.json")
    assert status["status"] == "complete" and status["updates"] == 384
    assert status["checkpoint_reload_exact"] and not status["test_inference"]
    assert config["arm"] == "cpt" and config["seed"] == 20260926 and config["task"] == "fluorescence"
    assert config["test_inference"] is False
    assert status["presentations_by_task"] == {"promoter": 0, "structural_class": 0, "fluorescence": 24576}
    assert status["per_entity_exposure_histogram"] == {"3": 8192}
    trace = load_jsonl(path / "training_trace.jsonl")
    assert len(trace) == 384 and {row["task"] for row in trace} == {"fluorescence"}
    assert_finite_positive_trace(trace)
    assert (path / "dev_epoch3_predictions.jsonl").exists()
    assert (path / "train_epoch3_predictions.jsonl").exists()


def build_command(seed, arm, output, smoke):
    args = [
        "--model-dir", str(MODEL),
        "--cpt-root", str(CPT),
        "--data-dir", str(DATA),
        "--output", str(output),
        "--arm", arm,
        "--task", "fluorescence",
    ]
    args.append("--smoke" if smoke else "--retain-model")
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--seed", str(seed), *args]


def run_one(root, phase, seed, arm):
    name = run_name(seed, arm)
    output = root / phase / name
    command = build_command(seed, arm, output, phase == "smoke")
    write_json(root / "status.json", {"status": "running", "phase": phase, "current": name})
    with (root / phase / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(
            root / "status.json",
            {"status": "failed", "phase": phase, "current": name, "returncode": result.returncode},
        )
        raise RuntimeError(f"{name} failed; inspect {root / phase / (name + '.log')}")
    status = load_json(output / "status.json")
    trace = load_jsonl(output / "training_trace.jsonl")
    if phase == "smoke":
        assert status["status"] == "complete" and status["updates"] == 1
        assert status["checkpoint_reload_exact"] and not status["model_retained"]
        assert len(trace) == 1 and trace[0]["task"] == "fluorescence"
        assert status["presentations_by_task"]["fluorescence"] == 64
    else:
        assert status["status"] == "complete" and status["updates"] == 384
        assert status["checkpoint_reload_exact"] and status["model_retained"]
        assert sha(output / "model.safetensors") == status["checkpoint_sha256"]
        assert len(trace) == 384 and {row["task"] for row in trace} == {"fluorescence"}
        assert status["presentations_by_task"] == {"promoter": 0, "structural_class": 0, "fluorescence": 24576}
        assert status["per_entity_exposure_histogram"] == {"3": 8192}
    assert not status["test_inference"]
    assert_finite_positive_trace(trace)
    print(json.dumps({"completed": name, "phase": phase, "elapsed_seconds": status["elapsed_seconds"]}), flush=True)


def controller(root):
    root.parent.mkdir(parents=True, exist_ok=True)
    assert not root.exists(), f"Refusing to reuse output directory: {root}"
    assert PLAN.exists() and TRACKER.exists()
    assert shutil.disk_usage(root.parent).free > 20 * 2**30
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    assert len(gpu.splitlines()) == 1 and int(gpu.strip()) < 500, gpu

    prior_manifest = load_json(PRIOR / "round/manifest.json")
    actual_hashes = {
        "training_data_manifest": sha(DATA / "manifest.json"),
        "initial_model": sha(MODEL / "model.safetensors"),
        "cpt_model": sha(CPT / "round/cpt/model.safetensors"),
        "frozen_trainer": sha(SOURCE),
        "control_runner": sha(Path(__file__)),
        "plan": sha(PLAN),
        "tracker": sha(TRACKER),
    }
    assert actual_hashes["training_data_manifest"] == prior_manifest["data_manifest_sha256"]
    assert actual_hashes["initial_model"] == prior_manifest["initial_model_sha256"]
    assert actual_hashes["cpt_model"] == prior_manifest["cpt_checkpoint_sha256"]
    assert actual_hashes["frozen_trainer"] == prior_manifest["code_sha256"][SOURCE.name]

    for path in REUSED.values():
        verify_reused_single(path)

    root.mkdir()
    (root / "smoke").mkdir()
    (root / "round").mkdir()
    shutil.copy2(PLAN, root / "EXPERIMENT_PLAN.md")
    shutil.copy2(TRACKER, root / "EXPERIMENT_TRACKER.md")
    write_json(
        root / "manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "hashes": actual_hashes,
            "new_runs": [{"seed": seed, "arm": arm, "task": "fluorescence"} for seed, arm in NEW_RUNS],
            "smokes": [{"seed": seed, "arm": arm, "task": "fluorescence"} for seed, arm in SMOKES],
            "reused_runs": {f"{seed}_{arm}": str(path) for (seed, arm), path in REUSED.items()},
            "source_round": str(PRIOR),
            "test_inference": False,
        },
    )

    for seed, arm in SMOKES:
        run_one(root, "smoke", seed, arm)
    for seed, arm in NEW_RUNS:
        run_one(root, "round", seed, arm)
    write_json(
        root / "status.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "new_formal_runs": len(NEW_RUNS),
            "reused_formal_runs": len(REUSED),
            "test_inference": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "artifacts/laya_jev_gfp_single_seed_controls/run")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    args, training_args = parser.parse_known_args()
    if args.worker:
        injected_train(args.seed, training_args)
    else:
        assert not training_args, training_args
        controller(args.root)


if __name__ == "__main__":
    main()
