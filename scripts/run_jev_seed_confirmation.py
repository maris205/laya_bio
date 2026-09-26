"""Run two paired seed confirmations using the immutable JEV round-1 trainer."""

import argparse
import hashlib
import importlib.util
import json
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
SEEDS = (20260927, 20260928)
ARMS = ("no_cpt", "cpt")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def injected_train(seed, args):
    assert seed in SEEDS
    sys.path.insert(0, str(SOURCE.parent))
    spec = importlib.util.spec_from_file_location("frozen_jev_train", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEED = seed
    sys.argv = [str(SOURCE), *args]
    module.main()


def controller(root):
    assert not root.exists(), f"Refusing to reuse output directory: {root}"
    assert shutil.disk_usage(root.parent).free > 20 * 2**30
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    assert len(gpu.splitlines()) == 1 and int(gpu.strip()) < 500, gpu
    prior_manifest = json.loads((PRIOR / "round/manifest.json").read_text(encoding="utf-8"))
    actual_hashes = {
        "training_data_manifest": sha(DATA / "manifest.json"),
        "initial_model": sha(MODEL / "model.safetensors"),
        "cpt_model": sha(CPT / "round/cpt/model.safetensors"),
        "frozen_trainer": sha(SOURCE),
        "seed_wrapper": sha(Path(__file__)),
    }
    assert actual_hashes["training_data_manifest"] == prior_manifest["data_manifest_sha256"]
    assert actual_hashes["initial_model"] == prior_manifest["initial_model_sha256"]
    assert actual_hashes["cpt_model"] == prior_manifest["cpt_checkpoint_sha256"]
    assert actual_hashes["frozen_trainer"] == prior_manifest["code_sha256"][SOURCE.name]
    root.mkdir()
    plan = PROJECT / "artifacts/laya_jev_multitask_seed_confirm/EXPERIMENT_PLAN.md"
    shutil.copy2(plan, root / "EXPERIMENT_PLAN.md")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(plan),
        "hashes": actual_hashes,
        "seeds": SEEDS,
        "arms": ARMS,
        "source_round": str(PRIOR),
        "test_inference": False,
    }
    write_json(root / "manifest.json", manifest)
    for phase in ("smoke", "round"):
        (root / phase).mkdir()
        for seed in SEEDS:
            for arm in ARMS:
                name = f"seed_{seed}_{arm}_joint"
                output = root / phase / name
                args = [
                    "--model-dir", str(MODEL), "--cpt-root", str(CPT),
                    "--data-dir", str(DATA), "--output", str(output),
                    "--arm", arm, "--task", "joint",
                ]
                args.append("--smoke" if phase == "smoke" else "--retain-model")
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--seed", str(seed), *args]
                write_json(root / "status.json", {"status": "running", "phase": phase, "current": name})
                with (root / phase / f"{name}.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write_json(root / "status.json", {"status": "failed", "phase": phase, "current": name, "returncode": result.returncode})
                    raise RuntimeError(f"{name} failed; inspect log")
                status = json.loads((output / "status.json").read_text(encoding="utf-8"))
                expected = 3 if phase == "smoke" else 1152
                assert status["status"] == "complete" and status["updates"] == expected
                assert status["checkpoint_reload_exact"]
                if phase == "smoke":
                    trace = [json.loads(line) for line in (output / "training_trace.jsonl").read_text().splitlines()]
                    assert {point["task"] for point in trace} == {"promoter", "structural_class", "fluorescence"}
                    assert all(point["encoder_gradient_norm"] > 0 and point["scorer_gradient_norm"] > 0 for point in trace)
                else:
                    assert status["model_retained"]
                print(json.dumps({"completed": name, "phase": phase, "elapsed_seconds": status["elapsed_seconds"]}), flush=True)
    write_json(root / "status.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "artifacts/laya_jev_multitask_seed_confirm/run")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    arguments, training_args = parser.parse_known_args()
    if arguments.worker:
        injected_train(arguments.seed, training_args)
    else:
        assert not training_args, training_args
        controller(arguments.root)


if __name__ == "__main__":
    main()
