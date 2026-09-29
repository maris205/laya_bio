"""Run frozen joint-path diagnostics for NO-CPT joint training."""

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
PLAN = PROJECT / "artifacts/laya_jev_joint_path_diagnostics/EXPERIMENT_PLAN.md"
TRACKER = PROJECT / "artifacts/laya_jev_joint_path_diagnostics/EXPERIMENT_TRACKER.md"

VARIANTS = (
    "task_block_gfp_last",
    "task_block_gfp_first",
    "task_block_gfp_last_fluo0p5",
    "task_block_gfp_last_fluo2p0",
    "task_block_gfp_last_blockeval",
    "task_block_gfp_first_blockeval",
    "round_robin_k8_gfp_last",
    "round_robin_k32_gfp_last",
    "task_block_gfp_first_lr_of_last",
    "task_block_gfp_last_lr_of_first",
    "interleave_cls_warmup64",
    "interleave_cls_warmup128",
    "hybrid_warmup64_final_block_gfp_last",
    "interleave_cls_warmup64_ep4",
    "interleave_cls_warmup64_fluo2p0",
)
ORDER_LAST = ("promoter", "structural_class", "fluorescence")
ORDER_FIRST = ("fluorescence", "promoter", "structural_class")
BLOCK_STEPS = 128
MATRICES = {
    "weighting": (
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_last_fluo0p5"},
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_last_fluo2p0"},
    ),
    "round_robin": (
        {"seed": 20260928, "arm": "no_cpt", "variant": "round_robin_k8_gfp_last", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "round_robin_k32_gfp_last", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "round_robin_k8_gfp_last", "rep": 2},
        {"seed": 20260928, "arm": "no_cpt", "variant": "round_robin_k32_gfp_last", "rep": 2},
    ),
    "lr_swap": (
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_first_lr_of_last", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_last_lr_of_first", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_first_lr_of_last", "rep": 2},
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_last_lr_of_first", "rep": 2},
    ),
    "cls_warmup": (
        {"seed": 20260928, "arm": "no_cpt", "variant": "interleave_cls_warmup64", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "interleave_cls_warmup128", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "interleave_cls_warmup64", "rep": 2},
        {"seed": 20260928, "arm": "no_cpt", "variant": "interleave_cls_warmup128", "rep": 2},
    ),
    "cls_warmup_xseed": (
        {"seed": 20260926, "arm": "no_cpt", "variant": "interleave_cls_warmup64", "rep": 1},
        {"seed": 20260927, "arm": "no_cpt", "variant": "interleave_cls_warmup64", "rep": 1},
        {"seed": 20260926, "arm": "no_cpt", "variant": "interleave_cls_warmup64", "rep": 2},
        {"seed": 20260927, "arm": "no_cpt", "variant": "interleave_cls_warmup64", "rep": 2},
    ),
    "hybrid": (
        {"seed": 20260927, "arm": "no_cpt", "variant": "hybrid_warmup64_final_block_gfp_last", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "hybrid_warmup64_final_block_gfp_last", "rep": 1},
        {"seed": 20260927, "arm": "no_cpt", "variant": "hybrid_warmup64_final_block_gfp_last", "rep": 2},
        {"seed": 20260926, "arm": "no_cpt", "variant": "hybrid_warmup64_final_block_gfp_last", "rep": 1},
    ),
    "gfp_last_xseed": (
        {"seed": 20260927, "arm": "no_cpt", "variant": "task_block_gfp_last_blockeval", "rep": 2},
        {"seed": 20260926, "arm": "no_cpt", "variant": "task_block_gfp_last_blockeval", "rep": 1},
        {"seed": 20260926, "arm": "no_cpt", "variant": "task_block_gfp_last_blockeval", "rep": 2},
    ),
    "gfp_dose": (
        {"seed": 20260927, "arm": "no_cpt", "variant": "interleave_cls_warmup64_ep4", "rep": 1},
        {"seed": 20260927, "arm": "no_cpt", "variant": "interleave_cls_warmup64_fluo2p0", "rep": 1},
        {"seed": 20260927, "arm": "no_cpt", "variant": "interleave_cls_warmup64_ep4", "rep": 2},
        {"seed": 20260927, "arm": "no_cpt", "variant": "interleave_cls_warmup64_fluo2p0", "rep": 2},
    ),
    "blockeval": (
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_first_blockeval"},
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_last_blockeval"},
    ),
    "replicate": (
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_last", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_first", "rep": 1},
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_last", "rep": 2},
        {"seed": 20260928, "arm": "no_cpt", "variant": "task_block_gfp_first", "rep": 2},
    ),
}
JOBS = MATRICES["weighting"]


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


def run_name(job):
    name = f"seed_{job['seed']}_{job['arm']}_joint_{job['variant']}"
    return f"{name}_rep{job['rep']}" if "rep" in job else name


def patch_task_block(module, order, lr_order=None):
    """Per-epoch task blocks. With `lr_order`, each update's schedule step (which sets its LR) is the step the
    same (epoch, task, index) update would occupy under `lr_order`; data and update order are unchanged."""
    assert sorted(order) == sorted(module.TASKS), order
    lr_order = lr_order or order
    assert sorted(lr_order) == sorted(module.TASKS), lr_order

    def schedule(rows):
        by = {task: [row for row in rows if row["task"] == task] for task in module.TASKS}
        assert len({len(value) for value in by.values()}) == 1
        assert all(len(value) % module.BATCH == 0 for value in by.values())
        task_batches = len(by[module.TASKS[0]]) // module.BATCH
        rng = module.random.Random(module.SEED + 41)
        step = 0
        for epoch in range(1, module.EPOCHS + 1):
            for task in module.TASKS:
                rng.shuffle(by[task])
            for task in order:
                for index in range(task_batches):
                    step += 1
                    lr_step = (epoch - 1) * len(order) * task_batches + lr_order.index(task) * task_batches + index + 1
                    start = index * module.BATCH
                    yield lr_step, epoch, task, by[task][start : start + module.BATCH]

    module.schedule = schedule


def patch_round_robin(module, order, chunk):
    """Rotate tasks every `chunk` batches in a fixed cycle; per-task shuffles match patch_task_block."""
    assert sorted(order) == sorted(module.TASKS), order

    def schedule(rows):
        by = {task: [row for row in rows if row["task"] == task] for task in module.TASKS}
        assert len({len(value) for value in by.values()}) == 1
        assert all(len(value) % module.BATCH == 0 for value in by.values())
        task_batches = len(by[module.TASKS[0]]) // module.BATCH
        assert task_batches % chunk == 0
        rng = module.random.Random(module.SEED + 41)
        step = 0
        for epoch in range(1, module.EPOCHS + 1):
            for task in module.TASKS:
                rng.shuffle(by[task])
            for first in range(0, task_batches, chunk):
                for task in order:
                    for index in range(first, first + chunk):
                        step += 1
                        start = index * module.BATCH
                        yield step, epoch, task, by[task][start : start + module.BATCH]

    module.schedule = schedule


def patch_cls_warmup(module, per_task):
    """Frozen random interleave, except the first `per_task` promoter and structural_class batches of epoch 1
    move to the front (relative order kept). Same batches, updates, exposures and LR curve."""
    original = module.schedule

    def schedule(rows):
        items = [(epoch, task, batch) for _, epoch, task, batch in original(rows)]
        first = [item for item in items if item[0] == 1]
        counts = {"promoter": 0, "structural_class": 0}
        front, rest = [], []
        for item in first:
            if item[1] in counts and counts[item[1]] < per_task:
                counts[item[1]] += 1
                front.append(item)
            else:
                rest.append(item)
        assert all(value == per_task for value in counts.values()), counts
        reordered = front + rest + [item for item in items if item[0] > 1]
        assert len(reordered) == len(items)
        for step, (epoch, task, batch) in enumerate(reordered, 1):
            yield step, epoch, task, batch

    module.schedule = schedule


def patch_final_epoch_blocks(module, order):
    """Keep the current schedule for earlier epochs; regroup the final epoch's batches into task blocks in
    `order`, each task keeping its own batch order. Same batches, updates, exposures and LR curve."""
    assert sorted(order) == sorted(module.TASKS), order
    original = module.schedule

    def schedule(rows):
        items = [(epoch, task, batch) for _, epoch, task, batch in original(rows)]
        early = [item for item in items if item[0] < module.EPOCHS]
        final = [item for item in items if item[0] == module.EPOCHS]
        blocks = [item for task in order for item in final if item[1] == task]
        assert len(blocks) == len(final)
        for step, (epoch, task, batch) in enumerate(early + blocks, 1):
            yield step, epoch, task, batch

    module.schedule = schedule


def add_block_eval(module, output, single_task_blocks=True):
    """After every task block, score the fixed train-diagnostic panel; training itself is unchanged."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    rows = module.read_lines(DATA / manifest["outputs"]["train"]["file"])
    panel = []
    for task in module.TASKS:
        pool = sorted([r for r in rows if r["task"] == task], key=lambda r: module.digest("train-eval:" + r["id"]))
        panel.extend(pool[:1024])
    original_step = module.train_step
    state = {"calls": 0, "tasks": []}

    def train_step(model, opt, rows, pad, micro, spec):
        result = original_step(model, opt, rows, pad, micro, spec)
        state["calls"] += 1
        state["tasks"].append(rows[0]["task"])
        smoke = len(set(state["tasks"])) == len(state["tasks"])  # smoke runs one update per task
        if state["calls"] % BLOCK_STEPS == 0 and not smoke:
            block_tasks = set(state["tasks"][-BLOCK_STEPS:])
            assert len(block_tasks) == 1 or not single_task_blocks, block_tasks
            metrics = module.summarize(module.evaluate(model, panel, pad, 16, spec), spec)
            record = {"update": state["calls"], "block": state["calls"] // BLOCK_STEPS, "epoch": (state["calls"] - 1) // (3 * BLOCK_STEPS) + 1,
                      "trained_task": block_tasks.pop() if len(block_tasks) == 1 else "mixed", "split": "train_diagnostic",
                      "metrics": {t: ({k: v[k] for k in ("rmse", "mae", "spearman", "prediction_std")} if t == "fluorescence"
                                      else {k: v[k] for k in ("accuracy", "macro_f1")}) for t, v in metrics.items()}}
            module.append(output / "block_eval.jsonl", record)
        return result

    module.train_step = train_step


def epochs_of(variant):
    return 4 if variant.endswith("_ep4") else 3


def scale_fluorescence_loss(module, scale):
    original_losses = module.losses

    def weighted_losses(logits, rows, score_spec):
        values = original_losses(logits, rows, score_spec)
        factor = values.new_tensor([scale if row["task"] == "fluorescence" else 1.0 for row in rows])
        return values * factor

    module.losses = weighted_losses


def apply_variant(module, variant):
    module.EPOCHS = epochs_of(variant)
    if variant == "task_block_gfp_last":
        patch_task_block(module, ("promoter", "structural_class", "fluorescence"))
    elif variant == "task_block_gfp_first":
        patch_task_block(module, ("fluorescence", "promoter", "structural_class"))
    elif variant in {"task_block_gfp_last_fluo0p5", "task_block_gfp_last_fluo2p0"}:
        patch_task_block(module, ("promoter", "structural_class", "fluorescence"))
        scale = 0.5 if variant.endswith("fluo0p5") else 2.0
        original_losses = module.losses
        def weighted_losses(logits, rows, score_spec):
            values = original_losses(logits, rows, score_spec)
            factor = values.new_tensor([scale if row['task'] == 'fluorescence' else 1.0 for row in rows])
            return values * factor
        module.losses = weighted_losses
    elif variant.startswith("round_robin_k"):
        chunk = int(variant.removeprefix("round_robin_k").split("_", 1)[0])
        patch_round_robin(module, ("promoter", "structural_class", "fluorescence"), chunk)
    elif variant == "task_block_gfp_first_lr_of_last":
        patch_task_block(module, ORDER_FIRST, lr_order=ORDER_LAST)
    elif variant == "task_block_gfp_last_lr_of_first":
        patch_task_block(module, ORDER_LAST, lr_order=ORDER_FIRST)
    elif variant == "hybrid_warmup64_final_block_gfp_last":
        patch_cls_warmup(module, 64)
        patch_final_epoch_blocks(module, ("promoter", "structural_class", "fluorescence"))
    elif variant in {"interleave_cls_warmup64_ep4", "interleave_cls_warmup64_fluo2p0"}:
        patch_cls_warmup(module, 64)
        if variant.endswith("fluo2p0"):
            scale_fluorescence_loss(module, 2.0)
    elif variant.startswith("interleave_cls_warmup"):
        patch_cls_warmup(module, int(variant.removeprefix("interleave_cls_warmup")))
    elif variant == "task_block_gfp_last_blockeval":
        patch_task_block(module, ("promoter", "structural_class", "fluorescence"))
    elif variant == "task_block_gfp_first_blockeval":
        patch_task_block(module, ("fluorescence", "promoter", "structural_class"))
    else:
        raise ValueError(f"unknown diagnostic variant: {variant}")


def injected_train(seed, variant, args):
    assert variant in VARIANTS
    sys.path.insert(0, str(SOURCE.parent))
    spec = importlib.util.spec_from_file_location("frozen_jev_train", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEED = seed
    apply_variant(module, variant)
    if variant.endswith("_blockeval") or "_lr_of_" in variant:
        add_block_eval(module, Path(args[args.index("--output") + 1]))
    elif variant.startswith(("round_robin_k", "interleave_cls_warmup", "hybrid_")):
        add_block_eval(module, Path(args[args.index("--output") + 1]), single_task_blocks=False)
    sys.argv = [str(SOURCE), *args]
    module.main()


def build_command(root, phase, job):
    output = root / phase / run_name(job)
    args = [
        "--model-dir",
        str(MODEL),
        "--cpt-root",
        str(CPT),
        "--data-dir",
        str(DATA),
        "--output",
        str(output),
        "--arm",
        job["arm"],
        "--task",
        "joint",
    ]
    args.append("--smoke" if phase == "smoke" else "--retain-model")
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--seed",
        str(job["seed"]),
        "--variant",
        job["variant"],
        *args,
    ]


def run_one(root, phase, job):
    name = run_name(job)
    write_json(root / "status.json", {"status": "running", "phase": phase, "current": name})
    command = build_command(root, phase, job)
    with (root / phase / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(
            root / "status.json",
            {"status": "failed", "phase": phase, "current": name, "returncode": result.returncode},
        )
        raise RuntimeError(f"{name} failed; inspect {root / phase / (name + '.log')}")
    output = root / phase / name
    status = load_json(output / "status.json")
    expected = 3 if phase == "smoke" else 384 * epochs_of(job["variant"])
    assert status["status"] == "complete" and status["updates"] == expected
    assert status["checkpoint_reload_exact"] and not status["test_inference"]
    trace = [json.loads(line) for line in (output / "training_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(trace) == expected
    if phase == "smoke":
        assert {row["task"] for row in trace} == {"promoter", "structural_class", "fluorescence"}
        assert not status["model_retained"]
    else:
        assert status["model_retained"]
        assert sha(output / "model.safetensors") == status["checkpoint_sha256"]
    print(json.dumps({"completed": name, "phase": phase, "elapsed_seconds": status["elapsed_seconds"]}), flush=True)


def controller(root, jobs):
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
        "diagnostic_runner": sha(Path(__file__)),
        "plan": sha(PLAN),
        "tracker": sha(TRACKER),
    }
    assert actual_hashes["training_data_manifest"] == prior_manifest["data_manifest_sha256"]
    assert actual_hashes["initial_model"] == prior_manifest["initial_model_sha256"]
    assert actual_hashes["cpt_model"] == prior_manifest["cpt_checkpoint_sha256"]
    assert actual_hashes["frozen_trainer"] == prior_manifest["code_sha256"][SOURCE.name]

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
            "jobs": jobs,
            "source_round": str(PRIOR),
            "test_inference": False,
        },
    )
    smoke_jobs = {(job["seed"], job["variant"]): job for job in jobs}
    for job in smoke_jobs.values():
        run_one(root, "smoke", job)
    for job in jobs:
        run_one(root, "round", job)
    write_json(
        root / "status.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "formal_runs": len(jobs),
            "test_inference": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "artifacts/laya_jev_joint_path_diagnostics/run_weighting_v2")
    parser.add_argument("--matrix", choices=sorted(MATRICES), default="weighting")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=VARIANTS)
    args, training_args = parser.parse_known_args()
    if args.worker:
        injected_train(args.seed, args.variant, training_args)
    else:
        assert not training_args, training_args
        controller(args.root, MATRICES[args.matrix])


if __name__ == "__main__":
    main()
