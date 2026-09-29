"""Compare NO-CPT joint-training recipes across seeds under one frozen success rule (dev metrics, no test)."""

import argparse
import json
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
A = PROJECT / "artifacts"
JP = A / "laya_jev_joint_path_diagnostics"

RULE = {"gfp_rmse_lt": 0.8357, "gfp_mae_lt": 0.5095, "gfp_sd_ge": 0.05, "promoter_acc_ge": 0.85, "structure_acc_ge": 0.54}


def runs():
    """(recipe, seed, run directory); only schedule-identical repeats share a recipe."""
    yield "plain_interleave", 20260926, A / "laya_jev_multitask_v1/round/no_cpt_joint"
    for seed in (20260927, 20260928):
        yield "plain_interleave", seed, A / f"laya_jev_multitask_seed_confirm/run/round/seed_{seed}_no_cpt_joint"
    yield "task_block_gfp_last", 20260928, JP / "run/round/seed_20260928_no_cpt_joint_task_block_gfp_last"
    yield "task_block_gfp_last", 20260927, JP / "run_followup/round/seed_20260927_no_cpt_joint_task_block_gfp_last"
    for rep in (1, 2):
        yield "task_block_gfp_last", 20260928, JP / f"run_replicate/round/seed_20260928_no_cpt_joint_task_block_gfp_last_rep{rep}"
    yield "task_block_gfp_last", 20260928, JP / "run_blockeval/round/seed_20260928_no_cpt_joint_task_block_gfp_last_blockeval"
    for path in sorted((JP / "run_gfp_last_xseed/round").glob("seed_*_rep*")):
        yield "task_block_gfp_last", int(path.name.split("_")[1]), path
    for path in sorted((JP / "run_cls_warmup/round").glob("seed_*_interleave_cls_warmup64_rep*")):
        yield "cls_warmup128_updates", int(path.name.split("_")[1]), path
    for path in sorted((JP / "run_cls_warmup_xseed/round").glob("seed_*_rep*")):
        yield "cls_warmup128_updates", int(path.name.split("_")[1]), path
    for path in sorted((JP / "run_cls_warmup/round").glob("seed_*_interleave_cls_warmup128_rep*")):
        yield "cls_warmup256_updates", int(path.name.split("_")[1]), path
    for path in sorted((JP / "run_hybrid/round").glob("seed_*_rep*")):
        yield "hybrid_warmup_final_blocks", int(path.name.split("_")[1]), path
    for k in (8, 32):
        for path in sorted((JP / "run_round_robin/round").glob(f"seed_*_round_robin_k{k}_gfp_last_rep*")):
            yield f"round_robin_k{k}", int(path.name.split("_")[1]), path


def final_dev(path):
    rows = [json.loads(line) for line in (path / "learning_curve.jsonl").read_text().splitlines()]
    metrics = [row for row in rows if row["split"] == "dev" and row["epoch"] == 3][0]["metrics"]
    gfp = metrics["fluorescence"]
    return {
        "gfp_rmse": gfp["rmse"], "gfp_mae": gfp["mae"], "gfp_sd": gfp["prediction_std"], "gfp_spearman": gfp["spearman"],
        "promoter_acc": metrics["promoter"]["accuracy"], "structure_acc": metrics["structural_class"]["accuracy"],
        "structure_macro_f1": metrics["structural_class"]["macro_f1"],
    }


def verdict(m):
    gfp = m["gfp_rmse"] < RULE["gfp_rmse_lt"] and m["gfp_mae"] < RULE["gfp_mae_lt"] and m["gfp_sd"] >= RULE["gfp_sd_ge"]
    return {"gfp_pass": gfp, "pass": gfp and m["promoter_acc"] >= RULE["promoter_acc_ge"] and m["structure_acc"] >= RULE["structure_acc_ge"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=JP / "analysis/recipe_comparison.json")
    args = parser.parse_args()
    table, pending = {}, []
    for recipe, seed, path in runs():
        if not path.is_dir():
            continue
        status = json.loads((path / "status.json").read_text())
        if status["status"] != "complete":
            pending.append(str(path))
            continue
        assert status["updates"] == 1152 and status["checkpoint_reload_exact"] and not status["test_inference"], path
        m = final_dev(path)
        table.setdefault(recipe, {}).setdefault(str(seed), []).append({"path": str(path.relative_to(PROJECT)), **m, **verdict(m)})
    summary = {}
    for recipe, seeds in table.items():
        per_seed = {seed: {"n": len(rs), "pass": sum(r["pass"] for r in rs), "gfp_pass": sum(r["gfp_pass"] for r in rs),
                           "gfp_mae_mean": statistics.mean(r["gfp_mae"] for r in rs),
                           "promoter_mean": statistics.mean(r["promoter_acc"] for r in rs),
                           "structure_mean": statistics.mean(r["structure_acc"] for r in rs)} for seed, rs in sorted(seeds.items())}
        all_runs = [r for rs in seeds.values() for r in rs]
        summary[recipe] = {"per_seed": per_seed, "runs": len(all_runs), "pass_rate": sum(r["pass"] for r in all_runs) / len(all_runs),
                           "gfp_pass_rate": sum(r["gfp_pass"] for r in all_runs) / len(all_runs),
                           "seeds_all_pass": [s for s, v in per_seed.items() if v["pass"] == v["n"]]}
    out = {"rule": RULE, "arm": "no_cpt", "split": "dev", "test_inference": False, "summary": summary, "runs": table, "pending": pending}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for recipe, s in summary.items():
        cells = "  ".join(f"{seed[-2:]}:{v['pass']}/{v['n']}(GFP {v['gfp_pass']}/{v['n']}, MAE {v['gfp_mae_mean']:.3f}, P {v['promoter_mean']*100:.1f}, S {v['structure_mean']*100:.1f})"
                          for seed, v in s["per_seed"].items())
        print(f"{recipe:28s} all {s['pass_rate']*100:5.1f}%  GFP {s['gfp_pass_rate']*100:5.1f}%  | {cells}")
    if pending:
        print("pending:", len(pending))


if __name__ == "__main__":
    main()
