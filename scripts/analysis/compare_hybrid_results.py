"""Compare Hybrid DiffPool vs Hybrid DMoN final_model_results and write a
markdown report.

Reads <path-output>/{diffpool,dmon}_hybrid{n_hybrid}_rep{rep}/final_model_results/
(metrics.csv, model_configs.json) -- the same directories
scripts/experiments/diffpool_experiment.py writes -- and optionally a run log
containing 'HYBRID_RERUN_TIMER <pooling_type> start|end <unix_epoch>' marker
lines (written by scripts/run_hybrid_rerun.sh) to report wall-clock time.

Usage:
    python scripts/analysis/compare_hybrid_results.py \
        --path-output outputs/hybrid_rerun --log hybrid_rerun.log \
        --out outputs/hybrid_rerun/comparison.md
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Prior documented Hybrid DiffPool result (plan.md / ARCHITECTURE.md /
# analysis-approaches.MD) -- not reproduced from a file on disk, only
# recorded in narrative docs. Used purely as a reference line in the report.
DOCUMENTED_BASELINE = {
    "pooling_type": "diffpool (documented, not reproduced from disk)",
    "test_accuracy": 0.703,
    "test_balanced_accuracy": None,
    "note": "n_hybrid=2, lr=0.0099, wd=0.0344, 127 epochs -- exact CLI config "
            "(batch size, num_samples, n_holdouts) not preserved anywhere in "
            "the repo, see plan.md / ARCHITECTURE.md / analysis-approaches.MD",
}


def load_run(path_output: Path, pooling_type: str, n_hybrid: int, rep: int):
    run_dir = path_output / f"{pooling_type}_hybrid{n_hybrid}_rep{rep}" / "final_model_results"
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "model_configs.json"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path, index_col=0)
    final = metrics.iloc[-1].to_dict()
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {
        "pooling_type": pooling_type,
        "rep": rep,
        "run_dir": run_dir,
        "n_epochs": len(metrics),
        "test_accuracy": final.get("test_accuracy"),
        "test_balanced_accuracy": final.get("test_balanced_accuracy"),
        "test_loss": final.get("test_loss"),
        "train_accuracy": final.get("train_accuracy"),
        "train_loss": final.get("train_loss"),
        "config": config,
    }


METRIC_KEYS = (
    "test_accuracy", "test_balanced_accuracy", "test_loss",
    "train_accuracy", "train_loss",
)


def load_and_aggregate(path_output: Path, pooling_type: str, n_hybrid: int, n_holdouts: int):
    """Load reps 0..n_holdouts-1 (skipping any that haven't finished yet) and
    return per-metric mean/std across the completed reps, plus the raw
    per-rep runs (each rep tunes its own hyperparameters, so configs are
    reported individually rather than averaged)."""
    runs = [load_run(path_output, pooling_type, n_hybrid, rep) for rep in range(n_holdouts)]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    df = pd.DataFrame(runs)
    agg = {"pooling_type": pooling_type, "n_reps": len(runs), "runs": runs}
    for key in METRIC_KEYS:
        agg[key] = df[key].mean()
        agg[f"{key}_std"] = df[key].std() if len(runs) > 1 else None
    agg["n_epochs"] = runs[0]["n_epochs"]
    return agg


def parse_log_durations(log_path: Path):
    """Parse 'HYBRID_RERUN_TIMER <type> start|end <epoch>' marker lines into
    {pooling_type: seconds}. Returns {} if the log or markers aren't found."""
    if log_path is None or not log_path.exists():
        return {}
    text = log_path.read_text()
    durations = {}
    for pooling_type in ("diffpool", "dmon"):
        start_m = re.search(rf"HYBRID_RERUN_TIMER {pooling_type} start (\d+)", text)
        end_m = re.search(rf"HYBRID_RERUN_TIMER {pooling_type} end (\d+)", text)
        if start_m and end_m:
            durations[pooling_type] = int(end_m.group(1)) - int(start_m.group(1))
    return durations


def format_duration(seconds):
    if seconds is None:
        return "n/a"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def format_pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def format_pct_std(mean, std):
    if mean is None:
        return "n/a"
    if std is None:
        return format_pct(mean)
    return f"{mean * 100:.2f}% +/- {std * 100:.2f}pp"


def format_loss_std(mean, std):
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def build_report(diffpool, dmon, durations, n_holdouts):
    lines = []
    lines.append("# Hybrid DiffPool vs Hybrid DMoN -- comparison\n")
    lines.append(
        "Matched config on both runs: `--n-hybrid 2 --n-hybrid-start 2 --tune "
        "--num-samples 16 --n-cycles 7 --batch-size 96 --gpu-per-trial 1 "
        f"--cpu-per-trial 8 --n-holdouts {n_holdouts} --use-train-set-weights` -- "
        "only `--pooling-type` (and its pooling-type-specific lambda search "
        "space) differs between the two runs. Machine: 24GB RTX 3090 Ti "
        "(batch size 96 empirically validated, see scripts/run_hybrid_rerun.sh "
        f"header). Metrics below are mean +/- std across {n_holdouts} holdout "
        "rep(s) (each rep tunes its own hyperparameters independently).\n"
    )

    lines.append("## Results\n")
    lines.append("| Metric | Hybrid DiffPool | Hybrid DMoN |")
    lines.append("|---|---|---|")
    for label, key, fmt in [
        ("Test accuracy", "test_accuracy", format_pct_std),
        ("Test balanced accuracy", "test_balanced_accuracy", format_pct_std),
        ("Test loss", "test_loss", format_loss_std),
        ("Train accuracy (final epoch)", "train_accuracy", format_pct_std),
        ("Train loss (final epoch)", "train_loss", format_loss_std),
    ]:
        dp_val = fmt(diffpool[key], diffpool[f"{key}_std"]) if diffpool else "n/a (run missing)"
        dm_val = fmt(dmon[key], dmon[f"{key}_std"]) if dmon else "n/a (run missing)"
        lines.append(f"| {label} | {dp_val} | {dm_val} |")
    lines.append(
        f"| Epochs (final retrain) | {diffpool['n_epochs'] if diffpool else 'n/a'} "
        f"| {dmon['n_epochs'] if dmon else 'n/a'} |"
    )
    lines.append(
        f"| Reps completed | {diffpool['n_reps'] if diffpool else 0}/{n_holdouts} "
        f"| {dmon['n_reps'] if dmon else 0}/{n_holdouts} |"
    )
    lines.append(
        f"| Wall-clock time (total, all reps) | {format_duration(durations.get('diffpool'))} "
        f"| {format_duration(durations.get('dmon'))} |"
    )
    lines.append("")

    if diffpool and dmon and diffpool["test_accuracy"] is not None and dmon["test_accuracy"] is not None:
        delta = dmon["test_accuracy"] - diffpool["test_accuracy"]
        better = "DMoN" if delta > 0 else ("DiffPool" if delta < 0 else "Tie")
        lines.append(
            f"**{better}** wins on mean test accuracy under this matched config "
            f"(DMoN {format_pct_std(dmon['test_accuracy'], dmon['test_accuracy_std'])} vs. DiffPool "
            f"{format_pct_std(diffpool['test_accuracy'], diffpool['test_accuracy_std'])}, "
            f"delta = {delta * 100:+.2f}pp). Treat this as noise if it's smaller than "
            f"either side's std across reps.\n"
        )

    lines.append("## Tuned hyperparameters (per rep, final retrain config)\n")
    for agg in (diffpool, dmon):
        if agg is None:
            continue
        for run in agg["runs"]:
            lines.append(f"**{run['pooling_type']} rep{run['rep']}** (`{run['run_dir']}`):")
            lines.append("```json")
            lines.append(json.dumps(run["config"], indent=2))
            lines.append("```\n")

    lines.append("## Reference: previously documented (not reproduced from disk)\n")
    lines.append(
        f"- Hybrid DiffPool, prior narrative record: **{format_pct(DOCUMENTED_BASELINE['test_accuracy'])}** "
        f"-- {DOCUMENTED_BASELINE['note']}"
    )
    if diffpool and diffpool["test_accuracy"] is not None:
        delta = diffpool["test_accuracy"] - DOCUMENTED_BASELINE["test_accuracy"]
        lines.append(
            f"- This run's Hybrid DiffPool (mean {format_pct(diffpool['test_accuracy'])}) vs. that record: "
            f"delta = {delta * 100:+.2f}pp"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-output", type=str, default="outputs/hybrid_rerun")
    parser.add_argument("--n-hybrid", type=int, default=2)
    parser.add_argument("--n-holdouts", type=int, default=1,
                        help="Number of holdout reps to load and aggregate (mean +/- std)")
    parser.add_argument("--log", type=str, default=None, help="Path to a log containing HYBRID_RERUN_TIMER markers")
    parser.add_argument("--out", type=str, default=None, help="Output .md path (default: <path-output>/comparison.md)")
    args = parser.parse_args()

    path_output = Path(args.path_output)
    diffpool = load_and_aggregate(path_output, "diffpool", args.n_hybrid, args.n_holdouts)
    dmon = load_and_aggregate(path_output, "dmon", args.n_hybrid, args.n_holdouts)
    durations = parse_log_durations(Path(args.log)) if args.log else {}

    report = build_report(diffpool, dmon, durations, args.n_holdouts)

    out_path = Path(args.out) if args.out else path_output / "comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote comparison report to {out_path}")


if __name__ == "__main__":
    main()
