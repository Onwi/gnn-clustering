"""Analyze a Full DMoN run's final_model_results and write a markdown report.

Reads <path-output>/dmon_full{n_hybrid}_rep{rep}/final_model_results/
(metrics.csv, model_configs.json) -- the directory scripts/experiments/
diffpool_experiment.py writes for a --full-mode --pooling-type dmon run --
and optionally a run log containing 'FULL_DMON_TIMER start|end <unix_epoch>'
marker lines (written by scripts/run_full_dmon_shrunk.sh) to report
wall-clock time.

Unlike scripts/analysis/compare_hybrid_results.py, this is a single-run
report (no matched Full DiffPool run exists to compare against), so it just
presents this run's own numbers alongside the previously recorded reference
points from analysis-approaches.MD / this run's comparison.md documents.

Usage:
    python scripts/analysis/analyze_full_dmon_results.py \
        --path-output outputs/full_dmon_shrunk --log full_dmon_shrunk.log \
        --out outputs/full_dmon_shrunk/results.md
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Reference points from prior runs, for context only -- not reproduced from
# disk by this script.
REFERENCES = [
    ("Full DiffPool, untuned defaults (analysis-approaches.MD)", 0.2767, None),
    ("Full DiffPool, rescued/tuned (plan.md 5.3.3: encoder + low aux weights + grad clip)", 0.7168, None),
    ("Hybrid DiffPool, this repo's corrected re-run (outputs/hybrid_rerun2, mean of 3 reps)", 0.6582, 0.6494),
    ("Hybrid DMoN, this repo's corrected re-run (outputs/hybrid_rerun2, mean of 3 reps)", 0.6263, 0.6387),
]

METRIC_KEYS = ("test_accuracy", "test_balanced_accuracy", "test_loss", "train_accuracy", "train_loss")


def load_run(path_output: Path, n_hybrid: int, rep: int):
    run_dir = path_output / f"dmon_full{n_hybrid}_rep{rep}" / "final_model_results"
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "model_configs.json"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path, index_col=0)
    final = metrics.iloc[-1].to_dict()
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {
        "run_dir": run_dir,
        "n_epochs": len(metrics),
        "config": config,
        **{key: final.get(key) for key in METRIC_KEYS},
    }


def parse_log_duration(log_path: Path):
    if log_path is None or not log_path.exists():
        return None
    text = log_path.read_text()
    start_m = re.search(r"FULL_DMON_TIMER start (\d+)", text)
    end_m = re.search(r"FULL_DMON_TIMER end (\d+)", text)
    if start_m and end_m:
        return int(end_m.group(1)) - int(start_m.group(1))
    return None


def format_duration(seconds):
    if seconds is None:
        return "n/a"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def format_pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def build_report(run, duration, n_hybrid, args):
    lines = []
    lines.append("# Full DMoN -- shrunk-scope tuning run results\n")
    lines.append(
        f"Config: `--full-mode --pooling-type dmon --n-hybrid {n_hybrid} --n-hybrid-start {n_hybrid} "
        f"--sparsify-density {args.sparsify_density} --tune --num-samples {args.num_samples} "
        f"--n-cycles {args.n_cycles} --batch-size {args.batch_size} --n-holdouts {args.n_holdouts} "
        "--use-train-set-weights`.\n"
    )
    lines.append(
        "This is a deliberately **shrunk-scope, single-rep, exploratory** run (see conversation/"
        "changes-from-claude.md fix #3) -- not directly comparable in statistical rigor to the "
        "3-rep, 16-sample Hybrid DiffPool/DMoN comparison (`outputs/hybrid_rerun2/comparison.md`). "
        f"Per-epoch cost at this scale (~43 min at batch={args.batch_size}, measured on this "
        "hardware) made a matched-scope run infeasible (~11 weeks); this run answers a narrower "
        "question -- does removing the 440:1 compression (progressive cluster schedule) and the "
        "dense-adjacency corruption (pooled-adjacency sparsification) let Full DMoN learn at all, "
        "not what its fully-tuned, variance-characterized ceiling is.\n"
    )

    lines.append("## Results\n")
    if run is None:
        lines.append("**Run did not produce a final_model_results directory -- check the log for errors.**\n")
    else:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Test accuracy | {format_pct(run['test_accuracy'])} |")
        lines.append(f"| Test balanced accuracy | {format_pct(run['test_balanced_accuracy'])} |")
        lines.append(f"| Test loss | {run['test_loss']:.4f} |" if run['test_loss'] is not None else "| Test loss | n/a |")
        lines.append(f"| Train accuracy (final epoch) | {format_pct(run['train_accuracy'])} |")
        lines.append(f"| Train loss (final epoch) | {run['train_loss']:.4f} |" if run['train_loss'] is not None else "| Train loss (final epoch) | n/a |")
        lines.append(f"| Epochs (final retrain) | {run['n_epochs']} |")
        lines.append(f"| Wall-clock time (total) | {format_duration(duration)} |")
        lines.append("")

        lines.append("## Tuned hyperparameters (final retrain config)\n")
        lines.append(f"`{run['run_dir']}`:")
        lines.append("```json")
        lines.append(json.dumps(run["config"], indent=2))
        lines.append("```\n")

    lines.append("## Reference points (not reproduced from disk, for context only)\n")
    for label, acc, bal_acc in REFERENCES:
        bal_str = f", balanced {format_pct(bal_acc)}" if bal_acc is not None else ""
        lines.append(f"- {label}: **{format_pct(acc)}**{bal_str}")
    if run is not None and run["test_accuracy"] is not None:
        lines.append("")
        for label, acc, _ in REFERENCES:
            delta = run["test_accuracy"] - acc
            lines.append(f"- This run vs. \"{label}\": delta = {delta * 100:+.2f}pp")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-output", type=str, default="outputs/full_dmon_shrunk")
    parser.add_argument("--n-hybrid", type=int, default=3)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--sparsify-density", type=float, default=0.04)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--n-cycles", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-holdouts", type=int, default=1)
    parser.add_argument("--log", type=str, default=None, help="Path to a log containing FULL_DMON_TIMER markers")
    parser.add_argument("--out", type=str, default=None, help="Output .md path (default: <path-output>/results.md)")
    args = parser.parse_args()

    path_output = Path(args.path_output)
    run = load_run(path_output, args.n_hybrid, args.rep)
    duration = parse_log_duration(Path(args.log)) if args.log else None

    report = build_report(run, duration, args.n_hybrid, args)

    out_path = Path(args.out) if args.out else path_output / "results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote results report to {out_path}")


if __name__ == "__main__":
    main()
