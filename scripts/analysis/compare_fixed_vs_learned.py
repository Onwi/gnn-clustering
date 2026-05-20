import argparse
from pathlib import Path
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def parse_results(path_output: Path):
    """Scan output directory and parse results from both model types."""
    records = []

    for dirpath in path_output.iterdir():
        dirname = dirpath.name
        metrics_path = dirpath / "final_model_results" / "metrics.csv"
        if not metrics_path.exists():
            continue

        metrics = pd.read_csv(metrics_path, index_col=0)
        last = metrics.iloc[-1, :].to_dict()

        # --- Fixed HEM model: nlevels{N}_rep{R}_wpool{W}_convs{C} ---
        m = re.match(r"nlevels(\d+)_rep(\d+)_wpool(True|False)_convs(True|False)", dirname)
        if m:
            record = {
                "model": "Fixed HEM",
                "n_levels": int(m.group(1)),
                "rep": int(m.group(2)),
                "weighted_pooling": m.group(3) == "True",
                "use_convs": m.group(4) == "True",
                "n_hybrid": None,
            }
            record.update(last)
            records.append(record)
            continue

        # --- DiffPool model: diffpool_hybrid{N}_rep{R} ---
        m = re.match(r"diffpool_hybrid(\d+)_rep(\d+)", dirname)
        if m:
            record = {
                "model": "Learned DiffPool",
                "n_levels": None,
                "rep": int(m.group(2)),
                "weighted_pooling": None,
                "use_convs": None,
                "n_hybrid": int(m.group(1)),
            }
            record.update(last)
            records.append(record)
            continue

    return pd.DataFrame(records)


def plot_comparison(df, metric, output_dir, show=False):
    """Plot comparison between Fixed HEM and Learned DiffPool."""
    df_fixed = df[df["model"] == "Fixed HEM"].copy()
    df_diffpool = df[df["model"] == "Learned DiffPool"].copy()

    if df_fixed.empty and df_diffpool.empty:
        print("No results found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # --- Left: Fixed HEM (varies n_levels, colored by config) ---
    if not df_fixed.empty:
        ax = axes[0]
        for label, grp in df_fixed.groupby(["weighted_pooling", "use_convs"]):
            wp, uc = label
            label_str = f"wpool={wp}, convs={uc}"
            means = grp.groupby("n_levels")[metric].mean()
            sems = grp.groupby("n_levels")[metric].sem()
            ax.errorbar(means.index, means.values, yerr=sems.values, label=label_str,
                        marker='o', capsize=3)
        ax.set_xlabel("Coarsening Levels")
        ax.set_ylabel(metric)
        ax.set_title("Fixed HEM Pooling")
        ax.legend(fontsize=8)
        ax.grid(True)

    # --- Right: Learned DiffPool (varies n_hybrid) ---
    if not df_diffpool.empty:
        ax = axes[1]
        means = df_diffpool.groupby("n_hybrid")[metric].mean()
        sems = df_diffpool.groupby("n_hybrid")[metric].sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    marker='s', color='tab:orange', capsize=3, linewidth=2)
        ax.set_xlabel("Number of Hybrid Levels")
        ax.set_title("Learned DiffPool")
        ax.grid(True)

    fig.suptitle(f"Comparison: {metric}", fontsize=14)
    fig.tight_layout()

    # --- Combined plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    if not df_fixed.empty:
        # Aggregate fixed HEM: take best config per n_levels
        best_fixed = (
            df_fixed.groupby("n_levels")[metric]
            .agg(["mean", "sem"])
            .reset_index()
        )
        ax2.errorbar(best_fixed["n_levels"], best_fixed["mean"], yerr=best_fixed["sem"],
                     marker='o', label="Fixed HEM (best config)", capsize=3, linewidth=2)

    if not df_diffpool.empty:
        dp = (
            df_diffpool.groupby("n_hybrid")[metric]
            .agg(["mean", "sem"])
            .reset_index()
        )
        ax2.errorbar(dp["n_hybrid"], dp["mean"], yerr=dp["sem"],
                     marker='s', color='tab:orange',
                     label="Learned DiffPool", capsize=3, linewidth=2)

    if not df_fixed.empty:
        # Also add nn baseline (n_levels=0)
        nn_records = df_fixed[df_fixed["n_levels"] == 0]
        if not nn_records.empty:
            nn_mean = nn_records[metric].mean()
            nn_sem = nn_records[metric].sem()
            ax2.axhline(y=nn_mean, color='gray', linestyle='--', alpha=0.7,
                        label=f"MLP baseline (n_levels=0)")
            ax2.fill_between(
                [ax2.get_xlim()[0], ax2.get_xlim()[1]],
                nn_mean - nn_sem, nn_mean + nn_sem,
                color='gray', alpha=0.1
            )

    ax2.set_xlabel("Coarsening Levels / Hybrid Levels")
    ax2.set_ylabel(metric)
    ax2.set_title(f"Fixed HEM vs Learned DiffPool — {metric}")
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f"comparison_split_{metric}.pdf")
    fig2.savefig(output_dir / f"comparison_combined_{metric}.pdf")

    if show:
        plt.show()

    plt.close(fig)
    plt.close(fig2)

    print(f"Saved plots to {output_dir}/")


def print_summary_table(df):
    """Print a markdown summary table."""
    print("\n## Summary Table\n")
    print("| Model | Config | Reps | Mean Accuracy | Std Dev |")
    print("|---|---|---|---|---|")

    for model, grp in df.groupby("model"):
        if model == "Fixed HEM":
            for (wp, uc), sub in grp.groupby(["weighted_pooling", "use_convs"]):
                acc = sub["test_accuracy"]
                print(f"| Fixed HEM | wpool={wp}, convs={uc} | {len(sub)} | {acc.mean():.4f} | {acc.std():.4f} |")
        else:
            for nh, sub in grp.groupby("n_hybrid"):
                acc = sub["test_accuracy"]
                print(f"| Learned DiffPool | n_hybrid={nh} | {len(sub)} | {acc.mean():.4f} | {acc.std():.4f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-output", type=str, default="./outputs",
                        help="Directory containing experiment outputs")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    args = parser.parse_args()

    path_output = Path(args.path_output)
    df = parse_results(path_output)

    if df.empty:
        print(f"No results found in {path_output}.")
        print("Make sure experiment outputs exist with patterns:")
        print("  Fixed HEM:    nlevels{N}_rep{R}_wpool{W}_convs{C}/")
        print("  DiffPool:     diffpool_hybrid{N}_rep{R}/")
        return

    print(f"Found {len(df)} result records:")
    print(df.groupby("model").size().to_string())

    print_summary_table(df)

    for metric in ["test_accuracy", "test_balanced_accuracy", "test_loss"]:
        if metric in df.columns:
            plot_comparison(df, metric, path_output / "comparison", show=args.show)
        else:
            print(f"Metric '{metric}' not found in results.")


if __name__ == "__main__":
    main()
