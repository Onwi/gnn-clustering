"""Comprehensive comparison analysis between Fixed HEM and Learned DiffPool.

Usage
-----
python -m scripts.analysis_v2.main --path-output outputs
"""

import argparse
from pathlib import Path
import pandas as pd

from .parsers import parse_all_results, load_predictions
from .confusion_matrix import (
    plot_confusion_side_by_side,
    plot_per_class_metrics,
    generate_classification_reports,
)
from .training_curves import plot_training_curves
from .error_analysis import (
    plot_disagreement_heatmap,
    plot_prediction_agreement_matrix,
)
from .hyperparam_sensitivity import (
    plot_hyperparameter_sensitivity,
    plot_accuracy_vs_complexity,
)


def _get_best_variants(df: pd.DataFrame) -> list:
    """Return (label, dirpath) for the best-performing variant of each model type."""
    best = []
    for model_type in ["Fixed HEM", "Learned DiffPool"]:
        sub = df[df["model"] == model_type]
        if sub.empty:
            continue
        # Pick the row with highest test_accuracy
        best_row = sub.loc[sub["test_accuracy"].idxmax()]
        label = (
            f"HEM L{best_row['n_levels']} W{best_row['weighted_pooling']} "
            f"C{best_row['use_convs']}"
            if model_type == "Fixed HEM"
            else f"DiffPool H{best_row['n_hybrid']}"
        )
        best.append((label, best_row["dirname"]))
    return best


def _load_labeled_predictions(base_path: Path, variants):
    """Return list of (label, DataFrame) tuples from variant descriptors."""
    result = []
    for label, dirname in variants:
        df = load_predictions(base_path / dirname)
        result.append((label, df))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of Fixed HEM vs Learned DiffPool"
    )
    parser.add_argument(
        "--path-output", type=str, default="./outputs",
        help="Directory containing experiment outputs"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plots interactively"
    )
    args = parser.parse_args()

    path_output = Path(args.path_output)
    output_dir = path_output / "analysis_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse all results ---
    print("Parsing results...")
    df = parse_all_results(path_output)
    df.attrs["path_output"] = str(path_output)

    if df.empty:
        print(f"No results found in {path_output}.")
        return

    print(f"  Found {len(df)} result records: {df.groupby('model').size().to_dict()}")

    # --- Summary table ---
    print("\n--- Summary Table ---")
    summary = df.groupby("model").agg(
        mean_acc=("test_accuracy", "mean"),
        std_acc=("test_accuracy", "std"),
        mean_bal_acc=("test_balanced_accuracy", "mean"),
        count=("test_accuracy", "count"),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(output_dir / "summary_table.csv")

    # --- Confusion matrices (best variant of each) ---
    print("\n--- Confusion Matrices & Per-Class Metrics ---")
    variants = _get_best_variants(df)
    dfs_labels = _load_labeled_predictions(path_output, variants)

    plot_confusion_side_by_side(dfs_labels, output_dir)
    plot_per_class_metrics(dfs_labels, output_dir)
    generate_classification_reports(dfs_labels, output_dir)

    # --- Training curves (all results) ---
    print("\n--- Training Curves ---")
    # If metrics.csv has multiple epochs, plot them
    dfs_curves = []
    for _, row in df.iterrows():
        metrics_path = (
            path_output / row["dirname"] / "final_model_results" / "metrics.csv"
        )
        metrics_df = pd.read_csv(metrics_path, index_col=0)
        if len(metrics_df) >= 2:
            key = (
                f"HEM L{row['n_levels']} W{row['weighted_pooling']} C{row['use_convs']}"
                if row["model"] == "Fixed HEM"
                else f"DP H{row['n_hybrid']}"
            )
            dfs_curves.append((key, metrics_df))

    if dfs_curves:
        plot_training_curves(dfs_curves, output_dir)
    else:
        print("  No multi-epoch training data found (metrics.csv has single rows).")

    # --- Error analysis (best variants) ---
    print("\n--- Error Analysis ---")
    if len(dfs_labels) >= 2:
        labels_a, labels_b = dfs_labels[0], dfs_labels[1]
        plot_disagreement_heatmap(
            dict(dfs_labels),
            labels_a[0], labels_b[0],
            output_dir,
        )

    # Agreement matrix across all variants
    models_order = [v[0] for v in variants]
    if len(variants) >= 2:
        plot_prediction_agreement_matrix(dict(dfs_labels), models_order, output_dir)

    # --- Hyperparameter sensitivity ---
    print("\n--- Hyperparameter Sensitivity ---")
    plot_hyperparameter_sensitivity(df, output_dir)
    plot_accuracy_vs_complexity(df, output_dir)

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
