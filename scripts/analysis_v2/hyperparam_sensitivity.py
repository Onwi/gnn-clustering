"""Hyperparameter sensitivity analysis."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_hyperparameter_sensitivity(
    df_meta: pd.DataFrame,
    output_dir: Path,
    metric: str = "test_accuracy",
    figsize=(14, 10),
):
    """Scatter plots of hyperparameter vs accuracy for each model type.

    For each numeric hyperparameter found in the metadata, plots accuracy vs
    that hyperparameter, colored by model type.
    """
    numeric_cols = df_meta.select_dtypes(include=[np.number]).columns.tolist()
    skip_cols = {"n_levels", "n_hybrid", "rep"}
    hp_cols = [c for c in numeric_cols if c not in skip_cols and c != metric]

    if not hp_cols:
        print("  No numeric hyperparameters found in metadata.")
        return

    n = len(hp_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], nrows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    colors = {"Fixed HEM": "#1f77b4", "Learned DiffPool": "#ff7f0e"}

    for ax, hp in zip(axes, hp_cols):
        for model_type, grp in df_meta.groupby("model"):
            if hp not in grp.columns or grp[hp].isnull().all():
                continue
            ax.scatter(grp[hp], grp[metric], label=model_type,
                       alpha=0.7, c=colors.get(model_type, "gray"))
        ax.set_xlabel(hp)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {hp}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    path = output_dir / "hyperparameter_sensitivity.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_accuracy_vs_complexity(
    df_meta: pd.DataFrame,
    output_dir: Path,
    metric: str = "test_accuracy",
):
    """Plot accuracy vs number of levels/hybrid for both models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    hemi = df_meta[df_meta["model"] == "Fixed HEM"]
    diff = df_meta[df_meta["model"] == "Learned DiffPool"]

    if not hemi.empty:
        means = hemi.groupby("n_levels")[metric].agg(["mean", "sem"]).reset_index()
        yerr = means["sem"].fillna(0)
        ax.errorbar(means["n_levels"], means["mean"], yerr=yerr,
                    marker="o", label="Fixed HEM", capsize=3, linewidth=2)

    if not diff.empty:
        means = diff.groupby("n_hybrid")[metric].agg(["mean", "sem"]).reset_index()
        yerr = means["sem"].fillna(0)
        ax.errorbar(means["n_hybrid"], means["mean"], yerr=yerr,
                    marker="s", color="tab:orange", label="Learned DiffPool",
                    capsize=3, linewidth=2)

    ax.set_xlabel("Levels (HEM) / Hybrid Levels (DiffPool)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Model Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = output_dir / "accuracy_vs_complexity.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
