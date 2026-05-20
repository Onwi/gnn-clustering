"""Error analysis: disagreement between model predictions, hardest samples."""

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_disagreement_heatmap(
    dfs_labels,
    model_a_label: str,
    model_b_label: str,
    output_dir: Path,
):
    """Plot a 2x2 grid showing where two models agree/disagree, split by correct/incorrect.

    Cells:
      (A correct, B correct) / (A correct, B wrong) / (A wrong, B correct) / (A wrong, B wrong)
    """
    df_a = dfs_labels[model_a_label]
    df_b = dfs_labels[model_b_label]

    common_idx = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]

    a_correct = df_a["predictions"] == df_a["labels"]
    b_correct = df_b["predictions"] == df_b["labels"]

    categories = ["Both correct", "A only correct", "B only correct", "Both wrong"]
    counts = [
        ((a_correct) & (b_correct)).sum(),
        ((a_correct) & (~b_correct)).sum(),
        ((~a_correct) & (b_correct)).sum(),
        ((~a_correct) & (~b_correct)).sum(),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#f39c12", "#3498db", "#e74c3c"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="gray")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(count), ha="center", fontsize=12)
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Agreement: {model_a_label} vs {model_b_label}")
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / "disagreement_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Confusion matrix of disagreement (true label vs predicted label of A and B)
    _plot_disagreement_cm(
        df_a, df_b, a_correct, b_correct, output_dir,
        model_a_label, model_b_label,
    )

    return pd.DataFrame({
        "category": categories,
        "count": counts,
        "pct": [c / len(common_idx) * 100 for c in counts],
    })


def _plot_disagreement_cm(df_a, df_b, a_correct, b_correct, output_dir,
                           model_a_label, model_b_label):
    """Show which true classes have the most disagreement."""
    both_wrong = (~a_correct) & (~b_correct)
    if both_wrong.sum() == 0:
        return

    both_wrong_df = pd.DataFrame({
        "true": df_a.loc[both_wrong, "labels"],
        f"pred_{model_a_label}": df_a.loc[both_wrong, "predictions"],
        f"pred_{model_b_label}": df_b.loc[both_wrong, "predictions"],
    })

    top_classes = both_wrong_df["true"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    top_classes.plot(kind="barh", ax=ax, color="#e74c3c")
    ax.set_xlabel("Number of samples both models got wrong")
    ax.set_ylabel("True class")
    ax.set_title("Top classes where both models fail")
    ax.grid(axis="x", alpha=0.3)
    path = output_dir / "both_wrong_by_class.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_prediction_agreement_matrix(
    dfs_labels,
    models_order,
    output_dir: Path,
):
    """Plot pairwise agreement matrix between multiple model predictions."""
    models = list(models_order)
    n = len(models)
    agreement = np.zeros((n, n))

    preds = {}
    for m in models:
        preds[m] = dfs_labels[m]["predictions"].values

    # Align on common index
    common_idx = None
    for df in dfs_labels.values():
        if common_idx is None:
            common_idx = set(df.index)
        else:
            common_idx = common_idx & set(df.index)

    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            pi = dfs_labels[mi].loc[list(common_idx), "predictions"]
            pj = dfs_labels[mj].loc[list(common_idx), "predictions"]
            agreement[i, j] = (pi == pj).mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(agreement, annot=True, fmt=".2f", xticklabels=models,
                yticklabels=models, cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Prediction Agreement Between Models")
    path = output_dir / "prediction_agreement.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
