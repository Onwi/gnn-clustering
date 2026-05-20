"""Side-by-side confusion matrices and per-class precision/recall/F1."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .parsers import load_predictions


def _per_class_metrics(y_true, y_pred, classes):
    """Compute per-class precision, recall, F1."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm = cm.astype(float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0)
    support = cm.sum(axis=1)
    return pd.DataFrame({
        "class": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def plot_confusion_side_by_side(
    dfs_labels,
    output_dir: Path,
    title: str = "Confusion Matrices",
    figsize=(16, 6),
):
    """Plot side-by-side confusion matrices for multiple models.

    Parameters
    ----------
    dfs_labels : list of (label, pd.DataFrame)
        Each DataFrame must have 'predictions' and 'labels' columns.
    """
    n = len(dfs_labels)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    all_classes = sorted(set().union(*(
        set(df["labels"].unique()) | set(df["predictions"].unique())
        for _, df in dfs_labels
    )))

    for ax, (label, df) in zip(axes, dfs_labels):
        cm = confusion_matrix(df["labels"], df["predictions"], labels=all_classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-15)
        sns.heatmap(cm_norm, annot=len(all_classes) <= 20, fmt=".2f",
                    xticklabels=all_classes, yticklabels=all_classes,
                    cmap="Blues", ax=ax, cbar=False, vmin=0, vmax=1)
        ax.set_title(label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path = output_dir / "confusion_side_by_side.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_class_metrics(
    dfs_labels,
    output_dir: Path,
    title: str = "Per-class F1 Score Comparison",
    figsize=(12, 6),
):
    """Grouped bar chart of per-class F1 for multiple models."""
    records = []
    for model_label, df in dfs_labels:
        classes = sorted(df["labels"].unique())
        pm = _per_class_metrics(df["labels"], df["predictions"], classes)
        pm["model"] = model_label
        records.append(pm)

    df_all = pd.concat(records, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        pivot = df_all.pivot_table(index="class", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, width=0.75)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path = output_dir / "per_class_metrics.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def generate_classification_reports(dfs_labels, output_dir: Path):
    """Save sklearn classification_report text per model."""
    for model_label, df in dfs_labels:
        classes = sorted(df["labels"].unique())
        report = classification_report(
            df["labels"], df["predictions"],
            labels=classes, target_names=classes, digits=4,
            zero_division=0,
        )
        path = output_dir / f"classification_report_{model_label.replace(' ', '_')}.txt"
        path.write_text(report)
        print(f"  Saved {path}")
