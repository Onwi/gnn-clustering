"""Training curves over epochs (requires multi-epoch data)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(
    dfs_meta: list,
    output_dir: Path,
    figsize=(14, 5),
):
    """Plot train/test loss and accuracy over epochs for each model.

    Parameters
    ----------
    dfs_meta : list of (label, pd.DataFrame)
        Each DataFrame has columns: epoch, train_loss, train_accuracy,
        test_loss, test_accuracy, test_balanced_accuracy.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for label, df in dfs_meta:
        if len(df) < 2:
            continue
        epochs = df["epoch"].values
        axes[0].plot(epochs, df["train_loss"], marker=".", label=f"{label} (train)")
        axes[0].plot(epochs, df["test_loss"], marker=".", label=f"{label} (test)", linestyle="--")
        axes[1].plot(epochs, df["train_accuracy"], marker=".", label=f"{label} (train)")
        axes[1].plot(epochs, df["test_accuracy"], marker=".", label=f"{label} (test)", linestyle="--")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Test Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training / Test Accuracy")
    axes[1].legend(fontsize=7)
    axes[1].grid(True)

    fig.tight_layout()
    path = output_dir / "training_curves.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
