import json
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_confusion_matrix(
    df_classif: pd.DataFrame,
    true_label_column: str = 'true_label',
    predicted_label_column: str = 'predicted'
):
    labels = df_classif[true_label_column].unique().tolist()
    cm = confusion_matrix(y_true=df_classif[true_label_column], y_pred=df_classif[predicted_label_column], labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots()
    sns.heatmap(df_cm, annot=True, ax=ax, fmt="g", cbar=False, cmap=sns.cm.rocket_r)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Label")

    return fig, ax


def savefig(fig, output_dir, filename_stem, extensions=("pdf", "jpg")):
    for ext in extensions:
        fig.savefig(output_dir / f"{filename_stem}.{ext}")


def write_json(obj, file_path: Union[str, Path]):
    file_path = Path(file_path)
    with open(str(file_path), "w") as f:
        return json.dump(obj, f)


def build_data_loaders(*args, batch_size, num_workers, device='cpu', drop_last=False, shuffle=True):
    loaders = []
    pin_memory = True if 'cuda' in device else False
    for dataset in args:
        dataset_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
        )
        loaders.append(dataset_loader)

    return tuple(loaders)
