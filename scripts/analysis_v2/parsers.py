"""Parse output directories into DataFrames for analysis."""

import re
from pathlib import Path
import pandas as pd
import numpy as np


def parse_all_results(path_output: Path) -> pd.DataFrame:
    """Scan output directory and parse metadata + metrics from every run."""
    records = []
    for dirpath in path_output.iterdir():
        metrics_path = dirpath / "final_model_results" / "metrics.csv"
        configs_path = dirpath / "final_model_results" / "model_configs.json"
        if not metrics_path.exists():
            continue

        metrics = pd.read_csv(metrics_path, index_col=0)
        last = metrics.iloc[-1, :].to_dict()

        configs = {}
        if configs_path.exists():
            configs = pd.read_json(configs_path, typ="series").to_dict()

        dirname = dirpath.name

        # Fixed HEM: nlevels{N}_rep{R}_wpool{W}_convs{C}
        m = re.match(r"nlevels(\d+)_rep(\d+)_wpool(True|False)_convs(True|False)", dirname)
        if m:
            record = dict(
                model="Fixed HEM",
                n_levels=int(m.group(1)),
                rep=int(m.group(2)),
                weighted_pooling=m.group(3) == "True",
                use_convs=m.group(4) == "True",
                n_hybrid=None,
                dirname=dirname,
            )
            record.update(last)
            record.update(configs)
            records.append(record)
            continue

        # Learned pooling: {diffpool,dmon}_{hybrid,full}{N}_rep{R}
        m = re.match(r"(diffpool|dmon)_(hybrid|full)(\d+)_rep(\d+)", dirname)
        if m:
            pooling_type, mode_tag, n_hybrid, rep = m.groups()
            model_label = {
                # "Learned DiffPool" kept as-is (rather than "Hybrid DiffPool")
                # for backward compatibility with scripts/analysis_v2/main.py
                # and hyperparam_sensitivity.py, which filter on this exact
                # string for existing diffpool_hybrid{N}_rep{R} output dirs.
                ("diffpool", "hybrid"): "Learned DiffPool",
                ("diffpool", "full"): "Full DiffPool",
                ("dmon", "hybrid"): "Hybrid DMoN",
                ("dmon", "full"): "Full DMoN",
            }[(pooling_type, mode_tag)]
            record = dict(
                model=model_label,
                pooling_type=pooling_type,
                full_mode=mode_tag == "full",
                n_levels=None,
                rep=int(rep),
                weighted_pooling=None,
                use_convs=None,
                n_hybrid=int(n_hybrid),
                dirname=dirname,
            )
            record.update(last)
            record.update(configs)
            records.append(record)
            continue

    return pd.DataFrame(records)


def load_predictions(dirpath: Path) -> pd.DataFrame:
    """Load predictions.csv with true labels and predicted labels."""
    path = dirpath / "final_model_results" / "predictions.csv"
    return pd.read_csv(path, index_col=0)


def load_outputs(dirpath: Path) -> pd.DataFrame:
    """Load outputs.csv (raw logits per class + true labels)."""
    path = dirpath / "final_model_results" / "outputs.csv"
    return pd.read_csv(path, index_col=0)


def load_all_predictions(df_meta: pd.DataFrame) -> dict:
    """Return dict mapping (model, config_key) -> predictions DataFrame."""
    base = Path(df_meta.attrs.get("path_output", "outputs"))
    result = {}
    for _, row in df_meta.iterrows():
        key = _config_key(row)
        result[key] = load_predictions(base / row["dirname"])
    return result


def load_all_outputs(df_meta: pd.DataFrame) -> dict:
    base = Path(df_meta.attrs.get("path_output", "outputs"))
    result = {}
    for _, row in df_meta.iterrows():
        key = _config_key(row)
        result[key] = load_outputs(base / row["dirname"])
    return result


def _config_key(row) -> str:
    if row["model"] == "Fixed HEM":
        return f"HEM_L{row['n_levels']}_W{row['weighted_pooling']}_C{row['use_convs']}_R{row['rep']}"
    prefix = "DMoN" if row.get("pooling_type") == "dmon" else "DP"
    return f"{prefix}_H{row['n_hybrid']}_R{row['rep']}"
