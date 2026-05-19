import argparse
from pathlib import Path
import pandas as pd


def generate_multitask_dataset_information_table(
    path_metadata: str,
    path_output: str = None,
    include_ratio: bool = False
):
    path_metadata_obj = Path(path_metadata)
    if path_output is None:
        path_output = Path(f"info_{path_metadata_obj.stem}.csv")

    metadata = pd.read_csv(path_metadata, index_col=0)
    counts = metadata.value_counts().reset_index()
    counts = counts.pivot(index=['cohort'], columns=['sample_type'])
    counts.columns = counts.columns.droplevel(0)
    counts.index = counts.index.str.upper()
    counts.loc['Total'] = counts.sum(numeric_only=True, axis=0)
    counts.loc[:,'Total'] = counts.sum(numeric_only=True, axis=1)
    counts.loc[:,'Ratio'] = counts['Solid Tissue Normal'] / counts['Primary Tumor']
    counts.to_csv(path_output)


def generate_dataset_information_table(
    path_metadata: str,
    path_output: str = None
):
    path_metadata_obj = Path(path_metadata)
    if path_output is None:
        path_output = Path(f"info_{path_metadata_obj.stem}.csv")

    metadata = pd.read_csv(path_metadata, index_col=0)
    counts = metadata.value_counts()
    counts.loc['Total'] = counts.sum()
    counts.to_csv(path_output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-metadata",
        required=True,
        type=str,
        help='Path to metadata file of the tcga dataset'
    )
    parser.add_argument(
        "--path-output",
        type=str,
        help='Path to output directory where to save artifacts'
    )
    parser.add_argument(
        '--multitask',
        action='store_true',
        help='Use it if the dataset metadata has two columns instead of one'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    path_metadata = args.path_metadata
    path_output = args.path_output

    if args.multitask:
        generate_multitask_dataset_information_table(path_metadata=path_metadata, path_output=path_output, include_ratio=True)
        return
    
    generate_dataset_information_table(path_metadata=path_metadata, path_output=path_output)



if __name__ == "__main__":
    main()
