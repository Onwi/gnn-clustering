from pathlib import Path
from typing import List, Tuple, Union, Literal
import numpy as np
import pandas as pd

from pooling_genomic.datasets import get_genomic_classification_dataset


class UnseenTaskIndicesBuilder:
    def __init__(
        self,
        path_dataset: Union[str, Path],
        output_dir: Union[str, Path],
        unseen_cohort_sets: Union[List[str], List[Tuple], None] = None,
        n_holdouts: int = 5
    ):
        self.path_dataset = Path(path_dataset)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.n_holdouts = n_holdouts

        self.random_states = []
        rng = np.random.default_rng(seed=123)
        for rep in range(self.n_holdouts):
            random_state = int(rng.integers(500))
            self.random_states.append(random_state)

        self.unseen_cohort_sets = unseen_cohort_sets

    def generate_and_save_indices(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

        unseen_cohort_sets = self.get_unseen_cohort_sets()
        for cohort_set in unseen_cohort_sets:
            assert len(cohort_set) == 1, "Using more than one cohort is not implemented."
            cohort = list(cohort_set)[0]

            for run in range(self.n_holdouts):                
                train_idx, val_idx, test_idx = self.get_unseen_cohort_dataset_run_indices(
                    cohort=cohort, run=run
                )
                fname_train = self.get_indices_filename(ds='train', cohort=cohort, run=run)
                fname_val = self.get_indices_filename(ds='val', cohort=cohort, run=run)
                fname_test = self.get_indices_filename(ds='test', cohort=cohort, run=run)
                
                path_train = output_dir / fname_train
                path_val = output_dir / fname_val
                path_test = output_dir / fname_test
                
                train_idx.to_csv(path_train)
                val_idx.to_csv(path_val)
                test_idx.to_csv(path_test)

    def get_indices_filename(self, ds: Literal['train', 'validation', 'test'], cohort: str, run: int):
        fname = f"indices_{ds}_{cohort}_{run}.csv"
        return fname

    def get_unseen_cohort_dataset_run_indices(self, cohort: str, run: int, return_metadata: bool = False):
        """Get train, validation and test indices for dataset using the specified cohort as the unseen
        cohort. Other than that, the samples present in the splitted datasets are like that of the cohort-specific
        datasets.
        """
        train_set, val_set, test_set, dataset = self.get_datasets(run=run)
        metadata = dataset.metadata.copy().reset_index()

        # Remove specified cohort from the training and validation sets
        train_metadata = metadata.iloc[train_set.idx_0 : train_set.idx_0 + train_set.ds_len, :]
        train_cohort_metadata = train_metadata[train_metadata['cohort'] != cohort]

        val_metadata = metadata.iloc[val_set.idx_0 : val_set.idx_0 + val_set.ds_len, :]
        val_cohort_metadata = val_metadata[val_metadata['cohort'] != cohort]

        # Keep only the unseen cohort in the test set
        test_metadata = metadata.iloc[test_set.idx_0 : test_set.idx_0 + train_set.ds_len, :]
        test_cohort_metadata = test_metadata[test_metadata['cohort'] == cohort]

        if return_metadata:
            return train_cohort_metadata, val_cohort_metadata, test_cohort_metadata
        
        return train_cohort_metadata['index'], val_cohort_metadata['index'], test_cohort_metadata['index']

    def get_unseen_cohort_sets(self):
        """Get list containing sets of cohorts that will be excluded from the training in an execution."""
        if self.unseen_cohort_sets is None:
            cohorts = self.get_cohorts()
            cohorts = [set([c,]) for c in cohorts]
            return cohorts
        
        return self.unseen_cohort_sets

    def get_cohorts(self):
        train_set, val_set, test_set, dataset = self.get_datasets(run=0)
        cohorts = dataset.metadata['cohort'].unique().tolist()
        return cohorts
    
    def get_datasets(self, run: int, metadata_column: str = 'sample_type'):
        random_state = self.random_states[run]
        train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
            path_dataset=self.path_dataset,
            return_original_set=True,
            random_state=random_state,
            metadata_column=metadata_column,
        )
        return train_set, val_set, test_set, dataset

def test_indices(
    path_dataset,
    output_dir
):
    indices_builder = UnseenTaskIndicesBuilder(
        path_dataset=path_dataset,
        output_dir=output_dir
    )
    n_total_cohorts = len(indices_builder.get_cohorts())

    train_set, val_set, test_set, dataset = indices_builder.get_datasets(run=0)
    metadata = dataset.metadata.copy()

    for fpath in output_dir.glob('*.csv'):
        ds, cohort, run = fpath.stem.split("_")[1:]
        indices = pd.read_csv(fpath, index_col=0)['index']
        cohorts_in_ds = metadata.loc[indices, 'cohort'].unique()
        
        if (ds == 'train') or (ds == 'val'):
            assert cohort not in cohorts_in_ds, f"Cohort {cohort} should not be in the indices of {fpath}"
            assert len(cohorts_in_ds) == n_total_cohorts - 1, f"File {fpath} should have {n_total_cohorts} cohorts, but it has {len(cohorts_in_ds)}"
        elif ds == 'test':
            assert len(cohorts_in_ds) == 1
            assert cohorts_in_ds[0] == cohort
        else:
            raise ValueError("Invalid ds found")


def main():
    path_dataset = '/home/thomas/Documents/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification'
    output_dir = Path('data/unseen_cohort_data/indices')

    indices_builder = UnseenTaskIndicesBuilder(
        path_dataset=path_dataset,
        output_dir=output_dir
    )
    unseen_sets = indices_builder.get_unseen_cohort_sets()
    
    cohort = 'esca'
    df_train, df_val, df_test = indices_builder.get_unseen_cohort_dataset_run_indices(cohort=cohort, run=0, return_metadata=True)
    assert cohort not in df_train['cohort']
    assert len(df_train['cohort'].unique()) == 15

    assert cohort not in df_val['cohort']
    assert len(df_val['cohort'].unique()) == 15

    assert df_test['cohort'].value_counts().index[0] == cohort
    assert len(df_test['cohort'].unique()) == 1

    indices_builder.generate_and_save_indices()

    test_indices(
        path_dataset=path_dataset,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
