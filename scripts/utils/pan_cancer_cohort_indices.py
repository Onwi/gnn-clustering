from pathlib import Path
import numpy as np
import pandas as pd
from pooling_genomic.datasets import get_genomic_classification_dataset, get_tcga_classification_datasets, PCRunIndicesLoader


class PanCancerDatasetIndicesGetter:
    def __init__(self, path_dataset: str) -> None:
        self.path_dataset = path_dataset

        self.n_holdouts = 5
        self.random_states = []
        rng = np.random.default_rng(seed=123)
        for rep in range(self.n_holdouts):
            random_state = int(rng.integers(500))
            self.random_states.append(random_state)

    def get_datasets(self, run: int, metadata_column: str = 'sample_type'):
        random_state = self.random_states[run]
        train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
            path_dataset=self.path_dataset,
            return_original_set=True,
            random_state=random_state,
            metadata_column=metadata_column,
        )
        return train_set, val_set, test_set, dataset
    
    def get_cohorts(self):
        train_set, val_set, test_set, dataset = self.get_datasets(run=0)
        cohorts = dataset.metadata['cohort'].unique().tolist()
        return cohorts

    def get_cohort_sets(self, cohort: str, run: int, return_metadata: bool = False):
        train_set, val_set, test_set, dataset = self.get_datasets(run=run)
        metadata = dataset.metadata.copy().reset_index()

        train_metadata = metadata.iloc[train_set.idx_0 : train_set.idx_0 + train_set.ds_len, :]
        train_cohort_metadata = train_metadata[train_metadata['cohort'] == cohort]

        val_metadata = metadata.iloc[val_set.idx_0 : val_set.idx_0 + val_set.ds_len, :]
        print(f"val metadata len {len(val_metadata)} val.ds_len {val_set.ds_len}")
        val_cohort_metadata = val_metadata[val_metadata['cohort'] == cohort]

        test_metadata = metadata.iloc[test_set.idx_0 : test_set.idx_0 + train_set.ds_len, :]
        test_cohort_metadata = test_metadata[test_metadata['cohort'] == cohort]

        if return_metadata:
            return train_cohort_metadata, val_cohort_metadata, test_cohort_metadata
        
        return train_cohort_metadata['index'], val_cohort_metadata['index'], test_cohort_metadata['index']

    def generate_and_save_indices(self, output_dir):
        output_dir = Path(output_dir)
        cohorts = self.get_cohorts()
        
        for run in range(self.n_holdouts):
            for cohort in cohorts:
                train_indices, val_indices, test_indices = self.get_cohort_sets(cohort=cohort, run=run)
                print(f'run {run} cohort {cohort} train {len(train_indices)} val {len(val_indices)}  test {len(test_indices)}')
                fname_template = 'indices_{}_{}_{}.csv'
                train_indices.to_csv(output_dir / fname_template.format('train', cohort, run))
                val_indices.to_csv(output_dir / fname_template.format('val', cohort, run))
                test_indices.to_csv(output_dir / fname_template.format('test', cohort, run))


def get_cohort_ds_samples(ds, cohort, metadata: pd.DataFrame):
    cohort_samples = []
    for i in range(len(ds)):
        i_org = ds.idx_0 + i
        i_cohort = metadata['cohort'][i_org]
        if i_cohort == cohort:
            cohort_samples.append(ds[i][0].numpy())
    cohort_samples = np.stack(cohort_samples, axis=0)
    return cohort_samples


def test_indices_match():
    path_dataset = '/home/thomas/Documents/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification'
    cohorts = PanCancerDatasetIndicesGetter(path_dataset).get_cohorts()
    output_dir = Path('artifacts')
    cohort = 'brca'
    run = 0
    indices_dir = Path('data/cs_vs_pc_data/indices')

    for cohort in cohorts:
        print("Processing cohort {}".format(cohort))

        rng = np.random.default_rng(seed=123)
        for run in range(0, 5):
            print("- Evaluating run {}...".format(run))
            random_state = int(rng.integers(500))
            
            indices_loader = PCRunIndicesLoader(
                path_indices=indices_dir,
                cohort=cohort,
                run=run
            )

            print("- Processing pan-cancer")
            train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
                path_dataset=path_dataset,
                return_original_set=True,
                random_state=random_state,
                metadata_column="sample_type",
            )
            metadata = dataset.metadata.copy().reset_index()
            print(f"-- Run {run} Pan-cancer {cohort} examples: {len(metadata[metadata['cohort'] == cohort])}")
            print(f"-- train set {len(train_set)} val set {len(val_set)} test_set {len(test_set)}")

            pc_cohort_test_samples = get_cohort_ds_samples(ds=test_set, cohort=cohort, metadata=metadata)
            pc_cohort_val_samples = get_cohort_ds_samples(ds=val_set, cohort=cohort, metadata=metadata)
            pc_cohort_train_samples = get_cohort_ds_samples(ds=train_set, cohort=cohort, metadata=metadata)
            print(len(pc_cohort_train_samples))
            print(len(pc_cohort_val_samples))
            print(len(pc_cohort_test_samples))

            print('- Processing cohort-specific')
            cs_train_set, cs_val_set, cs_test_set, cs_dataset = get_tcga_classification_datasets(
                path_dataset=path_dataset,
                metadata_column='sample_type',
                indices_loader=indices_loader,
                return_original_set=True,
                random_state=random_state
            )
            print(f'-- Cohort {cohort} train set {len(cs_train_set)} val set {len(cs_val_set)} test set {len(cs_test_set)}')

            cs_test_samples = []
            cs_val_samples = []
            cs_train_samples = []
            for i in range(len(cs_test_set)):
                cs_test_samples.append(cs_test_set[i][0].numpy())

            for i in range(len(cs_val_set)):
                cs_val_samples.append(cs_val_set[i][0].numpy())

            for i in range(len(cs_train_set)):
                cs_train_samples.append(cs_train_set[i][0].numpy())

            cs_test_samples = np.stack(cs_test_samples, axis=0)
            cs_train_samples = np.stack(cs_train_samples, axis=0)
            cs_val_samples = np.stack(cs_val_samples, axis=0)
            
            print(f"Train pc: {len(pc_cohort_train_samples)} Train cs: {len(cs_train_samples)}")
            print(f"Val pc: {len(pc_cohort_val_samples)} Val cs: {len(cs_val_samples)}")
            print(f"Test pc: {len(pc_cohort_test_samples)} Test cs: {len(cs_test_samples)}")
            
            cmp_test = cs_test_samples == pc_cohort_test_samples
            cmp_val = cs_val_samples == pc_cohort_val_samples
            cmp_train = cs_train_samples == pc_cohort_train_samples
            assert cmp_test.all() and cmp_val.all() and cmp_train.all()
            print("All equal")


def generate_indices():
    path_dataset = '/home/thomas/Documents/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification'
    output_dir = Path('data/cs_vs_pc_data/indices')
    output_dir.mkdir(exist_ok=True, parents=True)
    pc_indices = PanCancerDatasetIndicesGetter(path_dataset=path_dataset)
    pc_indices.generate_and_save_indices(output_dir=output_dir)


if __name__ == "__main__":
    generate_indices()
    test_indices_match()
