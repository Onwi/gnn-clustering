import math
from pathlib import Path
from typing import List

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def get_genomic_classification_dataset(
    path_dataset,
    **kwargs
):
    if 'tcga_cohort_classification' in str(path_dataset):
        return get_tcga_cohort_classification_datasets(
            path_dataset=path_dataset,
            **kwargs
        )
    if 'tcga_brca_subtypes_classification' in str(path_dataset):
        kwargs['metadata_column'] = 'Subtype_mRNA'
        return get_tcga_classification_datasets(
            path_dataset=path_dataset,
            # metadata_column='Subtype_mRNA',
            **kwargs
        )
    if ('tcga' in str(path_dataset)) and ('tumor_prediction' in str(path_dataset)):
        kwargs['metadata_column'] = 'sample_type'
        return get_tcga_classification_datasets(
            path_dataset=path_dataset,
            # metadata_column='sample_type',
            **kwargs
        )
    if 'tcga_cohorts_and_tumor_classification' in str(path_dataset):
        return get_tcga_classification_datasets(
            path_dataset=path_dataset,
            **kwargs
        )


class PCRunIndicesLoader:
    def __init__(self, path_indices: str, cohort: str, fname_template: str = 'indices_{}_{}_{}.csv', run=None):
        self.path_indices = Path(path_indices)
        self.fname_template = fname_template
        self.cohort = cohort
        self.run = run
    
    def get_indices(self):
        train_indices = pd.read_csv(self.path_indices / self.fname_template.format('train', self.cohort, self.run), index_col=0).index
        val_indices = pd.read_csv(self.path_indices / self.fname_template.format('val', self.cohort, self.run), index_col=0).index
        test_indices = pd.read_csv(self.path_indices / self.fname_template.format('test', self.cohort, self.run), index_col=0).index
        return train_indices, val_indices, test_indices


def get_tcga_cohort_classification_datasets(
    path_dataset=Path.home() / "siamesegnn_omics/data/tcga_cohort_classification",
    return_original_set: bool = False,
    train_proportion: float = 0.6,
    validation_proportion: float = 0.2,
    random_state: int = 123
):
    path_dataset = Path(path_dataset)
    dataset = TCGACohorts(path_dataset=path_dataset, random_state=random_state)

    train_size, val_size = math.floor(train_proportion * len(dataset)), math.floor(
        validation_proportion * len(dataset)
    )
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset,
        lengths=[train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_state),
    )

    if return_original_set:
        return train_set, val_set, test_set, dataset

    return train_set, val_set, test_set


class WrapperDataset(Dataset):
    def __init__(self, dataset, idx_0=None, ds_len=None, direct_indices=None):
        if direct_indices is None:
            assert (idx_0 + ds_len) <= len(dataset)

        self.dataset = dataset
        self.idx_0 = idx_0
        self.ds_len = ds_len
        self.direct_indices = direct_indices

    def __len__(self):
        if self.direct_indices is not None:
            return len(self.direct_indices)
        return self.ds_len
    
    def __getitem__(self, index):
        if self.direct_indices is not None:
            return self.dataset[self.direct_indices[index]]
        
        return self.dataset[index + self.idx_0]


class TCGACohorts(Dataset):
    def __init__(
        self,
        path_dataset: Path,
        cohorts: List=None,
        transform=None,
        scaler=None,
        random_state: int = 123,
        use_complete_dataset_labels: bool = False
    ):
        print(f"Using dataset from: {path_dataset}")
        self.path_dataset = Path(path_dataset)
        self.transform = transform
        self.scaler = scaler

        self.path_samples = self.path_dataset / 'samples'
        metadata = pd.read_csv(path_dataset / 'sample_metadata.csv', index_col=0)
        metadata = metadata.sample(frac=1, random_state=random_state)

        if cohorts is not None:
            self.samples = metadata.loc[metadata["cohort"].isin(cohorts), :].index
        else:
            self.samples = metadata.index
        
        sample_cohorts = metadata.loc[self.samples, 'cohort'].to_numpy()

        self.label_encoder = LabelEncoder()
        if use_complete_dataset_labels:
            self.label_encoder.fit(
                metadata.loc[:, 'cohort'].to_numpy()
            )
            self.y = self.label_encoder.transform(sample_cohorts)
            
        else:    
            self.y = self.label_encoder.fit_transform(sample_cohorts)
        self.n_classes = len(np.unique(self.y))

    def get_path_sample(self, index):
        sample_index = self.samples[index]

        path_sample = tcga_index_to_relative_path(
            sample_idx=sample_index, path_prefix=self.path_samples
        )
        return path_sample

    def get_genes(self):
        path_sample = self.get_path_sample(0)
        genes = pd.read_csv(path_sample, index_col=0).iloc[:, 0].index
        return genes
    
    def get_class_weights(self):
        """Compute class weights using the 'balanced' heuristic, as in scikit-learn."""
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y), y=self.y)
        class_weights = torch.from_numpy(class_weights).float()

        return class_weights

    def get_n_classes(self):
        return self.n_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path_sample = self.get_path_sample(index)

        df_sample = pd.to_numeric(pd.read_csv(path_sample, index_col=0).iloc[:, 0])
        sample = df_sample.to_numpy().transpose()
        if self.scaler is not None:
            sample = self.scaler.transform(sample.reshape(1, -1)).reshape(-1)

        sample = torch.from_numpy(sample).to(torch.float)
        if self.transform is not None:
            sample = self.transform(sample)

        y = self.y[index]
        return sample, y
    

class TCGADataset(Dataset):
    def __init__(
        self,
        path_dataset: Path,
        metadata_column: str, 
        transform=None,
        scaler=None,
        random_state: int = 123,
        use_complete_dataset_labels: bool = False
    ):
        print(f"Using dataset from: {path_dataset} Metadata column: {metadata_column}")
        self.path_dataset = Path(path_dataset)
        self.transform = transform
        self.scaler = scaler
        self.metadata_column = metadata_column

        self.path_samples = self.path_dataset / 'samples'
        metadata = pd.read_csv(path_dataset / 'sample_metadata.csv', index_col=0)
        metadata = metadata.sample(frac=1, random_state=random_state)
        
        self.metadata = metadata
        self.samples = metadata.index
        sample_labels = metadata.loc[self.samples, metadata_column].to_numpy()

        self.label_encoder = LabelEncoder()
        if use_complete_dataset_labels:
            self.label_encoder.fit(
                metadata.loc[:, metadata_column].to_numpy()
            )
            self.y = self.label_encoder.transform(sample_labels)
        else:    
            self.y = self.label_encoder.fit_transform(sample_labels)
        self.n_classes = len(np.unique(self.y))

    def get_path_sample(self, index):
        sample_index = self.samples[index]

        path_sample = tcga_index_to_relative_path(
            sample_idx=sample_index, path_prefix=self.path_samples
        )
        return path_sample

    def get_genes(self):
        path_sample = self.get_path_sample(0)
        genes = pd.read_csv(path_sample, index_col=0).iloc[:, 0].index
        return genes
    
    def get_class_weights(self, cohort: str = None):
        """Compute class weights using the 'balanced' heuristic, as in scikit-learn."""
        if cohort is not None:
            y = self.metadata.loc[self.metadata['cohort'] == cohort, self.metadata_column].to_numpy()
        else:
            y = self.y
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        class_weights = torch.from_numpy(class_weights).float()

        return class_weights

    def get_n_classes(self):
        return self.n_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path_sample = self.get_path_sample(index)

        df_sample = pd.to_numeric(pd.read_csv(path_sample, index_col=0).iloc[:, 0])
        sample = df_sample.to_numpy().transpose()
        if self.scaler is not None:
            sample = self.scaler.transform(sample.reshape(1, -1)).reshape(-1)

        sample = torch.from_numpy(sample).to(torch.float)
        if self.transform is not None:
            sample = self.transform(sample)

        y = self.y[index]
        return sample, y


def get_tcga_classification_datasets(
    path_dataset,
    metadata_column,
    return_original_set: bool = False,
    train_proportion: float = 0.6,
    validation_proportion: float = 0.2, 
    random_state = 123,
    train_indices = None,
    val_indices = None,
    test_indices = None,
    indices_loader: PCRunIndicesLoader = None
):
    # use wrapper dataset so that I have access to the metadata attribute and can analyze the results
    
    path_dataset = Path(path_dataset)
    dataset = TCGADataset(
        path_dataset=path_dataset, 
        metadata_column=metadata_column, 
        random_state=random_state
    )
    if indices_loader is not None:
        train_indices, val_indices, test_indices = indices_loader.get_indices()
        train_set = WrapperDataset(dataset, direct_indices=train_indices)
        val_set = WrapperDataset(dataset, direct_indices=val_indices)
        test_set = WrapperDataset(dataset, direct_indices=test_indices)
    elif (train_indices is not None) and (val_indices is not None) and (test_indices is not None):
        train_set = WrapperDataset(dataset, direct_indices=train_indices)
        val_set = WrapperDataset(dataset, direct_indices=val_indices)
        test_set = WrapperDataset(dataset, direct_indices=test_indices)
    else:
        train_set_len = int(train_proportion * len(dataset))
        validation_set_len = int(validation_proportion * len(dataset))
        test_set_len = len(dataset) - train_set_len - validation_set_len

        train_set = WrapperDataset(dataset, idx_0=0, ds_len=train_set_len)
        val_set = WrapperDataset(dataset, idx_0=train_set_len, ds_len=validation_set_len)
        test_set = WrapperDataset(dataset, idx_0=train_set_len + validation_set_len, ds_len=test_set_len)

    if return_original_set:
        return train_set, val_set, test_set, dataset

    return train_set, val_set, test_set

def tcga_index_to_relative_path(sample_idx: str, path_prefix: str = None):
    """Convert a TCGA identifier to a path in the filesystem.

    Parameters
    ----------
    sample_idx : str
        TCGA sample identifier
    path_prefix : str, optional
        Path prefix to add to the sample path, by default None

    Returns
    -------
    Path
        Path to sample file
    """
    name_parts = sample_idx.split("-")

    index_path = Path("/".join(name_parts[:-1])) / f"{sample_idx}.csv"
    if path_prefix is None:
        return index_path

    return Path(path_prefix) / index_path


def get_tcga_cohort_and_tumor_classification_datasets(
    path_dataset,
    return_original_set: bool = False,
    train_proportion: float = 0.6,
    validation_proportion: float = 0.2, 
    random_state = 123
):
    # use wrapper dataset so that I have access to the metadata attribute and can analyze the results
    
    path_dataset = Path(path_dataset)
    dataset = TCGACohortsAndTumor(path_dataset=path_dataset, random_state=random_state)

    train_set_len = int(train_proportion * len(dataset))
    validation_set_len = int(validation_proportion * len(dataset))
    test_set_len = len(dataset) - train_set_len - validation_set_len

    train_set = WrapperDataset(dataset, idx_0=0, ds_len=train_set_len)
    val_set = WrapperDataset(dataset, idx_0=train_set_len, ds_len=validation_set_len)
    test_set = WrapperDataset(dataset, idx_0=train_set_len + validation_set_len, ds_len=test_set_len)

    if return_original_set:
        return train_set, val_set, test_set, dataset

    return train_set, val_set, test_set


class TCGACohortsAndTumor(Dataset):
    def __init__(
        self,
        path_dataset: Path,
        cohorts: List=None,
        transform=None,
        scaler=None,
        random_state: int = 123,
        use_complete_dataset_labels: bool = False
    ):
        print(f"Using dataset from: {path_dataset}")
        self.path_dataset = Path(path_dataset)
        self.transform = transform
        self.scaler = scaler

        self.path_samples = self.path_dataset / 'samples'
        metadata = pd.read_csv(self.path_dataset / 'sample_metadata.csv', index_col=0)
        metadata = metadata.sample(frac=1, random_state=random_state)

        if cohorts is not None:
            self.samples = metadata.loc[metadata["cohort"].isin(cohorts), :].index
        else:
            self.samples = metadata.index
        
        sample_cohorts = metadata.loc[self.samples, 'cohort'].to_numpy()
        sample_types = metadata.loc[self.samples, 'sample_type'].to_numpy()

        self.cohorts_encoder = LabelEncoder()
        if use_complete_dataset_labels:
            self.cohorts_encoder.fit(
                metadata.loc[:, 'cohort'].to_numpy()
            )
            self.y_cohorts = self.cohorts_encoder.transform(sample_cohorts)
        else:    
            # self.y = self.label_encoder.fit_transform(sample_cohorts)
            self.y_cohorts = self.cohorts_encoder.fit_transform(sample_cohorts)
        
        self.n_cohorts = len(np.unique(self.y_cohorts))

        self.types_encoder = LabelEncoder()
        self.y_types = self.types_encoder.fit_transform(sample_types)
        print(type(self.y_types))

    def get_n_cohorts(self):
        return self.n_cohorts

    def get_path_sample(self, index):
        sample_index = self.samples[index]

        path_sample = tcga_index_to_relative_path(
            sample_idx=sample_index, path_prefix=self.path_samples
        )
        return path_sample

    def get_genes(self):
        path_sample = self.get_path_sample(0)
        genes = pd.read_csv(path_sample, index_col=0).iloc[:, 0].index
        return genes
    
    def get_class_weights(self):
        """
        Compute class weights using the 'balanced' heuristic, as in scikit-learn.
        
        Return cohort_weights and type_weights
        """
        cohort_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_cohorts), y=self.y_cohorts)
        cohort_weights = torch.from_numpy(cohort_weights).float()

        type_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_types), y=self.y_types)
        type_weights = torch.from_numpy(type_weights).float()

        return cohort_weights, type_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path_sample = self.get_path_sample(index)

        df_sample = pd.to_numeric(pd.read_csv(path_sample, index_col=0).iloc[:, 0])
        sample = df_sample.to_numpy().transpose()
        if self.scaler is not None:
            sample = self.scaler.transform(sample.reshape(1, -1)).reshape(-1)

        sample = torch.from_numpy(sample).to(torch.float)
        if self.transform is not None:
            sample = self.transform(sample)

        y_cohort = self.y_cohorts[index].astype(int)
        y_type = self.y_types[index].astype(np.float32)
        return sample, (y_cohort, y_type)