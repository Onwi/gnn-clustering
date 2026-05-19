# re-run models saved in .pt files
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pooling_genomic.datasets import get_genomic_classification_dataset
from pooling_genomic.engines import evaluate_clf

from pooling_genomic.models import build_coarsening_model
from pooling_genomic.networks import load_graph_levels


def run_model_in_test_set(
    n_levels=0,
    max_levels=7,
    path_model='/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_coarsening_sample_type_prediction/nlevels0_rep0_wpoolFalse_convsFalse/final_model_results/final_model.pt',
    path_dataset='/home/thomas/Documents/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification',
    path_levels='/home/thomas/Documents/PoolingGenomicGNNs/data/networks/levels',
    output_dims=2,
    weighted_pooling=False,
    use_convs=False,
    device='cpu',
    random_state=7,
    batch_size=64,
    num_workers=2,
    pin_memory=False,
    output_dir='artifacts',
    predictions_fname='predictions.csv'
):
    output_dir = Path(output_dir)
    graphs = load_graph_levels(
        path_levels=path_levels, n_levels=max_levels, device=device
    )
    model = build_coarsening_model(
        n_levels=n_levels,
        graphs=graphs[:n_levels] if n_levels > 0 else graphs[:1],
        output_dims=output_dims,
        weighted_pooling=weighted_pooling,
        use_convs=use_convs,
        device=device
    )
    model.load_state_dict(torch.load(str(path_model)))

    train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
        path_dataset=path_dataset, return_original_set=True, random_state=random_state, metadata_column='sample_type'
    )
    classes = dataset.label_encoder.classes_

    train_set_cohorts, val_set_cohorts, test_set_cohorts, dataset_cohorts = get_genomic_classification_dataset(
        path_dataset=path_dataset, return_original_set=True, random_state=random_state, metadata_column='cohort'
    )
    sample_cohorts = dataset_cohorts.label_encoder.inverse_transform(dataset_cohorts.y)
    test_sample_cohorts = sample_cohorts[
        test_set_cohorts.idx_0 : test_set_cohorts.idx_0 + test_set_cohorts.ds_len
    ]
        
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )
    print('Evaluating...')
    test_metrics, (predictions, labels) = evaluate_clf(
        model=model,
        validation_loader=test_loader,
        device=device,
        return_outputs=True,
    )
    print('Done.')
    predictions = np.argmax(predictions, axis=1)
    
    data = {"predictions": predictions, "labels": labels}
    df_test_predictions = pd.DataFrame.from_dict(data)
    df_test_predictions["predictions"] = df_test_predictions["predictions"].map(
        lambda x: classes[x]
    )
    df_test_predictions["labels"] = df_test_predictions["labels"].map(
        lambda x: classes[x]
    )
    df_test_predictions['cohort'] = test_sample_cohorts

    df_test_predictions.to_csv(output_dir / predictions_fname)


def main():
    n_holdouts = 5
    path_model_template = '/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_coarsening_sample_type_prediction/nlevels0_rep{}_wpoolFalse_convsFalse/final_model_results/final_model.pt'

    rng = np.random.default_rng(seed=123)
    for rep in range(n_holdouts):
        print('Execution {}/{}'.format(rep+1, n_holdouts))
        random_state = int(rng.integers(500))

        print(path_model_template)
        path_model = path_model_template.format(rep)

        run_model_in_test_set(
            path_model=path_model,
            random_state=random_state,
            output_dir='artifacts',
            predictions_fname='predictions_nlevels0_rep{}_wpoolFalse_convsFalse.csv'.format(rep)
        )


if __name__ == "__main__":
    main()