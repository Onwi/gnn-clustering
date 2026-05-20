import argparse
from functools import partial
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler

from pooling_genomic.datasets import get_genomic_classification_dataset, PCRunIndicesLoader
from pooling_genomic.models import build_diffpool_model
from pooling_genomic.networks import load_coarse_edges_for_diffpool, get_pyg_data
from pooling_genomic.settings import PoolingGenomicSettings
from pooling_genomic.engines import train_epoch_clf, evaluate_clf
from pooling_genomic.utils import plot_confusion_matrix, savefig, write_json


def build_data_loaders(*args, batch_size, num_workers, device='cpu'):
    loaders = []
    pin_memory = True if 'cuda' in device else False
    for dataset in args:
        drop_last = True if len(dataset) % batch_size == 1 else False
        dataset_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
        )
        loaders.append(dataset_loader)
    return tuple(loaders)


def build_hp_config(args):
    if args.tune:
        hp_config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-4, 1e-1),
            "lambda_link_pred": tune.loguniform(1e-5, 1e-1),
            "lambda_entropy": tune.loguniform(1e-5, 1e-1),
            "eta_min": 0.00001,
            "T_0": 1,
            "T_mult": 2,
        }
    else:
        hp_config = {
            "lr": 0.05,
            "weight_decay": 0.01,
            "lambda_link_pred": 0.001,
            "lambda_entropy": 0.001,
            "eta_min": 0.00001,
            "T_0": 1,
            "T_mult": 2,
        }
    return hp_config


def train_and_validate_model(
    hp_config,
    args,
    n_hybrid=2,
    random_state=123,
    rep=0,
    indices_loader: PCRunIndicesLoader | None = None
):
    path_dataset = Path(args.path_dataset)
    path_levels = Path(args.path_levels)
    path_network = Path(args.path_network)
    device = args.device
    batch_size = args.batch_size
    num_workers = int(args.cpu_per_trial)
    max_epochs = args.max_epochs
    n_cycles = max(args.n_cycles - 1, 1)
    using_ray_tune = args.tune

    lr = hp_config["lr"]
    weight_decay = hp_config["weight_decay"]
    eta_min = hp_config["eta_min"]
    T_0 = hp_config["T_0"]
    T_mult = hp_config["T_mult"]
    lambda_link_pred = hp_config["lambda_link_pred"]
    lambda_entropy = hp_config["lambda_entropy"]

    # ---- data ----
    dataset_kwargs = dict(return_original_set=True, random_state=random_state)
    if args.metadata_column is not None:
        dataset_kwargs['metadata_column'] = args.metadata_column
    if indices_loader is not None:
        dataset_kwargs['indices_loader'] = indices_loader
    train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
        path_dataset=path_dataset, **dataset_kwargs
    )
    train_loader, val_loader = build_data_loaders(
        train_set, val_set, batch_size=batch_size, num_workers=num_workers, device=device
    )
    output_dims = dataset.get_n_classes()

    # ---- graph ----
    genes = dataset.get_genes()
    base_graph = get_pyg_data(genes=genes, path_to_csv=path_network)
    base_graph = base_graph.to(device)

    max_levels = args.max_n_levels
    coarse_edges, parents_list = load_coarse_edges_for_diffpool(
        path_levels=path_levels, n_levels=max_levels, device=device
    )

    if n_cycles is not None:
        max_epochs = int(T_0 * (1 - T_mult**n_cycles) / (1 - T_mult))

    model = build_diffpool_model(
        base_graph=base_graph,
        coarse_edges=coarse_edges,
        parents_list=parents_list,
        output_dims=output_dims,
        n_hybrid=n_hybrid,
        max_filters=args.max_filters,
        max_clusters=args.max_clusters,
        dense_threshold=args.dense_threshold,
    )
    model = model.to(device=device)

    class_weights = None
    try:
        if args.use_train_set_weights:
            class_weights = dataset.get_class_weights()
        elif args.cohort_indices:
            class_weights = dataset.get_class_weights(cohort=args.cohort_indices)
    except Exception:
        pass
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )

    for epoch in range(max_epochs):
        print(f"Epoch [{epoch + 1} / {max_epochs}]")

        model, train_metrics = train_epoch_clf(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            epoch=epoch,
            loss_fn=loss_fn,
            lambda_link_pred=lambda_link_pred,
            lambda_entropy=lambda_entropy,
        )
        validation_metrics = evaluate_clf(
            model=model, validation_loader=val_loader, device=device, loss_fn=loss_fn
        )

        val_loss = validation_metrics["loss"]
        accuracy = validation_metrics["accuracy"]
        train_loss = train_metrics["loss"]
        train_accuracy = train_metrics["accuracy"]

        if using_ray_tune:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = str(Path(checkpoint_dir) / "checkpoint")
            tune.report(
                loss=val_loss,
                accuracy=accuracy,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                checkpoint=checkpoint_dir,
            )
        else:
            print("-- Validation accuracy: {:.2f}".format(accuracy))
            print("-- Validation loss: {:.4f}".format(val_loss))
            print("-- Train loss: {:.4f}".format(train_loss))
            print("-- Train accuracy: {:.4f}".format(train_accuracy))

    # ---- final test + save (non-tuning path) ----
    if not using_ray_tune:
        test_loader = build_data_loaders(
            test_set, batch_size=batch_size, num_workers=num_workers, device=device
        )[0]
        test_metrics, (test_outputs, test_labels) = evaluate_clf(
            model=model, validation_loader=test_loader, device=device,
            loss_fn=loss_fn, return_outputs=True,
        )
        print("-- Test metrics: {}".format(test_metrics))

        path_experiment = (
            Path(args.path_output)
            / f"diffpool_hybrid{n_hybrid}_rep{rep}"
        )
        analyze_final_model_results(
            pd.DataFrame({
                "epoch": [0], "train_loss": [train_loss],
                "train_accuracy": [train_accuracy],
                "test_loss": [test_metrics["loss"]],
                "test_accuracy": [test_metrics["accuracy"]],
                "test_balanced_accuracy": [test_metrics["balanced_accuracy"]],
            }),
            test_outputs, test_labels, hp_config, path_experiment,
            dataset.label_encoder.classes_, model=model,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("path_dataset", type=str)
    parser.add_argument("path_levels", type=str,
                        help="Path to pre-computed graph levels (edge_index / edge_weight per level)")
    parser.add_argument("--path-network", type=str, default=None,
                        help="Path to STRING-DB CSV edge list. If omitted, derived from settings.")
    parser.add_argument("--metadata-column", type=str, default=None)

    parser.add_argument("--max-n-levels", type=int, default=8)
    parser.add_argument("--n-hybrid", type=int, default=2,
                        help="Number of early levels that use hybrid mode (fixed coarse edges)")
    parser.add_argument("--max-filters", type=int, default=32,
                        help="Maximum feature dimension (grows progressivly: 1,2,4,...,max_filters)")
    parser.add_argument("--max-clusters", type=int, default=32,
                        help="Maximum clusters per DiffPoolLayer (bound on k)")
    parser.add_argument("--dense-threshold", type=int, default=500,
                        help="Node count below which full mode (dense adjacency pooling) is used")

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--cpu-per-trial", type=float, default=1)
    parser.add_argument("--gpu-per-trial", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--n-cycles", type=int, default=5)
    parser.add_argument("--path-output", type=str, default="./outputs")
    parser.add_argument("--n-holdouts", type=int, default=5)
    parser.add_argument("--path-indices", type=str, default=None)
    parser.add_argument("--cohort-indices", type=str, default=None)
    parser.add_argument("--use-train-set-weights", action="store_true")

    args = parser.parse_args()

    settings = PoolingGenomicSettings()
    if args.path_network is None:
        args.path_network = str(settings.path_data / 'networks' / 'stringdb_top100pc.csv')

    return args


def set_trainable_resources(trainable, device, cpu_per_trial, gpu_per_trial):
    if 'cuda' in device:
        trainable = tune.with_resources(
            trainable, {"cpu": cpu_per_trial, "gpu": gpu_per_trial}
        )
    else:
        trainable = tune.with_resources(trainable, {"cpu": cpu_per_trial})
    return trainable


def build_tuner(trainable, scheduler, configs, num_samples, path_ray):
    tuner = tune.Tuner(
        trainable=trainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            local_dir=str(path_ray),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="accuracy", num_to_keep=1
            ),
        ),
        param_space=configs,
    )
    return tuner


def train_and_test_model(results, args, path_experiment, n_hybrid, random_state,
                         indices_loader: PCRunIndicesLoader | None = None):
    path_dataset = Path(args.path_dataset)
    path_levels = Path(args.path_levels)
    path_network = Path(args.path_network)
    device = args.device
    batch_size = args.batch_size
    num_workers = int(args.cpu_per_trial)
    max_epochs = args.max_epochs
    n_cycles = args.n_cycles
    metadata_column = args.metadata_column

    dataset_kwargs = dict(return_original_set=True, random_state=random_state)
    if metadata_column is not None:
        dataset_kwargs['metadata_column'] = metadata_column
    if indices_loader is not None:
        dataset_kwargs['indices_loader'] = indices_loader
    train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
        path_dataset=path_dataset, **dataset_kwargs
    )
    train_set = torch.utils.data.ConcatDataset([train_set, val_set])
    train_loader, test_loader = build_data_loaders(
        train_set, test_set, batch_size=batch_size, num_workers=num_workers, device=device
    )
    output_dims = dataset.get_n_classes()

    genes = dataset.get_genes()
    base_graph = get_pyg_data(genes=genes, path_to_csv=path_network)
    base_graph = base_graph.to(device)

    max_levels = args.max_n_levels
    coarse_edges, parents_list = load_coarse_edges_for_diffpool(
        path_levels=path_levels, n_levels=max_levels, device=device
    )

    best_result = results.get_best_result(scope="last")
    config = best_result.config
    if n_cycles is not None:
        max_epochs = int(
            config["T_0"] * (1 - config["T_mult"] ** n_cycles) / (1 - config["T_mult"])
        )

    model = build_diffpool_model(
        base_graph=base_graph,
        coarse_edges=coarse_edges,
        parents_list=parents_list,
        output_dims=output_dims,
        n_hybrid=n_hybrid,
        max_filters=args.max_filters,
        max_clusters=args.max_clusters,
        dense_threshold=args.dense_threshold,
    )
    model = model.to(device=device)

    class_weights = None
    try:
        if args.use_train_set_weights:
            class_weights = dataset.get_class_weights()
        elif args.cohort_indices:
            class_weights = dataset.get_class_weights(cohort=args.cohort_indices)
    except Exception:
        pass
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=config["T_0"],
        T_mult=config["T_mult"],
        eta_min=config["eta_min"],
    )
    lambda_link_pred = config["lambda_link_pred"]
    lambda_entropy = config["lambda_entropy"]

    records = []
    predictions, labels = None, None
    for epoch in range(max_epochs):
        print(f"Epoch [{epoch + 1} / {max_epochs}]")

        model, train_metrics = train_epoch_clf(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            epoch=epoch,
            loss_fn=loss_fn,
            lambda_link_pred=lambda_link_pred,
            lambda_entropy=lambda_entropy,
        )

        if epoch == max_epochs - 1:
            test_metrics, (predictions, labels) = evaluate_clf(
                model=model,
                validation_loader=test_loader,
                device=device,
                return_outputs=True,
            )
            print("Test metrics: ", test_metrics)
        else:
            test_metrics = evaluate_clf(
                model=model, validation_loader=test_loader, device=device
            )
            print("Test metrics: ", test_metrics)

        records.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        })

    df_metrics = pd.DataFrame.from_records(records)
    return model, df_metrics, (predictions, labels), dataset.label_encoder.classes_


def analyze_final_model_results(
    df_metrics, outputs, labels, configs, output_dir, classes, model=None
):
    output_dir = Path(output_dir) / "final_model_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    df_outputs = pd.DataFrame(outputs, columns=classes)
    df_outputs["labels"] = labels
    df_outputs["labels"] = df_outputs["labels"].map(lambda x: classes[x])
    df_outputs.to_csv(output_dir / "outputs.csv")

    predictions = np.argmax(outputs, axis=1)
    data = {"predictions": predictions, "labels": labels}
    df_test_predictions = pd.DataFrame.from_dict(data)
    df_test_predictions["predictions"] = df_test_predictions["predictions"].map(
        lambda x: classes[x]
    )
    df_test_predictions["labels"] = df_test_predictions["labels"].map(
        lambda x: classes[x]
    )
    df_test_predictions.to_csv(output_dir / "predictions.csv")

    fig, ax = plot_confusion_matrix(
        df_test_predictions,
        true_label_column="labels",
        predicted_label_column="predictions",
    )
    savefig(fig, output_dir, "confusion_matrix")

    df_metrics.to_csv(output_dir / "metrics.csv")
    print(configs)
    write_json(obj=configs, file_path=(output_dir / "model_configs.json"))

    if model is not None:
        torch.save(model.state_dict(), output_dir / "final_model.pt")


def test_tuned_model(results, n_hybrid, args, path_experiment, random_state,
                     indices_loader: PCRunIndicesLoader | None = None):
    model, df_metrics, (predictions, labels), classes = train_and_test_model(
        results, n_hybrid=n_hybrid, args=args,
        path_experiment=path_experiment, random_state=random_state,
        indices_loader=indices_loader
    )
    model = model.to(device='cpu')

    analyze_final_model_results(
        df_metrics, predictions, labels,
        results.get_best_result(scope='last').config,
        output_dir=path_experiment, classes=classes, model=model,
    )

    if 'cuda' in args.device:
        torch.cuda.empty_cache()


def run_holdout(args, random_state, rep):
    if args.path_indices is not None:
        indices_loader = PCRunIndicesLoader(
            path_indices=args.path_indices, run=rep, cohort=args.cohort_indices
        )
    else:
        indices_loader = None

    for n_hybrid in range(0, min(args.max_n_levels, args.n_hybrid + 1)):
        hp_config = build_hp_config(args)
        path_experiment = (
            Path(args.path_output)
            / f"diffpool_hybrid{n_hybrid}_rep{rep}"
        )
        if path_experiment.exists():
            print(f"Path {path_experiment} already exists. Skipping")
            continue
        print(f"Path {path_experiment} does not exist.")

        if not args.tune:
            train_and_validate_model(
                hp_config=hp_config, args=args, n_hybrid=n_hybrid,
                random_state=random_state, rep=rep,
                indices_loader=indices_loader
            )
            continue

        path_ray = path_experiment / "ray_results"
        scheduler = ASHAScheduler(
            max_t=args.max_epochs,
            grace_period=args.max_epochs,
            reduction_factor=2,
        )
        trainable = partial(
            train_and_validate_model, args=args, n_hybrid=n_hybrid,
            random_state=random_state, indices_loader=indices_loader
        )
        trainable = set_trainable_resources(
            trainable, device=args.device,
            cpu_per_trial=args.cpu_per_trial, gpu_per_trial=args.gpu_per_trial,
        )
        tuner = build_tuner(trainable, scheduler, hp_config, args.num_samples, path_ray)
        results = tuner.fit()

        test_tuned_model(
            results, n_hybrid=n_hybrid, args=args,
            path_experiment=path_experiment, random_state=random_state,
            indices_loader=indices_loader
        )


def main():
    args = parse_args()

    n_holdouts = args.n_holdouts
    rng = np.random.default_rng(seed=123)
    for rep in range(n_holdouts):
        random_state = int(rng.integers(500))
        print("random state ", random_state)
        run_holdout(args=args, random_state=random_state, rep=rep)


if __name__ == "__main__":
    main()
