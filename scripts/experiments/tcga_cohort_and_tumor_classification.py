import argparse
from functools import partial
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt

from pooling_genomic.datasets import get_genomic_classification_dataset, get_tcga_cohort_and_tumor_classification_datasets, get_tcga_cohort_classification_datasets
from pooling_genomic.models import CohortAndTumorLoss, build_coarsening_model, build_fixed_supernodes_coarsening_model, build_gnn_pooling_tumor_and_cohort_clf, get_coarsening_convs_list, get_fixed_supernodes_convs_list
from pooling_genomic.networks import load_graph_levels
from pooling_genomic.settings import PoolingGenomicSettings
from pooling_genomic.engines import evaluate_cohort_tumor_clf, train_cohort_tumor_clf, evaluate_clf
from pooling_genomic.utils import plot_confusion_matrix, savefig, write_json
from pooling_genomic.utils import build_data_loaders


def build_coarsening_model_from_configs(configs):
    return build_fixed_supernodes_coarsening_model(**configs)


def build_tcga_cohort_and_tumor_model(**kwargs):
    graphs = kwargs['graphs']
    cohort_output_dims = kwargs['cohort_output_dims']
    max_levels = kwargs['max_levels']
    save_embedding_grad = kwargs['save_embedding_grad'] if 'save_embedding_grad' in kwargs else None
    weighted_pooling = True
    mlp_hidden_dim = (256, )

    first_level = None
    first_level = 4

    n_levels = None
    # n_levels = 2

    if first_level is not None:
        convs, out_channels = get_fixed_supernodes_convs_list(
            max_levels=max_levels, first_level=first_level
        )

        num_super_nodes = np.unique(graphs[-1].cluster_indices.cpu()).shape[0]

        model = build_gnn_pooling_tumor_and_cohort_clf(
            graphs=graphs,
            gnns=convs,
            mlp_input_dim=num_super_nodes * out_channels,
            mlp_cohort_output_dim=cohort_output_dims,
            weighted_pooling=weighted_pooling,
            save_embedding_grad=save_embedding_grad,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        return model

    if n_levels is not None:
        use_convs = True
        graphs_nl = graphs[:n_levels]
        
        convs, out_channels = get_coarsening_convs_list(n_levels=n_levels, use_convs=use_convs)
        num_super_nodes = np.unique(graphs[-1].cluster_indices.cpu()).shape[0]
        model = build_gnn_pooling_tumor_and_cohort_clf(
            graphs=graphs_nl,
            gnns=convs,
            mlp_input_dim=num_super_nodes * out_channels,
            mlp_cohort_output_dim=cohort_output_dims,
            weighted_pooling=weighted_pooling,
            save_embedding_grad=save_embedding_grad,
            mlp_hidden_dim=mlp_hidden_dim,
            **kwargs,
        )

        return model


def build_hp_config(args):
    if args.tune:
        if args.weighted_pooling:
            hp_config = {
                "lr": tune.loguniform(1e-4, 1e-1),
                "weight_decay": tune.loguniform(1e-4, 1e-1),
                "lambda_l1": tune.loguniform(1e-5, 1e-1),
                "eta_min": 0.00001,
                "T_0": 1,
                "T_mult": 2,
            }
        else:
            hp_config = {
                "lr": tune.loguniform(1e-4, 1e-1),
                "weight_decay": tune.loguniform(1e-4, 1e-1),
                "eta_min": 0.00001,
                "T_0": 1,
                "T_mult": 2,
            }

        return hp_config

    hp_config = {
        "lr": 0.05,
        "weight_decay": 0.01,
        "eta_min": 0.00001,
        "T_0": 1,
        "T_mult": 2,
    }
    if args.weighted_pooling:
        hp_config["lambda_l1"] = 0.001

    return hp_config


def train_and_validate_model(
    hp_config,
    args,
    first_level=1,
    random_state=123,
):
    print("HELLO TUNING")

    # args
    max_levels = args.max_n_levels
    path_dataset = Path(args.path_dataset)
    path_levels = Path(args.path_levels)
    path_output = Path(args.path_output)
    device = args.device
    batch_size = args.batch_size
    num_workers = int(args.cpu_per_trial)
    max_epochs = args.max_epochs
    n_cycles = args.n_cycles
    using_ray_tune = args.tune
    weighted_pooling = args.weighted_pooling

    # hp config
    lr = hp_config["lr"]
    weight_decay = hp_config["weight_decay"]
    eta_min = hp_config["eta_min"]  # eta min = 0.00001
    T_0 = hp_config["T_0"]  # suggested: 1
    T_mult = hp_config["T_mult"]  # suggested 2
    lambda_l1 = hp_config["lambda_l1"] if weighted_pooling else None
    graphs = load_graph_levels(
        path_levels=path_levels, n_levels=max_levels, device=device
    )
    if n_cycles is not None:
        max_epochs = int(T_0 * (1 - T_mult**n_cycles) / (1 - T_mult))

    train_set, val_set, test_set, dataset = get_tcga_cohort_and_tumor_classification_datasets(
        path_dataset=path_dataset, return_original_set=True, random_state=random_state
    )
    train_loader, val_loader = build_data_loaders(
        train_set, val_set, batch_size=batch_size, num_workers=num_workers, device=device
    )
    cohort_output_dims = dataset.get_n_cohorts()

    model = build_tcga_cohort_and_tumor_model(
        graphs=graphs,
        cohort_output_dims=cohort_output_dims,
        max_levels=max_levels
    )
    print(model)
    model = model.to(device=device)
    c_w, t_w = dataset.get_class_weights()
    loss_fn = CohortAndTumorLoss(cohort_weights=c_w, type_weights=t_w)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    for epoch in range(max_epochs):
        print(f"Epoch [{epoch + 1} / {max_epochs}]")

        model, train_metrics = train_cohort_tumor_clf(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            epoch=epoch,
            loss_fn=loss_fn,
            lambda_l1_node_importances=lambda_l1,
        )
        validation_metrics = evaluate_cohort_tumor_clf(
            model=model, validation_loader=val_loader, device=device, loss_fn=loss_fn
        )

        val_loss = validation_metrics["loss"]
        accuracy = validation_metrics["accuracy"]
        accuracy_type = validation_metrics["accuracy_type"]
        train_loss = train_metrics["loss"]
        train_accuracy = train_metrics["accuracy"]
        train_accuracy_type = train_metrics["accuracy_type"]

        if using_ray_tune:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = str(Path(checkpoint_dir) / "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            tune.report(
                loss=val_loss,
                accuracy=accuracy,
                accuracy_type=accuracy_type,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                train_accuracy_type=train_accuracy_type,
                checkpoint=checkpoint_dir,
            )
        else:
            print("-- Validation accuracy: {:.2f}".format(accuracy))
            print("-- Validation accuracy type: {:.2f}".format(accuracy_type))
            print("-- Validation loss: {:.4f}".format(val_loss))
            print("-- Train loss: {:.4f}".format(train_loss))
            print("-- Train accuracy: {:.4f}".format(train_accuracy))
            print("-- Train accuracy type: {:.4f}".format(train_accuracy_type))
            print("-- Validation Metrics: {}".format(validation_metrics))


def parse_args():
    settings = PoolingGenomicSettings()

    parser = argparse.ArgumentParser()
    parser.add_argument("path_dataset", type=str, help="Path to in-folder dataset.")

    parser.add_argument(
        "path_levels",
        type=str,
        # default=settings.path_data / "networks/levels",
        help="Path to directory containing graphs and their clusters and weights",
    )

    parser.add_argument(
        "--max-n-levels",
        type=int,
        default=8,
        help="Number of coarsening levels to use.",
    )

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    parser.add_argument(
        "--tune", action="store_true", help="Tune hyperparameters using raytune"
    )

    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--cpu-per-trial",
        type=float,
        default=1,
        help="CPUs needed per trial (default: %(default)s).",
    )
    parser.add_argument(
        "--gpu-per-trial",
        type=float,
        default=0.1,
        help="GPUs needed per trial (default: %(default)s).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of times to sample from the search space (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of times to sample from the search space (default: %(default)s).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of epochs for fitting each model (default: %(default)s).",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=5,
        help="Number of cycles of warm restarts. The number of epochs is calculated automatically (default: %(default)s).",
    )
    parser.add_argument(
        "--path-output",
        type=str,
        default="./outputs",
        help="Output path to results (default: %(default)s).",
    )

    parser.add_argument(
        "--path-tune-results",
        type=str,
        default=None,
        help="If given, loads results from given path instead of tuning (default: %(default)s).",
    )
    parser.add_argument("--save-final-model", action="store_true")
    args = parser.parse_args()
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
                checkpoint_score_attribute="loss", num_to_keep=1
            ),
        ),
        param_space=configs,
    )
    return tuner


def train_and_test_model(results, args, path_experiment, random_state):
    max_levels = args.max_n_levels
    path_dataset = Path(args.path_dataset)
    path_levels = Path(args.path_levels)
    path_output = Path(args.path_output)
    device = args.device
    batch_size = args.batch_size
    num_workers = int(args.cpu_per_trial)
    max_epochs = args.max_epochs
    n_cycles = args.n_cycles
    using_ray_tune = args.tune
    weighted_pooling = args.weighted_pooling

    graphs = load_graph_levels(
        path_levels=path_levels, n_levels=max_levels, device=device
    )

    train_set, val_set, test_set, dataset = get_tcga_cohort_and_tumor_classification_datasets(
        path_dataset=path_dataset, return_original_set=True, random_state=random_state
    )
    train_set = ConcatDataset([train_set, val_set])
    train_loader, test_loader = build_data_loaders(
        train_set, test_set, batch_size=batch_size, num_workers=num_workers, device=device
    )
    cohort_output_dims = dataset.get_n_cohorts()

    best_result = results.get_best_result(scope="last")
    config = best_result.config
    if n_cycles is not None:
        max_epochs = int(
            config["T_0"] * (1 - config["T_mult"] ** n_cycles) / (1 - config["T_mult"])
        )

    model = build_tcga_cohort_and_tumor_model(
        graphs=graphs,
        cohort_output_dims=cohort_output_dims,
        max_levels=max_levels
    )
    print(model)
    model = model.to(device=device)
    c_w, t_w = dataset.get_class_weights()
    loss_fn = CohortAndTumorLoss(cohort_weights=c_w, type_weights=t_w)

    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=config["T_0"],
        T_mult=config["T_mult"],
        eta_min=config["eta_min"],
    )
    lambda_l1 = config["lambda_l1"] if weighted_pooling else None

    records = []
    for epoch in range(max_epochs):
        print(f"Epoch [{epoch + 1} / {max_epochs}]")

        model, train_metrics = train_cohort_tumor_clf(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            epoch=epoch,
            loss_fn=loss_fn,
            lambda_l1_node_importances=lambda_l1,
        )

        if epoch == max_epochs - 1:
            test_metrics, output_and_labels = evaluate_cohort_tumor_clf(
                model=model,
                validation_loader=test_loader,
                device=device,
                loss_fn=loss_fn,
                return_outputs=True,
            )
            print("Test metrics: ", test_metrics)
        else:
            test_metrics = evaluate_cohort_tumor_clf(
                model=model, validation_loader=test_loader, device=device, loss_fn=loss_fn,
            )
            print("Test metrics: ", test_metrics)

        records.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_accuracy_type": train_metrics["accuracy_type"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_accuracy_type": test_metrics["accuracy_type"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            }
        )

    df_metrics = pd.DataFrame.from_records(records)
    return model, df_metrics, output_and_labels, (dataset.cohorts_encoder.classes_, dataset.types_encoder.classes_)


def analyze_final_model_results(
    df_metrics: pd.DataFrame, outputs_and_labels, configs, output_dir, cohorts, types, model=None
):
    output_dir = Path(output_dir) / "final_model_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save network outputs and labels for cohort classification
    outputs = outputs_and_labels['cohort_outputs']
    labels = outputs_and_labels['cohort_labels']
    df_outputs = pd.DataFrame(outputs, columns=cohorts)
    df_outputs["labels"] = labels
    df_outputs["labels"] = df_outputs["labels"].map(lambda x: cohorts[x])
    df_outputs.to_csv(output_dir / "cohort_outputs.csv")

    # Save predictions and labels for cohort classification
    predictions = np.argmax(outputs, axis=1)
    data = {"predictions": predictions, "labels": labels}
    df_test_predictions = pd.DataFrame.from_dict(data)
    df_test_predictions["predictions"] = df_test_predictions["predictions"].map(
        lambda x: cohorts[x]
    )
    df_test_predictions["labels"] = df_test_predictions["labels"].map(
        lambda x: cohorts[x]
    )
    df_test_predictions.to_csv(output_dir / "cohort_predictions.csv")

    # Save confusion matrix
    fig, ax = plot_confusion_matrix(
        df_test_predictions,
        true_label_column="labels",
        predicted_label_column="predictions",
    )
    savefig(fig, output_dir, "cohort_confusion_matrix")
    plt.close(fig)

    # Save network outputs and labels for type classification
    type_outputs = outputs_and_labels['type_outputs']
    type_predictions = outputs_and_labels['type_predictions']
    type_labels = outputs_and_labels['type_labels']
    # df_outputs = pd.DataFrame(type_outputs, columns=types)
    df_outputs = pd.DataFrame.from_dict({
        types[0]: type_outputs,
    })
    df_outputs["labels"] = type_labels
    df_outputs["labels"] = df_outputs["labels"].map(lambda x: types[int(x)])
    df_outputs.to_csv(output_dir / "type_outputs.csv")

    # Save predictions and labels for type classification
    data = {"predictions": type_predictions, "labels": type_labels}
    df_test_predictions = pd.DataFrame.from_dict(data)
    df_test_predictions["predictions"] = df_test_predictions["predictions"].map(
        lambda x: types[int(x)]
    )
    df_test_predictions["labels"] = df_test_predictions["labels"].map(
        lambda x: types[int(x)]
    )
    df_test_predictions.to_csv(output_dir / "type_predictions.csv")

    # Save confusion matrix
    fig, ax = plot_confusion_matrix(
        df_test_predictions,
        true_label_column="labels",
        predicted_label_column="predictions",
    )
    savefig(fig, output_dir, "type_confusion_matrix")
    plt.close(fig)

    # Save a few metrics per epoch
    df_metrics.to_csv(output_dir / "metrics.csv")

    # Save configs
    print(configs)
    write_json(obj=configs, file_path=(output_dir / "model_configs.json"))

    # Save model
    if model is not None:
        torch.save(model.state_dict(), output_dir / "final_model.pt")


def test_tuned_model(results, args, path_experiment, random_state):
    model, df_metrics, outputs_and_labels, (cohorts, types) = train_and_test_model(
        results,
        args=args,
        path_experiment=path_experiment,
        random_state=random_state,
    )
    model = model.to(device='cpu')

    analyze_final_model_results(
        df_metrics,
        outputs_and_labels,
        results.get_best_result(scope='last').config,
        output_dir=path_experiment,
        cohorts=cohorts,
        types=types,
        model=model,
    )

    if 'cuda' in args.device:
        torch.cuda.empty_cache()


def run_holdout(args, random_state, rep):
    args.weighted_pooling = True

    hp_config = build_hp_config(args)
    path_experiment = (
        Path(args.path_output)
        / f"tcga_cohort_and_tumor_rep{rep}"
    )
    if not args.tune:
        train_and_validate_model(
            hp_config=hp_config,
            args=args,
            random_state=random_state,
        )
        return
    
    path_ray = path_experiment / "ray_results"
    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=args.max_epochs,  # let warm restarts work
        reduction_factor=2,
    )
    trainable = partial(
        train_and_validate_model,
        args=args,
        random_state=random_state,
    )
    trainable = set_trainable_resources(
        trainable,
        device=args.device,
        cpu_per_trial=args.cpu_per_trial,
        gpu_per_trial=args.gpu_per_trial,
    )
    tuner = build_tuner(
        trainable, scheduler, hp_config, args.num_samples, path_ray
    )

    if not args.path_tune_results:
        results = tuner.fit()
    else:
        path_ray_resuls =args.path_tune_results
        print(f"Loading from {path_ray_resuls}")
        results = tuner.restore(path_ray_resuls).get_results()

    test_tuned_model(
        results,
        args=args,
        path_experiment=path_experiment,
        random_state=random_state,
    )


def main():
    args = parse_args()

    n_holdouts = 1
    rng = np.random.default_rng(seed=123)
    for rep in range(n_holdouts):
        random_state = int(rng.integers(500))
        print("random state ", random_state)
        run_holdout(args=args, random_state=random_state, rep=rep)


if __name__ == "__main__":
    main()


