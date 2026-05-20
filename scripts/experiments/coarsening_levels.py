import argparse
from functools import partial
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler

from pooling_genomic.datasets import get_tcga_cohort_classification_datasets, get_genomic_classification_dataset, PCRunIndicesLoader
from pooling_genomic.models import build_coarsening_model
from pooling_genomic.networks import load_graph_levels
from pooling_genomic.settings import PoolingGenomicSettings
from pooling_genomic.engines import train_epoch_clf, evaluate_clf
from pooling_genomic.utils import plot_confusion_matrix, savefig, write_json


def build_coarsening_model_from_configs(configs):
    return build_coarsening_model(**configs)


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
    n_levels=1,
    random_state=123,
    rep=0,
    indices_loader: PCRunIndicesLoader | None = None
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
    n_cycles = max(args.n_cycles - 1, 1)  # reduce during validation
    using_ray_tune = args.tune
    weighted_pooling = args.weighted_pooling
    use_convs = args.use_convs
    metadata_column = args.metadata_column

    # hp config
    lr = hp_config["lr"]
    weight_decay = hp_config["weight_decay"]
    eta_min = hp_config["eta_min"]  # eta min = 0.00001
    T_0 = hp_config["T_0"]  # suggested: 1
    T_mult = hp_config["T_mult"]  # suggested 2
    lambda_l1 = hp_config["lambda_l1"] if weighted_pooling else None
    print("HELLO TUNING")
    graphs = load_graph_levels(
        path_levels=path_levels, n_levels=max_levels, device=device
    )
    if n_cycles is not None:
        max_epochs = int(T_0 * (1 - T_mult**n_cycles) / (1 - T_mult))

    # modularize this part so that I can have other datasets trained using the same functions
    # I think it might be possible to simply create a get_dataset(args) function or get_dataloders(args)
    train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
        path_dataset=path_dataset, return_original_set=True, random_state=random_state,
        metadata_column=metadata_column, indices_loader=indices_loader
    )
    train_loader, val_loader = build_data_loaders(
        train_set, val_set, batch_size=batch_size, num_workers=num_workers, device=device
    )
    output_dims = dataset.get_n_classes()

    model = build_coarsening_model(
        n_levels=n_levels,
        graphs=graphs[:n_levels] if n_levels > 0 else graphs[:1],
        output_dims=output_dims,
        weighted_pooling=weighted_pooling,
        use_convs=use_convs,
        device=device
    )
    print(model)
    model = model.to(device=device)
    try:
        if args.use_train_set_weights:
            print("Using weights from the complete training set")
            class_weights = dataset.get_class_weights()
        else:
            print(f"Using weights from the training set belonging to {args.cohort_indices}")
            class_weights = dataset.get_class_weights(cohort=args.cohort_indices)
    except Exception as e:
        class_weights = None
    print("class weights: ", class_weights)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
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
            lambda_l1_node_importances=lambda_l1,
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
                # torch.save((model.state_dict(), optimizer.state_dict()), path)

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
            print("-- Validation Metrics: {}".format(validation_metrics))

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
            / f"nlevels{n_levels}_rep{rep}_wpool{weighted_pooling}_convs{use_convs}"
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
    settings = PoolingGenomicSettings()

    parser = argparse.ArgumentParser()
    parser.add_argument("path_dataset", type=str, help="Path to in-folder dataset.")
    parser.add_argument(
        "--metadata-column", 
        type=str, 
        default=None,
        help="Column to use from the metadata file. Optional."
    )

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
        help="Number of levels that are available in the given path levels.",
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
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument(
        "--n-holdouts",
        type=int,
        default=5,
        help="Number of holdouts to execute. Repetitions already executed present in the output directory are skipped (default: %(default)s).",
    )
    parser.add_argument(
        "--path-indices",
        type=str,
        default=None,
        help='Path to direct indices of the dataset to use.'
    )
    parser.add_argument(
        "--cohort-indices",
        type=str,
        default=None,
        help='Cohort whose indices are to be used.'
    )
    parser.add_argument(
        "--use-train-set-weights", 
        action="store_true",
        help=(
            'If True and cohort-indices is given, compute class weights using all the train '
            'set instead of restricting the computation to the cohort-indices cohort.'
        )
    )
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
                checkpoint_score_attribute="accuracy", num_to_keep=1
            ),
        ),
        param_space=configs,
    )
    return tuner


def train_and_test_model(results, args, path_experiment, n_levels, random_state, 
                         indices_loader: PCRunIndicesLoader | None = None):
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
    use_convs = args.use_convs
    metadata_column = args.metadata_column

    graphs = load_graph_levels(
        path_levels=path_levels, n_levels=max_levels, device=device
    )

    train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
        path_dataset=path_dataset, return_original_set=True, random_state=random_state,
        metadata_column=metadata_column, indices_loader=indices_loader
    )
    train_set = ConcatDataset([train_set, val_set])
    train_loader, test_loader = build_data_loaders(
        train_set, test_set, batch_size=batch_size, num_workers=num_workers, device=device
    )
    output_dims = dataset.get_n_classes()

    best_result = results.get_best_result(scope="last")
    config = best_result.config
    if n_cycles is not None:
        max_epochs = int(
            config["T_0"] * (1 - config["T_mult"] ** n_cycles) / (1 - config["T_mult"])
        )

    model = build_coarsening_model(
        n_levels=n_levels,
        graphs=graphs[:n_levels] if n_levels > 0 else graphs[:1],
        output_dims=output_dims,
        weighted_pooling=weighted_pooling,
        use_convs=use_convs,
        device=device
    )
    model = model.to(device=device)
    try:
        if args.use_train_set_weights:
            print("Using weights from the complete training set")
            class_weights = dataset.get_class_weights()
        else:
            print(f"Using weights from the training set belonging to {args.cohort_indices}")
            class_weights = dataset.get_class_weights(cohort=args.cohort_indices)
    except:
        class_weights = None
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
    lambda_l1 = config["lambda_l1"] if weighted_pooling else None

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
            lambda_l1_node_importances=lambda_l1,
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

        records.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            }
        )

    df_metrics = pd.DataFrame.from_records(records)
    return model, df_metrics, (predictions, labels), dataset.label_encoder.classes_


def analyze_final_model_results(
    df_metrics: pd.DataFrame, outputs, labels, configs, output_dir, classes, model=None
):
    output_dir = Path(output_dir) / "final_model_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save network outputs and labels
    df_outputs = pd.DataFrame(outputs, columns=classes)
    df_outputs["labels"] = labels
    df_outputs["labels"] = df_outputs["labels"].map(lambda x: classes[x])
    df_outputs.to_csv(output_dir / "outputs.csv")

    # Save predictions and labels
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

    # Save confusion matrix
    fig, ax = plot_confusion_matrix(
        df_test_predictions,
        true_label_column="labels",
        predicted_label_column="predictions",
    )
    savefig(fig, output_dir, "confusion_matrix")

    # Save a few metrics per epoch
    df_metrics.to_csv(output_dir / "metrics.csv")

    # Save configs
    print(configs)
    write_json(obj=configs, file_path=(output_dir / "model_configs.json"))

    # Save model
    if model is not None:
        torch.save(model.state_dict(), output_dir / "final_model.pt")


def test_tuned_model(results, n_levels, args, path_experiment, random_state, indices_loader: PCRunIndicesLoader | None = None):
    model, df_metrics, (predictions, labels), classes = train_and_test_model(
        results,
        n_levels=n_levels,
        args=args,
        path_experiment=path_experiment,
        random_state=random_state,
        indices_loader=indices_loader
    )
    model = model.to(device='cpu')

    analyze_final_model_results(
        df_metrics,
        predictions,
        labels,
        results.get_best_result(scope='last').config,
        output_dir=path_experiment,
        classes=classes,
        model=model,
    )

    if 'cuda' in args.device:
        torch.cuda.empty_cache()


def run_holdout(args, random_state, rep):
    if args.path_indices is not None:
        indices_loader = PCRunIndicesLoader(path_indices=args.path_indices, run=rep, cohort=args.cohort_indices)
    else:
        indices_loader = None

    for n_levels in range(0, args.max_n_levels):
        for weighted_pooling in [False, True]:
            if n_levels == 0:
                if weighted_pooling == True:
                    continue
            
            for use_convs in [False, True]:
                if n_levels == 0:
                    if use_convs == True:
                        continue
                
                args.weighted_pooling = weighted_pooling
                args.use_convs = use_convs
                hp_config = build_hp_config(args)
                path_experiment = (
                    Path(args.path_output)
                    / f"nlevels{n_levels}_rep{rep}_wpool{weighted_pooling}_convs{use_convs}"
                )
                if path_experiment.exists():
                    print(f"Path of experiment {path_experiment} already exists. Skipping")
                    continue
                else:
                    print(f"Path {path_experiment} does not exist.")

                if not args.tune:
                    train_and_validate_model(
                        hp_config=hp_config,
                        args=args,
                        n_levels=n_levels,
                        random_state=random_state,
                        rep=rep,
                        indices_loader=indices_loader
                    )
                    continue

                path_ray = path_experiment / "ray_results"
                scheduler = ASHAScheduler(
                    max_t=args.max_epochs,
                    grace_period=args.max_epochs,  # let warm restarts work
                    reduction_factor=2,
                )
                trainable = partial(
                    train_and_validate_model,
                    args=args,
                    n_levels=n_levels,
                    random_state=random_state,
                    indices_loader=indices_loader
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
                results = tuner.fit()
                
                test_tuned_model(
                    results,
                    n_levels=n_levels,
                    args=args,
                    path_experiment=path_experiment,
                    random_state=random_state,
                    indices_loader=indices_loader
                )


def main():
    args = parse_args()
    # args.path_dataset = (
    #     "/home/thomas/Documents/PoolingGenomicGNNs/data/tcga_cohort_classification"
    # )
    # args.path_levels = "/home/thomas/Documents/PoolingGenomicGNNs/data/networks/levels"

    n_holdouts = args.n_holdouts
    rng = np.random.default_rng(seed=123)
    for rep in range(n_holdouts):
        random_state = int(rng.integers(500))
        print("random state ", random_state)
        run_holdout(args=args, random_state=random_state, rep=rep)


if __name__ == "__main__":
    main()

