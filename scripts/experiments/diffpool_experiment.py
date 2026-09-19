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
            pin_memory=pin_memory,
            multiprocessing_context='spawn' if num_workers > 0 else None,
        )
        loaders.append(dataset_loader)
    return tuple(loaders)


def _warn_mismatched_lambda_flags(args):
    """--lambda-link-pred/--lambda-entropy only apply to --pooling-type diffpool,
    and --lambda-modularity/--lambda-collapse only apply to --pooling-type dmon;
    build_hp_config silently zeroes whichever pair doesn't match, so warn loudly
    instead of leaving a passed flag with no visible effect."""
    diffpool_flags = {'--lambda-link-pred': args.lambda_link_pred, '--lambda-entropy': args.lambda_entropy}
    dmon_flags = {'--lambda-modularity': args.lambda_modularity, '--lambda-collapse': args.lambda_collapse}
    ignored = dmon_flags if args.pooling_type == 'diffpool' else diffpool_flags
    set_flags = [name for name, val in ignored.items() if val is not None]
    if set_flags:
        print(
            f"WARNING: --pooling-type {args.pooling_type} ignores {', '.join(set_flags)} "
            f"(not used by this pooling type) -- these values will have no effect."
        )


def _cosine_restart_epochs(T_0, T_mult, n_cycles):
    """Epoch count at which CosineAnnealingWarmRestarts(T_0, T_mult) completes
    n_cycles restarts, i.e. lands back at a converged trough instead of
    stopping mid-cycle. Used to align both the tuning-phase epoch budget
    (ASHAScheduler's max_t) and the final-retrain epoch budget to schedule
    boundaries, so validation reads never land on a noisy mid-cycle spike."""
    return int(T_0 * (1 - T_mult**n_cycles) / (1 - T_mult))


def build_hp_config(args):
    pooling_type = args.pooling_type
    _warn_mismatched_lambda_flags(args)
    if args.tune:
        hp_config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-4, 1e-1),
            "eta_min": 0.00001,
            "T_0": 1,
            "T_mult": 2,
        }
        if pooling_type == "dmon":
            hp_config["lambda_link_pred"] = 0.0
            hp_config["lambda_entropy"] = 0.0
            # Modularity is bounded in [-0.5, 1] and the collapse term in
            # [0, sqrt(max_clusters)-1] -- both O(1), unlike DiffPool's
            # link-pred/entropy losses which need weights near 1e-4. Start
            # the tuned range an order of magnitude or two higher.
            hp_config["lambda_modularity"] = tune.loguniform(1e-3, 1e1)
            hp_config["lambda_collapse"] = tune.loguniform(1e-3, 1e1)
        else:
            # narrowed to plan.md's documented Full DiffPool search range
            # (lambda_link, lambda_ent in [1e-5, 1e-3]) rather than the
            # wider [1e-5, 1e-1] that mostly samples values too large.
            hp_config["lambda_link_pred"] = tune.loguniform(1e-5, 1e-3)
            hp_config["lambda_entropy"] = tune.loguniform(1e-5, 1e-3)
            hp_config["lambda_modularity"] = 0.0
            hp_config["lambda_collapse"] = 0.0
    else:
        hp_config = {
            # plan.md: default lr=0.05 fails, lr=0.0008 is the value that
            # actually converges for Full DiffPool -- use that as the
            # non-tuning default instead of the known-failing 0.05.
            "lr": args.lr if args.lr is not None else 0.0008,
            "weight_decay": args.weight_decay if args.weight_decay is not None else 0.01,
            "eta_min": 0.00001,
            "T_0": 1,
            "T_mult": 2,
        }
        if pooling_type == "dmon":
            hp_config["lambda_link_pred"] = 0.0
            hp_config["lambda_entropy"] = 0.0
            hp_config["lambda_modularity"] = args.lambda_modularity if args.lambda_modularity is not None else 1.0
            hp_config["lambda_collapse"] = args.lambda_collapse if args.lambda_collapse is not None else 1.0
        else:
            hp_config["lambda_link_pred"] = args.lambda_link_pred if args.lambda_link_pred is not None else 0.001
            hp_config["lambda_entropy"] = args.lambda_entropy if args.lambda_entropy is not None else 0.001
            hp_config["lambda_modularity"] = 0.0
            hp_config["lambda_collapse"] = 0.0
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
    full_mode = args.full_mode

    lr = hp_config["lr"]
    weight_decay = hp_config["weight_decay"]
    eta_min = hp_config["eta_min"]
    T_0 = hp_config["T_0"]
    T_mult = hp_config["T_mult"]
    lambda_link_pred = hp_config["lambda_link_pred"]
    lambda_entropy = hp_config["lambda_entropy"]
    lambda_modularity = hp_config["lambda_modularity"]
    lambda_collapse = hp_config["lambda_collapse"]

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

    if full_mode:
        coarse_edges = None
        parents_list = None
    else:
        max_levels = args.max_n_levels
        coarse_edges, parents_list = load_coarse_edges_for_diffpool(
            path_levels=path_levels, n_levels=max_levels, device=device
        )

    if n_cycles is not None:
        max_epochs = _cosine_restart_epochs(T_0, T_mult, n_cycles)

    model = build_diffpool_model(
        base_graph=base_graph,
        coarse_edges=coarse_edges,
        parents_list=parents_list,
        output_dims=output_dims,
        n_hybrid=n_hybrid,
        max_filters=args.max_filters,
        max_clusters=args.max_clusters,
        full_mode=full_mode,
        n_levels=n_hybrid,
        encoder_channels=args.encoder_channels,
        encoder_layers=args.encoder_layers,
        pooling_type=args.pooling_type,
        sparsify_density=args.sparsify_density,
        assign_dropout=args.assign_dropout,
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
            lambda_modularity=lambda_modularity,
            lambda_collapse=lambda_collapse,
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
                # This used to compute `path` and never write to it -- every
                # checkpoint Ray Tune tracked across every run in this repo
                # was an empty directory (only Ray's own `.is_checkpoint`/
                # `.tune_metadata` bookkeeping files, no model weights). That
                # silently made `train_and_test_model`'s final retrain always
                # start from a fresh random init, with no way to warm-start
                # from the tuning trial that found these hyperparameters --
                # see changes-from-claude.md fix #7 for the failure this
                # caused (Hybrid DMoN n_hybrid=5 rep0: the exact hyperparameters
                # that reached 83.2% val accuracy during tuning collapsed to a
                # frozen uniform-16-class prediction on an independent re-init
                # during the final retrain).
                torch.save(model.state_dict(), path)
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

        mode_tag = "full" if args.full_mode else "hybrid"
        path_experiment = (
            Path(args.path_output)
            / f"{args.pooling_type}_{mode_tag}{n_hybrid}_rep{rep}"
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
                        help="Number of early levels that use hybrid mode (max n_hybrid to try)")
    parser.add_argument("--n-hybrid-start", type=int, default=0,
                        help="First n_hybrid value to try (for partial runs)")
    parser.add_argument("--max-filters", type=int, default=32,
                        help="Maximum feature dimension (grows progressivly: 1,2,4,...,max_filters)")
    parser.add_argument("--max-clusters", type=int, default=32,
                        help="Maximum clusters per DiffPoolLayer (bound on k)")
    parser.add_argument("--encoder-channels", type=int, default=16,
                        help="Full-mode pre-pooling encoder output width (1D-Conv -> ChebConv). "
                             "plan.md 5.3.3: 16ch is the config that reaches 71.68%% (vs 27.7%% with no encoder)")
    parser.add_argument("--encoder-layers", type=int, default=2,
                        help="Full-mode pre-pooling encoder depth (1 Conv1d layer + (layers-1) ChebConv layers)")

    parser.add_argument("--full-mode", action="store_true",
                        help="Use full learned pooling everywhere (no hybrid levels, no HEM coarse edges)")
    parser.add_argument("--pooling-type", choices=["diffpool", "dmon"], default="diffpool",
                        help="Learned-assignment mechanism used by full-mode pooling levels: "
                             "'diffpool' (link-pred + entropy losses) or 'dmon' (Deep Modularity "
                             "Networks -- modularity + collapse-regularization losses, "
                             "Tsitsulin et al. 2023). Hybrid levels are unaffected either way.")
    parser.add_argument("--assign-dropout", type=float, default=0.5,
                        help="Dropout applied to each layer's raw assignment logits before the "
                             "softmax, for both pooling types, whenever the learned-assignment "
                             "branch is reached (hybrid mode's trailing layer included). Tsitsulin "
                             "et al. (DMoN paper) use 0.5 and report it specifically prevents "
                             "gradient descent from getting stuck in a degenerate assignment.")
    parser.add_argument("--sparsify-density", type=float, default=None,
                        help="Full mode only: prune each level's pooled output adjacency to this "
                             "fraction of edges per node (e.g. 0.04, matching stringdb_top100pc.csv's "
                             "actual ~4%% density) before passing it to the next level, instead of "
                             "leaving it fully connected (every softmax-derived entry is nonzero, "
                             "so dense_to_sparse alone returns a complete graph). A fraction rather "
                             "than a fixed edge count, since full-mode levels span very different "
                             "widths (e.g. 1854 down to 32 in a 3-level schedule) and a fixed count "
                             "can't match the same density at more than one of them. Only matters "
                             "when multiple full-mode levels chain together -- the last level's "
                             "output isn't consumed by anything downstream. Default None disables "
                             "pruning (prior behavior).")

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (non-tuning only)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override weight_decay (non-tuning only)")
    parser.add_argument("--lambda-link-pred", type=float, default=None,
                        help="Override lambda_link_pred (non-tuning only, --pooling-type diffpool)")
    parser.add_argument("--lambda-entropy", type=float, default=None,
                        help="Override lambda_entropy (non-tuning only, --pooling-type diffpool)")
    parser.add_argument("--lambda-modularity", type=float, default=None,
                        help="Override lambda_modularity (non-tuning only, --pooling-type dmon)")
    parser.add_argument("--lambda-collapse", type=float, default=None,
                        help="Override lambda_collapse (non-tuning only, --pooling-type dmon)")

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
    full_mode = args.full_mode

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

    if full_mode:
        coarse_edges = None
        parents_list = None
    else:
        max_levels = args.max_n_levels
        coarse_edges, parents_list = load_coarse_edges_for_diffpool(
            path_levels=path_levels, n_levels=max_levels, device=device
        )

    best_result = results.get_best_result(scope="all")
    config = best_result.config
    if n_cycles is not None:
        max_epochs = _cosine_restart_epochs(config["T_0"], config["T_mult"], n_cycles)

    model = build_diffpool_model(
        base_graph=base_graph,
        coarse_edges=coarse_edges,
        parents_list=parents_list,
        output_dims=output_dims,
        n_hybrid=n_hybrid,
        max_filters=args.max_filters,
        max_clusters=args.max_clusters,
        full_mode=full_mode,
        n_levels=n_hybrid,
        encoder_channels=args.encoder_channels,
        encoder_layers=args.encoder_layers,
        pooling_type=args.pooling_type,
        sparsify_density=args.sparsify_density,
        assign_dropout=args.assign_dropout,
    )
    model = model.to(device=device)

    # Warm-start from the tuning trial's own best-epoch checkpoint instead of
    # training from a fresh random init with the same hyperparameters. Without
    # this, the final retrain re-rolls initialization independently of the
    # tuning phase that selected `config` -- these hyperparameters are only
    # known to work from *that* trial's specific init; a different init can
    # land in a materially different basin (see changes-from-claude.md fix #7:
    # this exact failure mode collapsed a Hybrid DMoN n_hybrid=5 rep to a
    # frozen uniform-class prediction despite its hyperparameters reaching
    # 83.2% val accuracy during tuning). Falls back to the fresh init (with a
    # warning) if no checkpoint is available -- e.g. results loaded from a
    # run predating this fix, where every checkpoint directory is empty.
    checkpoint = best_result.checkpoint
    if checkpoint is not None:
        try:
            with checkpoint.as_directory() as checkpoint_dir:
                state_dict_path = Path(checkpoint_dir) / "checkpoint"
                state_dict = torch.load(state_dict_path, map_location=device)
                model.load_state_dict(state_dict)
            print(f"Warm-started final retrain from best trial's checkpoint: {state_dict_path.name}")
        except (FileNotFoundError, RuntimeError) as e:
            print(
                f"WARNING: could not warm-start from best trial's checkpoint ({e}) -- "
                "falling back to a fresh random init. If this run predates "
                "changes-from-claude.md fix #7, every checkpoint from it is "
                "empty by construction; this is expected, not a new bug."
            )
    else:
        print("WARNING: best trial has no checkpoint -- training final retrain from a fresh random init.")

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
    lambda_modularity = config["lambda_modularity"]
    lambda_collapse = config["lambda_collapse"]

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
            lambda_modularity=lambda_modularity,
            lambda_collapse=lambda_collapse,
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
        results.get_best_result(scope='all').config,
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

    mode_tag = "full" if args.full_mode else "hybrid"
    # full_mode has no valid n_hybrid=0 architecture (zero pooling levels) --
    # skip it rather than let it crash deep in the model's forward pass.
    n_hybrid_start = max(args.n_hybrid_start, 1) if args.full_mode else args.n_hybrid_start
    for n_hybrid in range(n_hybrid_start, min(args.max_n_levels, args.n_hybrid + 1)):
        hp_config = build_hp_config(args)
        path_experiment = (
            Path(args.path_output)
            / f"{args.pooling_type}_{mode_tag}{n_hybrid}_rep{rep}"
        )
        if path_experiment.exists():
            print(f"Path {path_experiment} already exists. Skipping")
            continue
        print(f"Path {path_experiment} does not exist.")

        if not args.tune:
            try:
                train_and_validate_model(
                    hp_config=hp_config, args=args, n_hybrid=n_hybrid,
                    random_state=random_state, rep=rep,
                    indices_loader=indices_loader
                )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                    print(f"OOM for n_hybrid={n_hybrid}, skipping: {e}")
                    if 'cuda' in args.device:
                        torch.cuda.empty_cache()
                else:
                    raise
            continue

        path_ray = path_experiment / "ray_results"
        # Tuning-phase epoch budget aligned to a CosineAnnealingWarmRestarts
        # trough (T_0=1, T_mult=2, matching build_hp_config) instead of the
        # arbitrary --max-epochs default -- otherwise trials get cut off
        # mid-cycle, where loss is still oscillating, and get_best_result
        # ends up comparing noise instead of converged validation loss.
        tune_n_cycles = max(args.n_cycles - 1, 1)
        tune_max_epochs = _cosine_restart_epochs(1, 2, tune_n_cycles)
        if tune_n_cycles > 1:
            # grace_period at the *previous* restart boundary rather than at
            # max_t, so ASHA gets one real pruning checkpoint instead of none
            # (grace_period == max_t means no trial is ever pruned early). The
            # 2^n-1 restart sequence's consecutive-boundary ratio converges to
            # reduction_factor=2 as n grows (3, 2.33, 2.14, 2.07, 2.03, ...), so
            # the rung Ray computes automatically (max_t / reduction_factor)
            # lands within ~1 epoch of this boundary -- still a converged
            # trough, not a mid-cycle read.
            grace_n_cycles = tune_n_cycles - 1
            grace_period = _cosine_restart_epochs(1, 2, grace_n_cycles)
            scheduler = ASHAScheduler(
                max_t=tune_max_epochs,
                grace_period=grace_period,
                reduction_factor=2,
            )
        else:
            # tune_n_cycles == 1 -> tune_max_epochs == 1 epoch: there is no
            # earlier restart boundary to use as a grace_period, so early
            # pruning isn't meaningful at this budget (every trial gets
            # exactly one epoch before being judged regardless). Previously
            # this fell through to grace_n_cycles = max(tune_n_cycles - 1, 1)
            # == 1, silently reconstructing grace_period == max_t -- the
            # exact "ASHA never prunes" no-op this whole block exists to
            # avoid, just via a different path. Skip ASHA outright instead so
            # the log says plainly that no pruning is happening, rather than
            # constructing a scheduler that looks active but isn't.
            print(
                f"WARNING: --n-cycles={args.n_cycles} gives a {tune_max_epochs}-epoch "
                "tuning budget -- too small for any early-stopping rung. Running "
                "without ASHA pruning (every trial runs to completion)."
            )
            scheduler = None
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
