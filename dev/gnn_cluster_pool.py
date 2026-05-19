from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN
from torch_geometric.nn.conv import GCNConv, ChebConv

from pooling_genomic.datasets import get_tcga_cohort_classification_datasets
from pooling_genomic.engines import evaluate_clf, train_epoch_clf
from pooling_genomic.settings import PoolingGenomicSettings
from pooling_genomic import networks, models
from pooling_genomic.utils import get_lr


def get_model(path_levels: Path, device: str = 'cpu'):
    randomize_clusters = False
    edge_index_0, edge_weight_0, parents_0 = networks.load_graph_level(
        path_levels=path_levels, level=0, device=device, randomize_clusters=randomize_clusters
    )
    edge_index_1, edge_weight_1, parents_1 = networks.load_graph_level(
        path_levels=path_levels, level=1, device=device, randomize_clusters=randomize_clusters
    )
    edge_index_2, edge_weight_2, parents_2 = networks.load_graph_level(
        path_levels=path_levels, level=2, device=device, randomize_clusters=randomize_clusters
    )
    edge_index_3, edge_weight_3, parents_3 = networks.load_graph_level(
        path_levels=path_levels, level=3, device=device, randomize_clusters=randomize_clusters
    )
    edge_index_4, edge_weight_4, parents_4 = networks.load_graph_level(
        path_levels=path_levels, level=4, device=device, randomize_clusters=randomize_clusters
    )
    edge_index_5, edge_weight_5, parents_5 = networks.load_graph_level(
        path_levels=path_levels, level=5, device=device, randomize_clusters=randomize_clusters
    )
    edge_index_6, edge_weight_6, parents_6 = networks.load_graph_level(
        path_levels=path_levels, level=6, device=device, randomize_clusters=randomize_clusters
    )
    edge_index_7, edge_weight_7, parents_7 = networks.load_graph_level(
        path_levels=path_levels, level=7, device=device, randomize_clusters=randomize_clusters
    )


    g_0 = Data(edge_index=edge_index_0, edge_weight=edge_weight_0)
    g_0.num_nodes = len(parents_0)
    g_0.cluster_indices = parents_0

    g_1 = Data(edge_index=edge_index_1, edge_weight=edge_weight_1)
    g_1.num_nodes = len(parents_1)
    g_1.cluster_indices = parents_1

    g_2 = Data(edge_index=edge_index_2, edge_weight=edge_weight_2)
    g_2.num_nodes = len(parents_2)
    g_2.cluster_indices = parents_2

    g_3 = Data(edge_index=edge_index_3, edge_weight=edge_weight_3)
    g_3.num_nodes = len(parents_3)
    g_3.cluster_indices = parents_3

    g_4 = Data(edge_index=edge_index_4, edge_weight=edge_weight_4)
    g_4.num_nodes = len(parents_4)
    g_4.cluster_indices = parents_4

    g_5 = Data(edge_index=edge_index_5, edge_weight=edge_weight_5)
    g_5.num_nodes = len(parents_5)
    g_5.cluster_indices = parents_5

    g_6 = Data(edge_index=edge_index_6, edge_weight=edge_weight_6)
    g_6.num_nodes = len(parents_6)
    g_6.cluster_indices = parents_6

    g_7 = Data(edge_index=edge_index_7, edge_weight=edge_weight_7)
    g_7.num_nodes = len(parents_7)
    g_7.cluster_indices = parents_7

    # gcn_0 = GCN(in_channels=1, hidden_channels=2, num_layers=1).to(device)
    # gcn_1 = GCN(in_channels=2, hidden_channels=2, num_layers=1).to(device)
    # gcn_2 = GCN(in_channels=1, hidden_channels=16, num_layers=1).to(device)
    # gcn_3 = GCN(in_channels=1, hidden_channels=4, num_layers=1).to(device)
    # gcn_4 = GCN(in_channels=1, hidden_channels=16, num_layers=2).to(device)

    # gcn_0 = ChebConv(in_channels=1, out_channels=4, K=1).to(device)
    # gcn_1 = ChebConv(in_channels=4, out_channels=16, K=1).to(device)
    # gcn_2 = ChebConv(in_channels=16, out_channels=32, K=1).to(device)
    # gcn_3 = ChebConv(in_channels=32, out_channels=32, K=1).to(device)
    # gcn_4 = ChebConv(in_channels=32, out_channels=16, K=1).to(device)

    gcn_0 = ChebConv(in_channels=1, out_channels=4, K=1).to(device)
    gcn_1 = ChebConv(in_channels=4, out_channels=8, K=1).to(device)
    gcn_2 = ChebConv(in_channels=8, out_channels=16, K=1).to(device)
    gcn_3 = ChebConv(in_channels=16, out_channels=16, K=1).to(device)
    gcn_4 = ChebConv(in_channels=16, out_channels=16, K=1).to(device)
    gcn_5 = ChebConv(in_channels=16, out_channels=16, K=1).to(device)
    gcn_6 = ChebConv(in_channels=16, out_channels=16, K=1).to(device)
    gcn_7 = ChebConv(in_channels=16, out_channels=4, K=1).to(device)

    num_super_nodes = np.unique(g_7.cluster_indices.cpu()).shape[0]
    print(f"Num super nodes: {num_super_nodes}")

    # graphs = [g_0, g_1, g_2, g_3, g_4]
    # gnn_modules = [None, None, gcn_2, gcn_3, gcn_4]

    graphs = [g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7]
    # gnn_modules = [gcn_0, gcn_1, gcn_2, gcn_3, gcn_4, gcn_5, gcn_6, gcn_7]
    gnn_modules = [None, None, None, None, None, None, None, None]

    # Define MLP
    mlp = models.FCModel(
        input_dim=num_super_nodes * 1, hidden_dim=(256,), output_dim=32
    ).to(device)

    model = models.GNNClassifier(
        gcn_module=gnn_modules, mlp_module=mlp, graph=graphs, pooling="cluster_pool"
    ).to(device)

    return model


def main():
    device = "cuda"

    settings = PoolingGenomicSettings()
    path_levels = settings.path_data / "networks/levels"
    path_dataset = settings.path_data / "tcga_cohort_classification"

    model = get_model(path_levels=path_levels, device=device)

    path_dataset = settings.path_data / "tcga_cohort_classification"

    train_set, val_set, test_set, dataset = get_tcga_cohort_classification_datasets(
        path_dataset=path_dataset, return_original_set=True
    )

    batch_size = 128
    num_workers = 2

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )

    try:
        class_weights = dataset.get_class_weights().to(device)
    except:
        class_weights = None

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("1 - Training with Warm Restarts")
    learning_rate = 0.005
    weight_decay = 0.001
    max_epochs = 30
    max_epochs_initialization = 15

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=2, T_mult=2, eta_min=0.00001
    )
    n_epochs = max_epochs
    for epoch in range(max_epochs_initialization):
        print(f"Warm Restarts Epoch [{epoch+1}/{max_epochs_initialization}]")
        model, train_metrics = train_epoch_clf(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            scheduler=scheduler,
            epoch=epoch,
            print_period=1,
        )
        print("Train metrics: ", train_metrics)
        test_metrics = evaluate_clf(
            model=model, validation_loader=test_loader, device=device
        )
        print("Test metrics: ", test_metrics)

    print("2 - Training (reducing LR on plateau)")
    lr = get_lr(optimizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    records = []
    predictions, labels = None, None
    for epoch in range(n_epochs):
        print(f"Epoch [{epoch+1}/{n_epochs}]")
        model, train_metrics = train_epoch_clf(
            model=model, train_loader=train_loader, optimizer=optimizer, device=device, loss_fn=loss_fn, scheduler=scheduler, epoch=epoch
        )
        print("Train metrics: ", train_metrics)

        if epoch == n_epochs - 1:
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
                "test_balanced_accuracy": test_metrics["balanced_accuracy"]
            }
        )

        scheduler.step(train_metrics["loss"])

    df_metrics = pd.DataFrame.from_records(records)
    df_metrics.to_csv(f"{datetime.now().isoformat()}_metrics.csv")



if __name__ == "__main__":
    main()
