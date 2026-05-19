from pathlib import Path

import torch
import networkx as nx
from torch_geometric.utils import from_scipy_sparse_matrix

from pooling_genomic import datasets, networks, coarsening
from pooling_genomic.settings import PoolingGenomicSettings


def generate_graph_levels(
    path_dataset: Path = None,
    path_network: Path = None,
    path_output: Path = None
):
    settings = PoolingGenomicSettings()
    
    if path_dataset is None:
        path_dataset = settings.path_data / 'tcga_cohort_classification'

    if path_network is None:
        path_network = settings.path_data / 'networks/stringdb_top100pc.csv'

    if path_output is None:
        path_output = settings.path_data / 'networks/levels'
        path_output.mkdir(exist_ok=True, parents=True)

    # load genes available in the dataset
    train_set, val_set, test_set, dataset = datasets.get_tcga_cohort_classification_datasets(
        path_dataset=path_dataset, return_original_set=True
    )
    genes = dataset.get_genes()
    # get corresponding genes
    g = networks.get_pyg_data(genes=genes, path_to_csv=path_network)
    print(f"Num Nodes: {g.num_nodes}")

    # load network data
    nx_graph = networks.pyg_graph_to_networkx(g, g.num_nodes)
    print(f"Number of connected components: {nx.number_connected_components(nx_graph)}")
    adj = nx.to_scipy_sparse_array(nx_graph)

    # coarsen
    results = coarsening.HEM(adj, levels=8)
    
    # process and save results
    for lvl, (adj, parents) in enumerate(zip(results[0], results[1])):
        adj.setdiag(0)  # remove self loops

        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())  # min-max scaling
        parents = torch.from_numpy(parents)

        edge_index, edge_weight, parents = edge_index.long(), edge_weight.float(), parents.long()

        torch.save(
            edge_index, path_output / f"edge_index_lvl{lvl}.pt"
        )
        torch.save(
            edge_weight, path_output / f"edge_weight_lvl{lvl}.pt"
        )
        torch.save(
            parents, path_output / f"parents_lvl{lvl}.pt"
        )


if __name__ == "__main__":
    generate_graph_levels()