from pathlib import Path
from typing import List

import torch
import joblib
import pandas as pd
import networkx as nx
from torch_geometric.data import Data


def get_pyg_data(genes: list, path_to_csv: Path, random_permutation: bool = False, include_singletons: bool = True):
    """Get the stringdb network in a pyg Data format

    Parameters
    ----------
    genes : list
        Genes to retain in the network. The nodes of the network will be indexed according to the order of the genes.
    path_to_csv : Path
        Path to pandas edge list in a CSV file.
    """
    df_edges = pd.read_csv(path_to_csv)
    if not include_singletons:
        connected_genes = df_edges['protein1'].unique()
        connected_genes = pd.Index(connected_genes).union(df_edges['protein2'].unique())
        genes = connected_genes.intersection(genes)

    mapping = {gene: int(idx) for idx, gene in enumerate(genes, 0)}

    keep_mask = (df_edges['protein1'].isin(genes) & df_edges['protein2'].isin(genes))
    df_edges = df_edges.loc[keep_mask, :]
    df_edges['protein1'] = df_edges.loc[:, 'protein1'].map(lambda x: int(mapping.get(x)))
    df_edges['protein2'] = df_edges.loc[:, 'protein2'].map(lambda x: int(mapping.get(x)))
    
    # now make edge attr
    edge_index = torch.from_numpy(df_edges[['protein1', 'protein2']].to_numpy().transpose()).long()
    if random_permutation:
        indexes = torch.randperm(edge_index.shape[1])
        edge_index = edge_index[:, indexes]

    edge_weight = torch.from_numpy(df_edges['combined_score'].to_numpy() / 1000).float()

    pyg_graph = Data(edge_index=edge_index, edge_weight=edge_weight)
    pyg_graph.num_nodes = len(genes)
    return pyg_graph


def pyg_graph_to_networkx(pyg_graph, num_nodes):
    list_of_edges = pyg_graph.edge_index.T
    edge_weights = pyg_graph.edge_weight.reshape(-1, 1)
    bunch = torch.concat((list_of_edges, edge_weights), dim=1)

    bunch = bunch.tolist()
    bunch = [(int(round(e[0])), int(round(e[1])), e[2]) for e in bunch]

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from([i for i in range(0, num_nodes)])
    nx_graph.add_weighted_edges_from(bunch)
    return nx_graph


def load_graph_level(path_levels: Path, level: int, device: str = 'cpu', randomize_clusters: bool = False):
    edge_index = torch.load(path_levels / f"edge_index_lvl{level}.pt")
    edge_weight = torch.load(path_levels / f"edge_weight_lvl{level}.pt")
    parents = torch.load(path_levels / f"parents_lvl{level}.pt")
    if randomize_clusters:
        torch.manual_seed(42)
        parents = parents[torch.randperm(parents.size()[0])]

    return edge_index, edge_weight, parents


def load_graph_levels(
    path_levels: Path, n_levels: int, device: str='cpu', randomize_clusters: bool = False
) -> List[Data]:
    """Load the graphs corresponding to its coarsening levels. """
    path_levels = Path(path_levels)

    graph_levels = []
    for level in range(n_levels):
        edge_index, edge_weight, parents = load_graph_level(path_levels=path_levels, level=level, device=device, randomize_clusters=randomize_clusters)
        
        g = Data(edge_index=edge_index, edge_weight=edge_weight)
        g.num_nodes = len(parents)
        g.cluster_indices = parents
        graph_levels.append(g)

    return graph_levels


def load_coarse_edges_for_diffpool(
    path_levels: Path,
    n_levels: int,
    device: str = 'cpu'
):
    """Load pre-computed coarse edge_index/edge_weight/parents for each level.

    Used by DiffPool's hybrid early-mode levels that keep fixed coarse edges
    and use HEM parent mappings for efficient scatter pooling.

    Returns
    -------
    coarse_edges : list of (edge_index, edge_weight) tuples
        One per level, edges for the graph at that level.
    parents_list : list of torch.Tensor
        One per level, parent mapping from level i to level i+1.
    """
    path_levels = Path(path_levels)
    coarse_edges = []
    parents_list = []
    for level in range(n_levels):
        edge_index = torch.load(path_levels / f"edge_index_lvl{level}.pt")
        edge_weight = torch.load(path_levels / f"edge_weight_lvl{level}.pt")
        parents = torch.load(path_levels / f"parents_lvl{level}.pt")
        coarse_edges.append((edge_index.to(device), edge_weight.to(device)))
        parents_list.append(parents.to(device))
    return coarse_edges, parents_list
