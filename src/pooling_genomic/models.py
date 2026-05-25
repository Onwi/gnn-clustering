from functools import partial
from typing import Union, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import ChebConv
from torch_geometric.utils import from_networkx, to_dense_adj, dense_to_sparse
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_scatter import scatter


def build_gnn_pooling_classifier(
    graphs: List,
    gnns: List,
    mlp_input_dim: int,
    mlp_output_dim: int,
    weighted_pooling: bool = False,
    save_embedding_grad: bool = False,
    mlp_hidden_dim: Union[int, Tuple[int, ...]] = (256, ),
    mlp_dropout: float = 0.5,
    device='cpu',
    **kwargs
):
    # in this case, return the fully connected network
    if len(gnns) == 0 and len(graphs) == 0 and weighted_pooling == False:
        mlp_model = FCModel(
            input_dim=mlp_input_dim,
            output_dim=mlp_output_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout
        )
        return mlp_model
    
    assert len(gnns) > 0
    assert len(graphs) > 0

    gnn_model = GNNPooling(
        gnn=gnns,
        graph=graphs,
        device=device,
        weighted_pooling=weighted_pooling,
        flatten_outputs=True,
        save_embedding_grad=save_embedding_grad
    )

    mlp_model = FCModel(
        input_dim=mlp_input_dim,
        output_dim=mlp_output_dim,
        hidden_dim=mlp_hidden_dim,
        dropout=mlp_dropout
    )

    clf = nn.Sequential(
        gnn_model,
        mlp_model
    )
    
    return clf


class GNNPooling(torch.nn.Module):
    def __init__(
        self,
        gnn,
        graph: Union[Data, List[Data]],
        device="cpu",
        weighted_pooling=False,
        save_embedding_grad=False,
        flatten_outputs: bool = True,
    ):
        super(GNNPooling, self).__init__()
        if (not isinstance(graph, Data)) and (not isinstance(graph, List)):
            raise ValueError(
                "`graph` should be a pytorch geometric Data object or a list of Data objects"
            )
        print("GNN POOLING INIT")
        self.flatten_outputs = flatten_outputs

        if not isinstance(graph, list):
            graph = [graph]

        self.relus = nn.ModuleList()
        for g in graph:
            self.relus.append(nn.ReLU())

        self.weighted_pooling = weighted_pooling
        if self.weighted_pooling:
            self.node_importances = nn.ParameterList()
            for g in graph:
                self.node_importances.append(nn.Parameter(torch.randn(g.num_nodes)))

        self.save_embedding_grad = save_embedding_grad
        if self.save_embedding_grad:
            self.Hs = []
            self.Hs_grad = []
            for g in graph:
                self.Hs.append(None)
                self.Hs_grad.append(None)

        if not isinstance(gnn, list):
            gnn = [gnn]
        
        self.gnn = nn.ModuleList()
        for conv in gnn:
            self.gnn.append(conv)

        assert len(self.gnn) == len(
            graph
        ), f"Number of graphs {len(graph)} should be equal to the number of GCNs {len(self.gnn)}"

        for g in graph:
            assert hasattr(
                g, "cluster_indices"
            ), "All graphs must have a `cluster_indices` attribute"
        
        print("Registering graph as buffer")
        for i, g in enumerate(graph):
            self.register_buffer(f'edge_index_lvl{i}', g.edge_index)
            self.register_buffer(f'edge_weight_lvl{i}', g.edge_weight)
            self.register_buffer(f'cluster_indices_lvl{i}', g.cluster_indices)

    def save_grad(self, grad, level):
        self.Hs_grad[level] = grad.clone().detach()
        return grad

    def forward_cluster_pool(self, X):
        H = X
        for lvl, gnn in enumerate(self.gnn):
            # edge_weight = g.edge_weight if hasattr(g, "edge_weight") else None
            if gnn is not None:
                H: torch.Tensor = gnn(
                    H, self.state_dict()[f'edge_index_lvl{lvl}'], 
                    edge_weight=self.state_dict()[f'edge_weight_lvl{lvl}']
                )
            
            if self.weighted_pooling:
                H = torch.mul(H, self.node_importances[lvl].view(-1, 1))

            H = self.relus[lvl](H)

            H = scatter(src=H, index=self.state_dict()[f'cluster_indices_lvl{lvl}'], dim=-2, reduce="sum")

            if self.save_embedding_grad:
                H.register_hook(partial(self.save_grad, level=lvl))
                self.Hs[lvl] = H.data.clone().detach()

        return H

    def forward(self, X):
        num_samples = X.shape[0]
        num_features = X.shape[1]
        X = torch.reshape(X, (num_samples, num_features, 1))

        H = self.forward_cluster_pool(X)
        if self.flatten_outputs:
            num_nodes, num_embedding_dims = H.shape[-2], H.shape[-1]
            emb_cat = torch.reshape(H, (-1, num_nodes * num_embedding_dims))
            return emb_cat
        return H


class FCModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Union[int, Tuple[int, ...]] = (),
        dropout: float = 0.5,
    ):
        """Construct a general fully-connected network where each layer other than the last
        is composed of a linear transformation, batch normalization, ReLU and dropout.

        The last layer contains only the linear transformation.

        Parameters
        ----------
        input_dim : int
            Number of dimensions of the input
        output_dim : int
            Number of dimensions of the output.
        hidden_dim : Union[int, Tuple[int, ...]], optional
            Number of dimensions in each hidden layers, by default ()
        dropout : float, optional
            Dropout probability, by default 0.5
        """
        super(FCModel, self).__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = (hidden_dim,)

        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.dropout = dropout

        if len(hidden_dim) == 0:
            # build 1 layer FC
            self.fcs.append(nn.Linear(in_features=input_dim, out_features=output_dim))
        else:
            # first layer
            self.fcs.append(
                nn.Linear(in_features=input_dim, out_features=hidden_dim[0])
            )
            self.bns.append(nn.BatchNorm1d(num_features=hidden_dim[0]))
            self.relus.append(nn.ReLU())

            # hidden layers
            for l in range(0, len(hidden_dim) - 1):
                self.fcs.append(
                    nn.Linear(in_features=hidden_dim[l], out_features=hidden_dim[l + 1])
                )
                self.bns.append(nn.BatchNorm1d(num_features=hidden_dim[l + 1]))
                self.relus.append(nn.ReLU())

            # last layer
            self.fcs.append(
                nn.Linear(in_features=hidden_dim[-1], out_features=output_dim)
            )

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        num_layers = len(self.fcs)
        x = x.float()

        for l in range(num_layers - 1):
            x = self.fcs[l](x)
            x = self.bns[l](x)
            x = self.relus[l](x)
            x = F.dropout(x, training=self.training)

        x = self.fcs[-1](x)
        return x


def get_fixed_supernodes_convs_list(
    max_levels: int,
    first_level: int,
    max_filters = 32,
    K = 2
):
    convs = []
    out_channels = 1

    print(f"No. of coarsening levels: {max_levels}")
    # base is no convs
    for i in range(max_levels):
        convs.append(None)

    # add convs starting at the first level
    for i in range(first_level, max_levels):
        j = i - first_level
        out_channels = min(2 ** (j + 1), max_filters)
        in_channels = min(2**j, max_filters)
        conv = ChebConv(in_channels=in_channels, out_channels=out_channels, K=K)
        convs[i] = conv

    return convs, out_channels


def build_fixed_supernodes_coarsening_model(
    first_level: int,
    graphs: List,
    output_dims: int,
    weighted_pooling = False,
    save_embedding_grad: bool = False,
    device: str = "cpu",
    **kwargs,
):
    mlp_hidden_dim = (256,)
    max_filters = 32
    K = 2
    max_levels = len(graphs)
    print(f"No. of coarsening levels: {max_levels}")

    # convs = []
    # out_channels = 1
    # # base is no convs
    # for i in range(max_levels):
    #     convs.append(None)

    # # add convs starting at the first level
    # for i in range(first_level, max_levels):
    #     j = i - first_level
    #     out_channels = min(2 ** (j + 1), max_filters)
    #     in_channels = min(2**j, max_filters)
    #     conv = ChebConv(in_channels=in_channels, out_channels=out_channels, K=K)
    #     convs[i] = conv

    convs, out_channels = get_fixed_supernodes_convs_list(
        max_levels=max_levels,
        first_level=first_level,
        max_filters=max_filters,
        K=K
    )
    num_super_nodes = np.unique(graphs[-1].cluster_indices.cpu()).shape[0]

    model = build_gnn_pooling_classifier(
        graphs=graphs,
        gnns=convs,
        mlp_input_dim=num_super_nodes * out_channels,
        mlp_output_dim=output_dims,
        weighted_pooling=weighted_pooling,
        save_embedding_grad=save_embedding_grad,
        mlp_hidden_dim=mlp_hidden_dim,
        device=device,
        **kwargs,
    )

    return model


def get_coarsening_convs_list(
    n_levels: int,
    use_convs: bool = True,
    max_filters = 32,
    K = 2
):
    convs = []
    out_channels = 1
    for i in range(n_levels):
        if use_convs:
            out_channels = min(2 ** (i + 1), max_filters)
            in_channels = min(2**i, max_filters)
            conv = ChebConv(in_channels=in_channels, out_channels=out_channels, K=K)
        else:
            conv = None
        convs.append(conv)

    return convs, out_channels


def build_coarsening_model(
    n_levels: int,
    graphs: List,
    output_dims: int,
    use_convs: bool = True,
    weighted_pooling = False,
    save_embedding_grad: bool = False,
    device: str = "cpu",
    **kwargs,
):
    mlp_hidden_dim = (256,)
    max_filters = 32
    K = 2

    if n_levels == 0:
        assert weighted_pooling == False, "If n_levels == 0, then weighted_pooling must be False."
        assert save_embedding_grad == False, "If n_levels == 0, then save_embedding_grad must be False."

        # no coarsening levels means we use just the fully connected network
        model = build_gnn_pooling_classifier(
            graphs=[],
            gnns=[],
            mlp_input_dim=graphs[0].num_nodes,
            mlp_output_dim=output_dims,
            weighted_pooling=False,
            save_embedding_grad=False
        )
        return model
    else:
        convs, out_channels = get_coarsening_convs_list(
            n_levels=n_levels,
            use_convs=use_convs,
            max_filters=max_filters,
            K=K
        )

        num_super_nodes = np.unique(graphs[-1].cluster_indices.cpu()).shape[0]

        model = build_gnn_pooling_classifier(
            graphs=graphs,
            gnns=convs,
            mlp_input_dim=num_super_nodes * out_channels,
            mlp_output_dim=output_dims,
            weighted_pooling=weighted_pooling,
            save_embedding_grad=save_embedding_grad,
            mlp_hidden_dim=mlp_hidden_dim,
            device=device,
            **kwargs,
        )
        return model


class DiffPoolLayer(nn.Module):
    """A single differentiable pooling layer (DiffPool-style).

    Learns a soft assignment matrix S that clusters N nodes into K super-nodes.
    Two modes:
      - hybrid:  keeps pre-computed coarse edges for the next level
                 (identity message passing, learned clustering only)
      - full:    pools adjacency via S^T A S and extracts sparse edges back

    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    max_clusters : int
        Upper bound on the number of clusters this layer can produce.
    K : int
        Chebyshev filter order.
    """
    def __init__(self, in_channels: int, hidden_channels: int, max_clusters: int, K: int = 2):
        super().__init__()
        self.embed_gnn = ChebConv(in_channels, hidden_channels, K=K)
        self.pool_gnn = ChebConv(in_channels, max_clusters, K=K)
        self.logit_pool_ratio = nn.Parameter(torch.tensor(0.0))
        self._coarse_edge_index: Optional[torch.Tensor] = None
        self._coarse_edge_weight: Optional[torch.Tensor] = None
        self._parents: Optional[torch.Tensor] = None

    def set_coarse_edges(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, parents: Optional[torch.Tensor] = None):
        self._coarse_edge_index = edge_index
        self._coarse_edge_weight = edge_weight
        self._parents = parents

    @property
    def pool_ratio(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_pool_ratio)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        min_nodes: int = 2,
    ):
        """Forward pass.

        Returns
        -------
        x_next : (batch, k, hidden_channels)
        edge_index_next : (2, e)
        edge_weight_next : (e,)
        aux : dict with keys 'link_pred_loss', 'entropy_loss'
        """
        batch_size, n, _ = x.shape

        # Embed features with message passing
        z = F.relu(self.embed_gnn(x, edge_index, edge_weight=edge_weight))

        # --- Hybrid mode with pre-computed HEM parents ---
        if self._parents is not None:
            # Efficient scatter-based pooling (no learned assignments)
            x_next = scatter(z, self._parents, dim=1, reduce='mean')
            aux = {'link_pred_loss': 0.0, 'entropy_loss': 0.0}
            return x_next, self._coarse_edge_index, self._coarse_edge_weight, aux

        # --- Full mode: learn soft assignments via pool_gnn ---
        ratio = self.pool_ratio
        k_raw = int(torch.ceil(torch.tensor(n, dtype=torch.float) * ratio).item())
        k = max(min_nodes, min(k_raw, self.pool_gnn.out_channels))

        s_raw = self.pool_gnn(x, edge_index, edge_weight=edge_weight)
        S = F.softmax(s_raw[:, :, :k], dim=-1)

        # Pool features: X' = S^T Z
        x_next = torch.bmm(S.transpose(1, 2), z)

        # --- auxiliary losses ---
        A_dense = to_dense_adj(edge_index, edge_attr=edge_weight)  # (1, n, n)
        A_dense = A_dense.squeeze(0)

        S_mean = S.mean(dim=0)
        SSt = S_mean @ S_mean.t()
        link_pred_loss = F.mse_loss(
            A_dense / (A_dense.norm(p='fro') + 1e-8),
            SSt / (SSt.norm(p='fro') + 1e-8),
        )

        S_entropy = -(S * torch.log(S.clamp(min=1e-8))).sum(dim=-1).mean()

        aux = {
            'link_pred_loss': link_pred_loss,
            'entropy_loss': S_entropy,
        }

        # --- output graph (full mode: extract sparse edges from pooled adjacency) ---
        A_next_dense = torch.bmm(
            S.transpose(1, 2),
            torch.bmm(
                A_dense.unsqueeze(0).expand(batch_size, -1, -1),
                S,
            ),
        )
        A_mean = A_next_dense.mean(dim=0)
        A_mean = A_mean * (1 - torch.eye(k, device=A_mean.device))
        edge_index_next, edge_weight_next = dense_to_sparse(A_mean)

        return x_next, edge_index_next, edge_weight_next, aux


def _compute_channel_list(n_levels: int, max_filters: int = 32):
    """Progressive channel sizes matching the original model: 1,2,4,...,max_filters."""
    channels = []
    for i in range(n_levels):
        in_ch = min(2 ** i, max_filters)
        out_ch = min(2 ** (i + 1), max_filters)
        channels.append((in_ch, out_ch))
    return channels


class DiffPoolGNN(nn.Module):
    """Hierarchical pooling GNN with learnable cluster assignments.

    Two modes:

    **Hybrid mode** (``full_mode=False``, default):
      Early layers use pre-computed HEM coarse edges + parent-based scatter
      pooling. Later layers switch to full learned DiffPool when the number
      of nodes drops below ``dense_threshold``.  The number of layers is
      ``n_hybrid + 1``.

    **Full mode** (``full_mode=True``):
      Every layer learns its own soft-assignment matrix ``S`` via
      ``pool_gnn`` and pools features/adjacency as ``X' = S^T Z``,
      ``A' = S^T A S``.  No pre-computed coarse edges are needed.
      The number of layers is given by ``n_levels``.
    """
    def __init__(
        self,
        base_edge_index: torch.Tensor,
        base_edge_weight: torch.Tensor,
        coarse_edges: Optional[List] = None,
        n_hybrid: int = 2,
        parents_list: Optional[List] = None,
        max_filters: int = 32,
        max_clusters: int = 32,
        dense_threshold: int = 500,
        K: int = 2,
        full_mode: bool = False,
        n_levels: Optional[int] = None,
    ):
        super().__init__()
        self.max_filters = max_filters

        self.register_buffer('base_edge_index', base_edge_index)
        self.register_buffer('base_edge_weight', base_edge_weight)

        if full_mode:
            levels = n_levels if n_levels is not None else 3
        else:
            levels = n_hybrid + 1

        channels = _compute_channel_list(levels, max_filters)

        if parents_list is None:
            parents_list = []

        self.diffpool_layers = nn.ModuleList()
        for i in range(levels):
            in_ch, out_ch = channels[i]
            layer = DiffPoolLayer(
                in_channels=in_ch,
                hidden_channels=out_ch,
                max_clusters=max_clusters,
                K=K,
            )
            if not full_mode and i < n_hybrid and coarse_edges is not None and i + 1 < len(coarse_edges):
                ei, ew = coarse_edges[i + 1]
                parents = parents_list[i] if i < len(parents_list) else None
                layer.set_coarse_edges(ei, ew, parents=parents)
            self.diffpool_layers.append(layer)

    def forward(self, X: torch.Tensor):
        num_samples, num_features = X.shape
        H = torch.reshape(X, (num_samples, num_features, 1))

        edge_index = self.base_edge_index
        edge_weight = self.base_edge_weight
        aux_records = []

        for layer in self.diffpool_layers:
            H, edge_index, edge_weight, aux = layer(H, edge_index, edge_weight)
            aux_records.append(aux)

        H = H.reshape(H.size(0), -1)

        self._aux_records = aux_records
        return H


def build_diffpool_model(
    base_graph: Data,
    output_dims: int,
    coarse_edges: Optional[List] = None,
    n_hybrid: int = 2,
    parents_list: Optional[List] = None,
    max_filters: int = 32,
    max_clusters: int = 32,
    dense_threshold: int = 500,
    mlp_hidden_dim: Union[int, Tuple[int, ...]] = (256,),
    mlp_dropout: float = 0.5,
    K: int = 2,
    full_mode: bool = False,
    n_levels: Optional[int] = None,
    **kwargs,
):
    """Build a DiffPool-based classifier.

    Parameters
    ----------
    base_graph : Data
        The original gene graph with ``.edge_index`` and ``.edge_weight``.
    output_dims : int
        Number of output classes.
    coarse_edges : List of (edge_index, edge_weight), optional
        Pre-computed coarse edges for hybrid levels.  Required when
        ``full_mode=False``.
    n_hybrid : int
        Number of early levels that use hybrid mode.  Only used when
        ``full_mode=False``.  In full mode this is ignored in favour of
        ``n_levels``.
    parents_list : List of Tensor, optional
        Pre-computed HEM parent mappings for each hybrid level.
    max_filters : int
        Maximum feature dimension (grows progressively: 1,2,4,...,max_filters).
    max_clusters : int
        Maximum clusters per DiffPoolLayer.  The final layer always pools
        to at most this many nodes.
    dense_threshold : int
        Node count below which hybrid mode switches to full mode
        (only used when ``full_mode=False``).
    mlp_hidden_dim : int or tuple
    mlp_dropout : float
    K : int
        Chebyshev filter order.
    full_mode : bool
        When True, every DiffPoolLayer uses learned assignments
        (no pre-computed coarse edges / HEM parents).
    n_levels : int, optional
        Number of DiffPool layers when ``full_mode=True``.
    """
    gnn_model = DiffPoolGNN(
        base_edge_index=base_graph.edge_index,
        base_edge_weight=base_graph.edge_weight,
        coarse_edges=coarse_edges,
        n_hybrid=n_hybrid,
        parents_list=parents_list,
        max_filters=max_filters,
        max_clusters=max_clusters,
        dense_threshold=dense_threshold,
        K=K,
        full_mode=full_mode,
        n_levels=n_levels,
    )

    if full_mode:
        levels = n_levels if n_levels is not None else 3
    else:
        levels = n_hybrid + 1

    # Last layer always pools to max_clusters nodes → flatten
    last_channels = min(2 ** levels, max_filters)
    mlp_input_dim = max_clusters * last_channels

    mlp_model = FCModel(
        input_dim=mlp_input_dim,
        output_dim=output_dims,
        hidden_dim=mlp_hidden_dim,
        dropout=mlp_dropout,
    )

    clf = nn.Sequential(gnn_model, mlp_model)
    return clf


def get_diffpool_aux_losses(model: nn.Module, lambda_link_pred: float, lambda_entropy: float):
    """Extract and sum auxiliary losses from a DiffPoolGNN inside a Sequential."""
    for module in model.modules():
        if isinstance(module, DiffPoolGNN) and hasattr(module, '_aux_records'):
            records = module._aux_records
            link_loss = sum(r['link_pred_loss'] for r in records)
            ent_loss = sum(r['entropy_loss'] for r in records)
            return lambda_link_pred * link_loss + lambda_entropy * ent_loss
    return 0.0


class CohortAndTumorLoss(nn.Module):
    def __init__(self, cohort_weights: torch.Tensor = None, type_weights: torch.Tensor = None) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=cohort_weights)
        pos_weight = type_weights[1] / type_weights[0]
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, y_pred, t):
        y_cohort, y_type = t[0], t[1]
        y_pred_cohort, y_pred_type = y_pred[0], y_pred[1]

        loss_cohort = self.ce_loss(y_pred_cohort, y_cohort)
        loss_type = self.bce_loss(y_pred_type, y_type)

        return loss_cohort + loss_type


def build_gnn_pooling_tumor_and_cohort_clf(
    graphs: List,
    gnns: List,
    mlp_input_dim: int,
    mlp_cohort_output_dim: int,
    weighted_pooling: bool = False,
    save_embedding_grad: bool = False,
    mlp_hidden_dim: Union[int, Tuple[int, ...]] = (256, ),
    mlp_dropout: float = 0.5,
    device='cpu',
    **kwargs
):
    # in this case, return the fully connected network
    if len(gnns) == 0 and len(graphs) == 0 and weighted_pooling == False:
        mlp_shared = FCModel(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_hidden_dim[0],
            dropout=mlp_dropout
        )

        mlp_cohort_model = FCModel(
            input_dim=mlp_hidden_dim[0],
            output_dim=mlp_cohort_output_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout
        )

        mlp_tumor_model = FCModel(
            input_dim=mlp_hidden_dim[0],
            output_dim=1,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout
        )

        mlp_both = CohortAndTumorClassifier(
            cohort_clf=mlp_cohort_model, 
            tumor_clf=mlp_tumor_model
        )
        clf = nn.Sequential(
            mlp_shared,
            mlp_both
        )

        return clf
    
    assert len(gnns) > 0
    assert len(graphs) > 0

    gnn_model = GNNPooling(
        gnn=gnns,
        graph=graphs,
        device=device,
        weighted_pooling=weighted_pooling,
        flatten_outputs=True,
        save_embedding_grad=save_embedding_grad
    )

    mlp_cohort_model = FCModel(
        input_dim=mlp_input_dim,
        output_dim=mlp_cohort_output_dim,
        hidden_dim=mlp_hidden_dim,
        dropout=mlp_dropout
    )

    mlp_tumor_model = FCModel(
        input_dim=mlp_input_dim,
        output_dim=1,
        hidden_dim=mlp_hidden_dim,
        dropout=mlp_dropout
    )

    mlp_both = CohortAndTumorClassifier(
        cohort_clf=mlp_cohort_model, 
        tumor_clf=mlp_tumor_model
    )

    clf = nn.Sequential(
        gnn_model,
        mlp_both
    )
    
    return clf


class CohortAndTumorClassifier(nn.Module):
    def __init__(self, cohort_clf: nn.Module, tumor_clf: nn.Module) -> None:
        super().__init__()
        self.cohort_clf = cohort_clf
        self.tumor_clf = tumor_clf

    def forward(self, x):
        y_c = self.cohort_clf(x)
        y_t = torch.squeeze(self.tumor_clf(x))
        return y_c, y_t

