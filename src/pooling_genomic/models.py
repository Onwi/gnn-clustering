from functools import partial
import math
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
        # The base topology (edge_index/edge_weight) is identical for every sample in
        # the batch -- only S varies per sample -- so `A @ S` never needs a dense
        # (n, n) copy of A, let alone one expanded per batch item. We compute `A @ S`
        # via a single batched sparse-dense matmul rather than one `torch.sparse.mm`
        # call per sample: torch.sparse.mm only accepts a 2D dense operand, so the
        # batch dim is folded into the column dim (n, batch*k) and split back out
        # afterwards. Besides being one op instead of `batch_size`, this matters for
        # autograd: differentiating `batch_size` separate sparse.mm calls each pays
        # its own sparse-transpose/coalesce cost in the backward pass, which is the
        # dominant cost of DMoNLayer's structurally-identical loop (its modularity
        # loss, unlike this layer's link_pred_loss, actually depends on this A @ S
        # branch, so that backward cost is real there -- see DMoNLayer.forward).
        A_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, size=(n, n)).coalesce()
        S_flat = S.permute(1, 0, 2).reshape(n, batch_size * k)
        AS = torch.sparse.mm(A_sparse, S_flat).reshape(n, batch_size, k).permute(1, 0, 2)

        A_next_dense = torch.bmm(S.transpose(1, 2), AS)             # (batch, k, k)
        A_mean = A_next_dense.mean(dim=0)
        A_mean = A_mean * (1 - torch.eye(k, device=A_mean.device))
        edge_index_next, edge_weight_next = dense_to_sparse(A_mean)

        return x_next, edge_index_next, edge_weight_next, aux


class DMoNLayer(nn.Module):
    """A single Deep Modularity Network (DMoN) pooling layer.

    Learns a soft cluster assignment C via a GNN, same shape/role as
    DiffPool's S, and pools features as X' = C^T Z. Unlike DiffPool, the
    auxiliary objective is unsupervised graph clustering quality instead of
    adjacency reconstruction: a modularity loss (rewards intra-cluster edge
    density that exceeds the configuration-model null expectation) plus a
    collapse regularization term (penalizes uneven cluster sizes, which
    prevents the degenerate solution of assigning every node to one cluster).

    Reference: Tsitsulin, Palowitch, Perozzi, Muller, "Graph Clustering with
    Graph Neural Networks", JMLR 2023 (https://arxiv.org/abs/2006.16904).

    Two modes, identical to DiffPoolLayer:
      - hybrid: keeps pre-computed coarse edges for the next level
                (identity message passing, learned clustering only)
      - full:   pools adjacency via C^T A C and extracts sparse edges back

    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    max_clusters : int
        Upper bound on the number of clusters this layer can produce.
    K : int
        Chebyshev filter order.
    collapse_regularization : float
        Weight of the collapse-regularization term relative to the
        modularity term (the paper's single DMoN hyperparameter, typically
        ~1.0). This is folded into the returned 'collapse_loss' so an outer
        lambda_collapse only needs to scale the whole term against the
        classification loss.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        max_clusters: int,
        K: int = 2,
        collapse_regularization: float = 1.0,
    ):
        super().__init__()
        self.embed_gnn = ChebConv(in_channels, hidden_channels, K=K)
        self.pool_gnn = ChebConv(in_channels, max_clusters, K=K)
        self.logit_pool_ratio = nn.Parameter(torch.tensor(0.0))
        self.collapse_regularization = collapse_regularization
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
        aux : dict with keys 'modularity_loss', 'collapse_loss'
        """
        batch_size, n, _ = x.shape

        # Embed features with message passing
        z = F.relu(self.embed_gnn(x, edge_index, edge_weight=edge_weight))

        # --- Hybrid mode with pre-computed HEM parents ---
        if self._parents is not None:
            # Efficient scatter-based pooling (no learned assignments)
            x_next = scatter(z, self._parents, dim=1, reduce='mean')
            aux = {'modularity_loss': 0.0, 'collapse_loss': 0.0}
            return x_next, self._coarse_edge_index, self._coarse_edge_weight, aux

        # --- Full mode: learn soft cluster assignments via pool_gnn ---
        ratio = self.pool_ratio
        k_raw = int(torch.ceil(torch.tensor(n, dtype=torch.float) * ratio).item())
        k = max(min_nodes, min(k_raw, self.pool_gnn.out_channels))

        c_raw = self.pool_gnn(x, edge_index, edge_weight=edge_weight)
        C = F.softmax(c_raw[:, :, :k], dim=-1)

        # Pool features: X' = C^T Z
        x_next = torch.bmm(C.transpose(1, 2), z)

        # --- pooled adjacency C^T A C, computed per-sample via sparse-dense
        # matmul (same rationale as DiffPoolLayer: avoid materializing a
        # dense (n, n) copy of A, which is identical across the batch) ---
        # Batched sparse-dense matmul (one torch.sparse.mm call instead of a
        # `batch_size`-iteration Python loop): unlike DiffPoolLayer, this
        # layer's modularity loss genuinely depends on A_next_dense/degree
        # below, so autograd must differentiate through this op. A loop of
        # `batch_size` separate sparse.mm calls each pays its own sparse
        # transpose/coalesce cost in backward (measured: ~3.2x slower per
        # layer than DiffPoolLayer on the real 3534-node/3.7M-edge level-2
        # graph, dominated by `batch_size` redundant coalesce calls);
        # batching into one call cuts that down to a single coalesce.
        A_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, size=(n, n)).coalesce()
        C_flat = C.permute(1, 0, 2).reshape(n, batch_size * k)
        AC = torch.sparse.mm(A_sparse, C_flat).reshape(n, batch_size, k).permute(1, 0, 2)
        A_next_dense = torch.bmm(C.transpose(1, 2), AC)  # (batch, k, k) = C^T A C

        # --- modularity loss ---
        # Q = (1/2m) * [Tr(C^T A C) - (1/2m) * ||C^T d||^2], where d is the
        # (weighted) degree vector and m is the total edge weight. The base
        # graph is shared across the batch, so d and m are computed once from
        # the sparse adjacency rather than per-sample. edge_weight is assumed
        # to list both directions of each undirected edge (as elsewhere in
        # this codebase), hence the /2 when turning summed weight into m.
        degree = torch.sparse.mm(A_sparse, torch.ones(n, 1, device=x.device, dtype=edge_weight.dtype)).squeeze(-1)
        m = edge_weight.sum() / 2

        trace_CAC = torch.diagonal(A_next_dense, dim1=-2, dim2=-1).sum(dim=-1)  # (batch,)
        Cd = torch.einsum('bnk,n->bk', C, degree)  # (batch, k) = C^T d
        deg_term = (Cd ** 2).sum(dim=-1)  # (batch,)
        modularity = trace_CAC / (2 * m) - deg_term / (2 * m) ** 2
        modularity_loss = -modularity.mean()

        # --- collapse regularization ---
        # L_c = (sqrt(k)/n) * ||sum_nodes C||_2 - 1: 0 when cluster sizes are
        # perfectly balanced (n/k nodes each), sqrt(k)-1 in the degenerate
        # case where every node is assigned to a single cluster.
        cluster_sizes = C.sum(dim=1)  # (batch, k)
        collapse_loss = (
            (torch.sqrt(torch.tensor(float(k), device=x.device)) / n) * cluster_sizes.norm(dim=-1) - 1
        ).mean()

        aux = {
            'modularity_loss': modularity_loss,
            'collapse_loss': self.collapse_regularization * collapse_loss,
        }

        # --- output graph (full mode: extract sparse edges from pooled adjacency) ---
        A_mean = A_next_dense.mean(dim=0)
        A_mean = A_mean * (1 - torch.eye(k, device=A_mean.device))
        edge_index_next, edge_weight_next = dense_to_sparse(A_mean)

        return x_next, edge_index_next, edge_weight_next, aux


def _compute_channel_list(n_levels: int, max_filters: int = 32, start_channels: int = 1):
    """Progressive channel sizes: start_channels, 2*start_channels, ..., max_filters."""
    channels = []
    for i in range(n_levels):
        in_ch = min(start_channels * 2 ** i, max_filters)
        out_ch = min(start_channels * 2 ** (i + 1), max_filters)
        channels.append((in_ch, out_ch))
    return channels


class PrePoolingEncoder(nn.Module):
    """1D-Conv -> ChebConv encoder that enriches raw per-node scalar features
    to ``encoder_channels`` dims before cluster-assignment learning.

    Without this, Full DiffPool's first pooling layer has to learn cluster
    assignments directly from a 1-dim raw expression value per node, which
    plan.md's own ablation found collapses to near-random performance
    (27.7%); adding this encoder alone recovers to 58.3% (5.3.3).
    """
    def __init__(self, encoder_channels: int = 16, encoder_layers: int = 2, K: int = 2):
        super().__init__()
        if encoder_layers < 2:
            raise ValueError("encoder_layers must be >= 2 (1 Conv1d layer + >= 1 ChebConv layer)")
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=encoder_channels, kernel_size=1)
        self.cheb_layers = nn.ModuleList([
            ChebConv(encoder_channels, encoder_channels, K=K)
            for _ in range(encoder_layers - 1)
        ])

    def forward(self, x, edge_index, edge_weight):
        # x: (batch, n_nodes, 1)
        h = x.transpose(1, 2)          # (batch, 1, n_nodes)
        h = F.relu(self.conv1d(h))     # (batch, encoder_channels, n_nodes)
        h = h.transpose(1, 2)          # (batch, n_nodes, encoder_channels)
        for cheb in self.cheb_layers:
            h = F.relu(cheb(h, edge_index, edge_weight=edge_weight))
        return h


def _compute_cluster_schedule(n_start: int, n_final: int, levels: int) -> List[int]:
    """Geometrically-spaced per-level cluster-count targets.

    Without this, every ``DiffPoolLayer`` in full mode caps its output at the
    same global ``max_clusters`` (since ``pool_gnn.out_channels`` bounds
    ``k``), so the very first layer collapses ``n_start`` nodes straight down
    to ``max_clusters`` in one hop regardless of how many levels are
    configured. This spreads that compression geometrically across levels,
    e.g. 14000 -> 670 -> 32 instead of 14000 -> 32 -> 32.

    Returns a list of length ``levels``, strictly decreasing, ending exactly
    at ``n_final``.
    """
    if levels <= 0:
        return []
    if levels == 1 or n_start <= n_final:
        return [n_final] * levels

    log_start, log_final = math.log(n_start), math.log(n_final)
    schedule = []
    prev = n_start
    for i in range(1, levels + 1):
        frac = i / levels
        target = round(math.exp(log_start + (log_final - log_start) * frac))
        # keep strictly decreasing and never below the final target
        target = max(n_final, min(target, prev - 1))
        schedule.append(target)
        prev = target
    schedule[-1] = n_final
    return schedule


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
      The number of layers is given by ``n_levels``.  When ``n_nodes`` is
      given and there is more than one level, each layer's cluster budget is
      spread geometrically from ``n_nodes`` down to ``max_clusters`` (see
      ``_compute_cluster_schedule``) instead of every layer collapsing to
      ``max_clusters`` in a single hop.

    ``pooling_type`` selects the assignment-learning mechanism used by every
    layer's full-mode (learned) branch: ``'diffpool'`` (default, link
    prediction + entropy losses) or ``'dmon'`` (Deep Modularity Networks,
    modularity + collapse-regularization losses -- see ``DMoNLayer``).
    Hybrid-mode levels behave identically either way, since they never reach
    the learned branch.
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
        n_nodes: Optional[int] = None,
        encoder_channels: int = 16,
        encoder_layers: int = 2,
        pooling_type: str = 'diffpool',
        collapse_regularization: float = 1.0,
    ):
        super().__init__()
        self.max_filters = max_filters

        if pooling_type not in ('diffpool', 'dmon'):
            raise ValueError(f"pooling_type must be 'diffpool' or 'dmon', got {pooling_type!r}")
        self.pooling_type = pooling_type

        self.register_buffer('base_edge_index', base_edge_index)
        self.register_buffer('base_edge_weight', base_edge_weight)

        if full_mode:
            levels = n_levels if n_levels is not None else 3
        else:
            levels = n_hybrid + 1

        if levels < 1:
            raise ValueError(
                f"DiffPoolGNN needs at least 1 pooling level, got levels={levels} "
                f"(full_mode={full_mode}, n_levels={n_levels}, n_hybrid={n_hybrid}). "
                "There is no valid zero-pooling architecture -- use n_hybrid/n_levels >= 1."
            )

        # Full mode learns cluster assignments from scratch with no structural
        # prior, which plan.md found collapses to near-random performance
        # unless raw 1-dim node features are first enriched by this encoder
        # (5.3.3: 27.7% -> 58.3% from the encoder alone). Hybrid mode's early
        # levels already have a structural prior (HEM coarse edges), so it
        # keeps starting from the raw 1-dim feature.
        self.encoder = PrePoolingEncoder(encoder_channels, encoder_layers, K) if full_mode else None
        start_channels = encoder_channels if full_mode else 1

        channels = _compute_channel_list(levels, max_filters, start_channels=start_channels)

        if full_mode and levels > 1 and n_nodes is not None:
            cluster_schedule = _compute_cluster_schedule(n_nodes, max_clusters, levels)
        else:
            cluster_schedule = [max_clusters] * levels
        self.cluster_schedule = cluster_schedule

        if parents_list is None:
            parents_list = []

        LayerClass = DMoNLayer if pooling_type == 'dmon' else DiffPoolLayer
        layer_extra_kwargs = {'collapse_regularization': collapse_regularization} if pooling_type == 'dmon' else {}

        self.diffpool_layers = nn.ModuleList()
        for i in range(levels):
            in_ch, out_ch = channels[i]
            layer = LayerClass(
                in_channels=in_ch,
                hidden_channels=out_ch,
                max_clusters=cluster_schedule[i],
                K=K,
                **layer_extra_kwargs,
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

        if self.encoder is not None:
            H = self.encoder(H, edge_index, edge_weight)

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
    encoder_channels: int = 16,
    encoder_layers: int = 2,
    pooling_type: str = 'diffpool',
    collapse_regularization: float = 1.0,
    **kwargs,
):
    """Build a DiffPool- or DMoN-based hierarchical pooling classifier.

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
        Number of DiffPool layers when ``full_mode=True``.  When there is
        more than one level, per-layer cluster budgets are spread
        geometrically from ``base_graph.num_nodes`` down to ``max_clusters``
        instead of every layer collapsing to ``max_clusters`` in one hop
        (see ``DiffPoolGNN`` / ``_compute_cluster_schedule``).
    pooling_type : str
        ``'diffpool'`` (default) or ``'dmon'`` -- see ``DiffPoolGNN``.
    collapse_regularization : float
        DMoN-only: weight of the collapse term relative to modularity
        within each layer (ignored when ``pooling_type='diffpool'``).
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
        n_nodes=base_graph.num_nodes,
        encoder_channels=encoder_channels,
        encoder_layers=encoder_layers,
        pooling_type=pooling_type,
        collapse_regularization=collapse_regularization,
    )

    if full_mode:
        levels = n_levels if n_levels is not None else 3
    else:
        levels = n_hybrid + 1

    # Last layer always pools to max_clusters nodes → flatten. Channel
    # doubling starts from encoder_channels in full mode (the pre-pooling
    # encoder's output width) instead of the raw 1-dim input.
    start_channels = encoder_channels if full_mode else 1
    last_channels = min(start_channels * 2 ** levels, max_filters)
    mlp_input_dim = max_clusters * last_channels

    mlp_model = FCModel(
        input_dim=mlp_input_dim,
        output_dim=output_dims,
        hidden_dim=mlp_hidden_dim,
        dropout=mlp_dropout,
    )

    clf = nn.Sequential(gnn_model, mlp_model)
    return clf


def get_diffpool_aux_losses(
    model: nn.Module,
    lambda_link_pred: float = 0.0,
    lambda_entropy: float = 0.0,
    lambda_modularity: float = 0.0,
    lambda_collapse: float = 0.0,
):
    """Extract and sum weighted auxiliary losses from a DiffPoolGNN inside a Sequential.

    Handles both DiffPool's aux keys ('link_pred_loss', 'entropy_loss') and
    DMoN's ('modularity_loss', 'collapse_loss'): whichever pooling_type the
    model was built with populates only its own pair of keys in each aux
    record, so the other pair's weight simply multiplies a missing-key
    default of 0.0 and contributes nothing.
    """
    weights = {
        'link_pred_loss': lambda_link_pred,
        'entropy_loss': lambda_entropy,
        'modularity_loss': lambda_modularity,
        'collapse_loss': lambda_collapse,
    }
    for module in model.modules():
        if isinstance(module, DiffPoolGNN) and hasattr(module, '_aux_records'):
            records = module._aux_records
            total = 0.0
            for key, weight in weights.items():
                if weight == 0:
                    continue
                total = total + weight * sum(r.get(key, 0.0) for r in records)
            return total
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

