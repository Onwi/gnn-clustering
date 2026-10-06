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
from torch_geometric.utils import from_networkx, dense_to_sparse
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


def _compute_pool_k(n: int, min_nodes: int, max_clusters: int) -> int:
    """Number of output clusters for a pooling layer: ``max_clusters``, clamped so
    it's never below ``min_nodes`` nor above the actual input node count ``n``.

    Previously this scaled a layer-local ``pool_ratio`` parameter against ``n``
    (``k_raw = ceil(n * pool_ratio)``, then clamped to ``max_clusters``). That
    parameter was dead: ``torch.ceil`` has zero gradient almost everywhere, so
    ``pool_ratio`` never received a gradient and stayed frozen at its init value
    (0.5) for the life of training -- and since ``k_raw`` at that fixed ratio was
    always far larger than ``max_clusters`` in every configuration this codebase
    actually runs (levels only shrink node counts by less than 2x when
    ``max_clusters`` is the smaller of the two), the clamp to ``max_clusters``
    was binding every time regardless. The output was always exactly
    ``max_clusters`` in practice; this makes that explicit instead of routing it
    through a parameter that looked learnable but wasn't.
    """
    return max(min_nodes, min(max_clusters, n))


def _pool_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    assignment: torch.Tensor,
    n: int,
    batch_size: int,
    k: int,
    compute_degree: bool = False,
):
    """Shared DiffPool/DMoN pooling step: batched sparse `A @ assignment` and the
    resulting dense pooled adjacency `assignment^T @ A @ assignment`.

    The base graph topology is identical for every sample in the batch -- only
    `assignment` (DiffPool's S / DMoN's C) varies per sample -- so this is one
    batched sparse-dense matmul (`torch.sparse.mm` only accepts a 2D dense
    operand, so the batch dim is folded into the column dim and split back out
    afterwards) rather than `batch_size` separate calls.

    If `compute_degree` is set, the (weighted) degree vector is fused into the
    same sparse matmul by appending a ones-column to the assignment matrix,
    instead of paying for a second sparse-dense matmul pass over `A_sparse`.
    """
    A_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, size=(n, n)).coalesce()
    assign_flat = assignment.permute(1, 0, 2).reshape(n, batch_size * k)
    degree = None
    if compute_degree:
        ones_col = torch.ones(n, 1, device=assignment.device, dtype=edge_weight.dtype)
        out = torch.sparse.mm(A_sparse, torch.cat([assign_flat, ones_col], dim=1))
        A_assign = out[:, :-1].reshape(n, batch_size, k).permute(1, 0, 2)
        degree = out[:, -1]
    else:
        A_assign = torch.sparse.mm(A_sparse, assign_flat).reshape(n, batch_size, k).permute(1, 0, 2)
    A_next_dense = torch.bmm(assignment.transpose(1, 2), A_assign)  # (batch, k, k)
    return A_assign, A_next_dense, degree


def _pool_adjacency_batched(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    assignment: torch.Tensor,
    n: int,
    batch_size: int,
    k: int,
    compute_degree: bool = False,
):
    """Per-sample counterpart to ``_pool_adjacency``, for a *block-diagonal*
    input graph: sample ``b``'s ``n`` nodes occupy indices ``[b*n, (b+1)*n)``
    of a single ``(batch_size*n, batch_size*n)`` sparse adjacency with no
    cross-sample edges (see ``_build_pooled_output_graph_batched``, which
    produces exactly this layout -- used from full-mode level 1 onward,
    where each sample's pooled graph genuinely differs from every other
    sample's, so there is no shared graph left to exploit the way
    ``_pool_adjacency`` does for the base graph / HEM levels).

    No permute-and-fold trick is needed here (unlike ``_pool_adjacency``):
    each sample's assignment rows already sit in their own disjoint block of
    the sparse matrix, so a plain reshape lines them up correctly.

    Uses ``torch_scatter.scatter`` (gather-multiply-scatter) instead of
    ``torch.sparse.mm`` -- unlike every existing caller of ``_pool_adjacency``
    (level 0's base graph, HEM levels, a hybrid trailing layer's input),
    which always get a *fixed* edge_weight (a registered buffer / precomputed
    HEM weights, never requiring grad), this function's edge_weight is
    itself the output of a previous full-mode level's
    ``_build_pooled_output_graph_batched`` and genuinely requires grad.
    ``torch.sparse.mm``'s CUDA backward through sparse-tensor *values* that
    require grad reproducibly crashed with an illegal-memory-access a few
    backward calls into real training (every one of 5 random seeds failed
    by iteration 3-4, forward always succeeded, only backward crashed) --
    a known-unstable path in this PyTorch/CUDA version. ``scatter`` is used
    elsewhere in this codebase already (the HEM-parents branch) with no
    such issue.

    If `compute_degree` is set, returns each node's *within-its-own-block*
    degree, reshaped to (batch, n) -- NOT a single shared (n,) vector, since
    blocks can have different total edge weight.
    """
    row, col = edge_index[0], edge_index[1]
    assign_flat = assignment.reshape(batch_size * n, k)
    degree = None
    if compute_degree:
        ones_col = torch.ones(batch_size * n, 1, device=assignment.device, dtype=edge_weight.dtype)
        operand = torch.cat([assign_flat, ones_col], dim=1)  # (batch*n, k+1)
        messages = edge_weight.unsqueeze(-1) * operand[col]
        out = scatter(messages, row, dim=0, dim_size=batch_size * n, reduce='sum')
        A_assign = out[:, :-1].reshape(batch_size, n, k)
        degree = out[:, -1].reshape(batch_size, n)
    else:
        messages = edge_weight.unsqueeze(-1) * assign_flat[col]
        A_assign = scatter(messages, row, dim=0, dim_size=batch_size * n, reduce='sum').reshape(batch_size, n, k)
    A_next_dense = torch.bmm(assignment.transpose(1, 2), A_assign)  # (batch, k, k)
    return A_assign, A_next_dense, degree


def _build_pooled_output_graph_batched(A_next_dense: torch.Tensor, k: int, sparsify_density: Optional[float] = None):
    """Per-sample counterpart to ``_build_pooled_output_graph``: instead of
    averaging the batch's pooled adjacencies into one shared graph, drops
    the diagonal and (optionally) sparsifies EACH sample's own pooled
    adjacency, then assembles them into a single block-diagonal
    ``(edge_index, edge_weight)`` over ``batch_size*k`` nodes (sample b's
    nodes at indices ``[b*k, (b+1)*k)``) -- see changes-from-claude.md #13
    (full-mode DMoN review, finding 1): without this, every full-mode level
    past the first pools on a graph averaged across whatever patients happen
    to share a batch, so a patient's prediction depended on its batch
    companions (measured: up to ~3.6% of test predictions flipped between
    batch sizes 1 and 8 on the pre-fix model). This makes every level's
    graph genuinely that patient's own, for every patient, regardless of
    batch composition or size.

    Returns a flat ``x_next`` is NOT produced here (that is the caller's
    job -- see ``DiffPoolLayer``/``DMoNLayer``.forward): this only builds
    the graph.
    """
    batch_size, k_, _ = A_next_dense.shape
    assert k_ == k
    A = A_next_dense * (1 - torch.eye(k, device=A_next_dense.device))
    if sparsify_density is not None:
        top_k = max(1, round(sparsify_density * (k - 1)))
        if top_k < k - 1:
            _, topk_idx = A.topk(top_k, dim=-1)  # (batch, k, top_k)
            mask = torch.zeros_like(A, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            mask = mask | mask.transpose(-1, -2)
            A = A * mask
    edge_index_parts, edge_weight_parts = [], []
    for b in range(batch_size):
        ei, ew = dense_to_sparse(A[b])
        edge_index_parts.append(ei + b * k)
        edge_weight_parts.append(ew)
    edge_index = torch.cat(edge_index_parts, dim=1) if edge_index_parts else A.new_zeros((2, 0), dtype=torch.long)
    edge_weight = torch.cat(edge_weight_parts, dim=0) if edge_weight_parts else A.new_zeros((0,))
    return edge_index, edge_weight


def _build_pooled_output_graph(A_next_dense: torch.Tensor, k: int, sparsify_density: Optional[float] = None):
    """Mean the batch's pooled adjacency, drop the diagonal (self-loops),
    optionally prune to a target per-row density, and convert to sparse
    (edge_index, edge_weight) -- the shared final step of both
    DiffPoolLayer and DMoNLayer's full-mode forward pass.

    Without ``sparsify_density``, every entry of the softmax-derived pooled
    adjacency is numerically nonzero, so ``dense_to_sparse`` returns a fully
    connected k x k graph every level -- "sparse" only in tensor
    representation, not in structure (see changes-from-claude.md fix #3).

    When given, keeps only the top ``round(sparsify_density * (k - 1))``
    (at least 1) outgoing edges per row, then ORs the mask with its
    transpose so an edge survives if either endpoint ranked it in its own
    top-k (rows aren't in general symmetric even though A_mean is, since
    pruning is per-row). ``sparsify_density`` is a *fraction of this level's
    own width* rather than a fixed edge count, because full-mode levels span
    wildly different widths (e.g. 1854 down to 32 in a 3-level schedule) --
    a fixed count can't match the base PPI graph's actual density (~4%, as
    loaded: 8,165,154 edge entries over the 14,133 genes shared with the
    expression data, i.e. after restricting stringdb_top100pc.csv's 11.9M
    rows / 19,385 proteins) at more than one of them simultaneously.
    """
    A_mean = A_next_dense.mean(dim=0)
    A_mean = A_mean * (1 - torch.eye(k, device=A_mean.device))
    if sparsify_density is not None:
        top_k = max(1, round(sparsify_density * (k - 1)))
        if top_k < k - 1:
            _, topk_idx = A_mean.topk(top_k, dim=-1)
            mask = torch.zeros_like(A_mean, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
            mask = mask | mask.t()
            A_mean = A_mean * mask
    return dense_to_sparse(A_mean)


class PoolAssignmentHead(nn.Module):
    """Chebyshev-conv assignment head producing raw (pre-softmax) cluster
    logits, used as `pool_gnn` by `DiffPoolLayer`/`DMoNLayer`.

    With ``n_layers=1`` (the default, used everywhere so far) this is
    exactly a single ``ChebConv`` -- identical parameters and output to the
    original code, and NOT wrapped in this class (see
    ``_build_pool_assignment_head``), so every existing checkpoint still
    loads unchanged.

    With ``n_layers>1``, stacks ``n_layers - 1`` ReLU-separated
    ``ChebConv(K=K)`` layers of width ``hidden_channels`` before the final
    projection to ``max_clusters``, giving the assignment decision a
    receptive field of ``n_layers`` hops of order ``K-1`` each, instead of
    one. Motivation: `embed_gnn` only has to produce good per-node
    features, but `pool_gnn` has to decide a hard partition of up to
    thousands of nodes in one shot from a `K-1`-hop view -- the same
    "1-hop filter can't see global community structure" limitation
    `plan.md` identified for Full DiffPool generally. See
    changes-from-claude.md #13, finding 1.
    """
    def __init__(self, in_channels: int, max_clusters: int, K: int = 2, n_layers: int = 2,
                 hidden_channels: Optional[int] = None):
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"PoolAssignmentHead needs n_layers >= 2 (use a plain ChebConv for n_layers=1), got {n_layers}")
        hidden_channels = hidden_channels or in_channels
        self.out_channels = max_clusters
        convs = []
        cur = in_channels
        for _ in range(n_layers - 1):
            convs.append(ChebConv(cur, hidden_channels, K=K))
            cur = hidden_channels
        convs.append(ChebConv(cur, max_clusters, K=K))
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        h = x
        for conv in self.convs[:-1]:
            h = F.relu(conv(h, edge_index, edge_weight=edge_weight))
        return self.convs[-1](h, edge_index, edge_weight=edge_weight)


def _build_pool_assignment_head(in_channels: int, max_clusters: int, K: int, n_layers: int):
    """`pool_gnn` factory: a plain `ChebConv` for `n_layers=1` (bit-identical
    to every pre-existing model/checkpoint), a `PoolAssignmentHead` otherwise."""
    if n_layers <= 1:
        return ChebConv(in_channels, max_clusters, K=K)
    return PoolAssignmentHead(in_channels, max_clusters, K=K, n_layers=n_layers)


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
    sparsify_density : float, optional
        Full mode only: prune the pooled output adjacency to this fraction
        of edges per node (see ``_build_pooled_output_graph``) instead of
        leaving it fully connected.
    assign_dropout : float
        Dropout applied to the raw assignment logits before the softmax.
        Tsitsulin et al. (DMoN paper, JMLR 2023) use 0.5 here and report it
        specifically prevents gradient descent from getting stuck in a
        degenerate assignment -- applied to both pooling types (not just
        DMoN) to avoid introducing a new architectural asymmetry between
        them, consistent with keeping everything else about the two
        assignment heads identical.
    pool_gnn_layers : int
        Depth of the assignment head (`pool_gnn`); see `PoolAssignmentHead`.
        Default 1 is a plain single `ChebConv`, identical to the original
        code.
    per_patient_graphs : bool
        Full mode only. True (default): each sample's own pooled graph
        feeds the next level (see `_build_pooled_output_graph_batched`).
        False: the pre-fix batch-averaged graph (`_build_pooled_output_graph`)
        -- costs much less memory (no dense per-edge gather tensor) at the
        price of reintroducing batch-composition dependence; see
        changes-from-claude.md #13/#14.
    """
    def __init__(self, in_channels: int, hidden_channels: int, max_clusters: int, K: int = 2,
                 sparsify_density: Optional[float] = None, assign_dropout: float = 0.5,
                 pool_gnn_layers: int = 1, per_patient_graphs: bool = True):
        super().__init__()
        self.embed_gnn = ChebConv(in_channels, hidden_channels, K=K)
        self.pool_gnn = _build_pool_assignment_head(in_channels, max_clusters, K, pool_gnn_layers)
        self.sparsify_density = sparsify_density
        self.assign_dropout = assign_dropout
        self.per_patient_graphs = per_patient_graphs
        self._coarse_edge_index: Optional[torch.Tensor] = None
        self._coarse_edge_weight: Optional[torch.Tensor] = None
        self._parents: Optional[torch.Tensor] = None

    def set_coarse_edges(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, parents: Optional[torch.Tensor] = None):
        self._coarse_edge_index = edge_index
        self._coarse_edge_weight = edge_weight
        self._parents = parents

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        min_nodes: int = 2,
        batch_size: Optional[int] = None,
        is_last_level: bool = False,
    ):
        """Forward pass.

        `x` is either:
          - (batch, n, in_channels): one graph (`edge_index`/`edge_weight`,
            `n` nodes) shared by the whole batch -- the base PPI graph, a
            HEM level, or a hybrid trailing layer's input.
          - (batch*n, in_channels): block-diagonal, sample b's nodes at
            indices [b*n, (b+1)*n) -- a previous full-mode layer's output
            (see `_build_pooled_output_graph_batched`); `batch_size` must
            be given in this case.

        Returns
        -------
        x_next : (batch, k, hidden_channels) if `is_last_level` (nothing
            downstream needs a graph for it), else (batch*k, hidden_channels)
            -- block-diagonal, matching `edge_index_next`/`edge_weight_next`.
        edge_index_next, edge_weight_next : (None, None) if `is_last_level`,
            else the next level's block-diagonal graph over `batch*k` nodes.
        aux : dict with keys 'link_pred_loss', 'entropy_loss'
        """
        shared_graph = x.dim() == 3
        if shared_graph:
            batch_size, n, _ = x.shape
            z = F.relu(self.embed_gnn(x, edge_index, edge_weight=edge_weight))
        else:
            assert batch_size is not None, "batch_size required for a block-diagonal (2D) input"
            n = x.shape[0] // batch_size
            z = F.relu(self.embed_gnn(x, edge_index, edge_weight=edge_weight)).view(batch_size, n, -1)

        # --- Hybrid mode with pre-computed HEM parents ---
        if self._parents is not None:
            # Efficient scatter-based pooling (no learned assignments). HEM
            # levels always see a shared graph (HEM coarsening is computed
            # once, identically for every patient).
            x_next = scatter(z, self._parents, dim=1, reduce='mean')
            aux = {'link_pred_loss': 0.0, 'entropy_loss': 0.0}
            return x_next, self._coarse_edge_index, self._coarse_edge_weight, aux

        # --- Full mode (or hybrid's trailing layer): learn soft assignments ---
        k = _compute_pool_k(n, min_nodes, self.pool_gnn.out_channels)

        if shared_graph:
            s_raw = self.pool_gnn(x, edge_index, edge_weight=edge_weight)
        else:
            s_raw = self.pool_gnn(x, edge_index, edge_weight=edge_weight).view(batch_size, n, -1)
        s_raw = F.dropout(s_raw, p=self.assign_dropout, training=self.training)
        S = F.softmax(s_raw[:, :, :k], dim=-1)

        # Pool features: X' = S^T Z
        x_next = torch.bmm(S.transpose(1, 2), z)

        # --- pooled adjacency S^T A S: computed once, shared by the
        # link-prediction loss below and the output graph. Shared-graph
        # levels use `_pool_adjacency`'s permute-and-fold trick (one graph
        # for the whole batch); block-diagonal levels (full mode, level >=
        # 1, where every sample's graph genuinely differs -- see
        # changes-from-claude.md #13) use `_pool_adjacency_batched`.
        if shared_graph:
            _, A_next_dense, _ = _pool_adjacency(edge_index, edge_weight, S, n, batch_size, k)
        else:
            _, A_next_dense, _ = _pool_adjacency_batched(edge_index, edge_weight, S, n, batch_size, k)

        # --- auxiliary losses ---
        # Link-prediction loss, per sample then averaged -- NOT the same as
        # averaging S over the batch before computing SS^T (batch samples have
        # different node features/assignments; averaging S first collapses that
        # per-sample structure into one "population-average" assignment before
        # the quadratic term, which is a materially weaker target than a true
        # per-sample reconstruction loss).
        #
        # A per-sample dense (n, n) SS^T is infeasible at this codebase's scale
        # (n up to 14,133 at full-mode level 0), so this uses an algebraic
        # identity to get the exact per-sample MSE(A/||A||, SS^T/||SS^T||)
        # without ever materializing an (n, n) matrix:
        #   <A, SS^T>_F = trace(A S S^T) = trace(S^T A S)         (cyclic trace)
        #               = trace(A_next_dense)                     (already computed above)
        #   ||SS^T||_F  = ||S^T S||_F   since for symmetric M = SS^T,
        #                 ||M||_F^2 = trace(M^2) = trace(S S^T S S^T)
        #                           = trace((S^T S)^2) = ||S^T S||_F^2
        #                 -- S^T S is only (k, k), cheap to form per sample.
        #   ||A||_F     = sqrt(sum(edge_weight^2)), sparse-based, no dense copy
        #                 -- per sample when the graph is block-diagonal (each
        #                 sample's own subgraph has its own edge mass).
        # With X = A/||A||_F and Y = SS^T/||SS^T||_F (so ||X||_F = ||Y||_F = 1):
        #   MSE(X, Y) = (1/n^2) * (2 - 2<X, Y>_F) = (2/n^2) * (1 - <A,SS^T>_F / (||A||_F ||SS^T||_F))
        if shared_graph:
            A_fro = torch.sqrt((edge_weight ** 2).sum() + 1e-8)  # scalar: same graph for every sample
        else:
            edge_sample = torch.div(edge_index[0], n, rounding_mode='trunc')
            A_fro = torch.sqrt(
                scatter(edge_weight ** 2, edge_sample, dim=0, dim_size=batch_size, reduce='sum') + 1e-8
            )  # (batch,): each sample's own subgraph
        trace_ASAS = torch.diagonal(A_next_dense, dim1=-2, dim2=-1).sum(dim=-1)  # (batch,)
        StS = torch.bmm(S.transpose(1, 2), S)  # (batch, k, k)
        SSt_fro = StS.flatten(1).norm(dim=-1) + 1e-8  # (batch,)
        link_pred_loss = ((2.0 / n ** 2) * (1 - trace_ASAS / (A_fro * SSt_fro))).mean()

        S_entropy = -(S * torch.log(S.clamp(min=1e-8))).sum(dim=-1).mean()

        aux = {
            'link_pred_loss': link_pred_loss,
            'entropy_loss': S_entropy,
        }

        # --- output graph: a hybrid trailing layer or full mode's last
        # level has nothing downstream to consume one (`DiffPoolGNN.forward`
        # reshapes x_next straight to the classifier), so skip building it.
        # Otherwise (full mode, not the last level) build either each
        # sample's own pooled graph (flattening x_next to match) or the
        # cheaper batch-averaged one, per `self.per_patient_graphs`.
        if is_last_level:
            edge_index_next, edge_weight_next = None, None
        elif self.per_patient_graphs:
            edge_index_next, edge_weight_next = _build_pooled_output_graph_batched(A_next_dense, k, self.sparsify_density)
            x_next = x_next.reshape(batch_size * k, -1)
        else:
            edge_index_next, edge_weight_next = _build_pooled_output_graph(A_next_dense, k, self.sparsify_density)

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
    sparsify_density : float, optional
        Full mode only: prune the pooled output adjacency to this fraction
        of edges per node (see ``_build_pooled_output_graph``) instead of
        leaving it fully connected.
    assign_dropout : float
        Dropout applied to the raw assignment logits before the softmax.
        Tsitsulin et al. (DMoN paper, JMLR 2023) use 0.5 here and report it
        specifically prevents gradient descent from getting stuck in a
        degenerate assignment -- applied to both pooling types (not just
        DMoN) to avoid introducing a new architectural asymmetry between
        them.

    Note: there used to be a separate ``collapse_regularization`` parameter
    here, a fixed multiplier on ``collapse_loss``. It was removed --
    mathematically redundant with the outer, independently-tuned
    ``lambda_collapse`` (both linearly scale the same term with nothing
    else combining them, so searching over both adds no expressiveness
    beyond what ``lambda_collapse`` alone already covers).

    pool_gnn_layers : int
        Depth of the assignment head (`pool_gnn`); see `PoolAssignmentHead`.
        Default 1 is a plain single `ChebConv`, identical to the original
        code.
    per_patient_graphs : bool
        Full mode only. True (default): each sample's own pooled graph
        feeds the next level (see `_build_pooled_output_graph_batched`).
        False: the pre-fix batch-averaged graph (`_build_pooled_output_graph`)
        -- costs much less memory (no dense per-edge gather tensor) at the
        price of reintroducing batch-composition dependence; see
        changes-from-claude.md #13/#14.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        max_clusters: int,
        K: int = 2,
        sparsify_density: Optional[float] = None,
        assign_dropout: float = 0.5,
        pool_gnn_layers: int = 1,
        per_patient_graphs: bool = True,
    ):
        super().__init__()
        self.embed_gnn = ChebConv(in_channels, hidden_channels, K=K)
        self.pool_gnn = _build_pool_assignment_head(in_channels, max_clusters, K, pool_gnn_layers)
        self.sparsify_density = sparsify_density
        self.assign_dropout = assign_dropout
        self.per_patient_graphs = per_patient_graphs
        self._coarse_edge_index: Optional[torch.Tensor] = None
        self._coarse_edge_weight: Optional[torch.Tensor] = None
        self._parents: Optional[torch.Tensor] = None

    def set_coarse_edges(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, parents: Optional[torch.Tensor] = None):
        self._coarse_edge_index = edge_index
        self._coarse_edge_weight = edge_weight
        self._parents = parents

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        min_nodes: int = 2,
        batch_size: Optional[int] = None,
        is_last_level: bool = False,
    ):
        """Forward pass.

        `x` is either:
          - (batch, n, in_channels): one graph (`edge_index`/`edge_weight`,
            `n` nodes) shared by the whole batch -- the base PPI graph, a
            HEM level, or a hybrid trailing layer's input.
          - (batch*n, in_channels): block-diagonal, sample b's nodes at
            indices [b*n, (b+1)*n) -- a previous full-mode layer's output
            (see `_build_pooled_output_graph_batched`); `batch_size` must
            be given in this case.

        Returns
        -------
        x_next : (batch, k, hidden_channels) if `is_last_level` (nothing
            downstream needs a graph for it), else (batch*k, hidden_channels)
            -- block-diagonal, matching `edge_index_next`/`edge_weight_next`.
        edge_index_next, edge_weight_next : (None, None) if `is_last_level`,
            else the next level's block-diagonal graph over `batch*k` nodes.
        aux : dict with keys 'modularity_loss', 'collapse_loss'
        """
        shared_graph = x.dim() == 3
        if shared_graph:
            batch_size, n, _ = x.shape
            z = F.relu(self.embed_gnn(x, edge_index, edge_weight=edge_weight))
        else:
            assert batch_size is not None, "batch_size required for a block-diagonal (2D) input"
            n = x.shape[0] // batch_size
            z = F.relu(self.embed_gnn(x, edge_index, edge_weight=edge_weight)).view(batch_size, n, -1)

        # --- Hybrid mode with pre-computed HEM parents ---
        if self._parents is not None:
            # Efficient scatter-based pooling (no learned assignments). HEM
            # levels always see a shared graph (HEM coarsening is computed
            # once, identically for every patient).
            x_next = scatter(z, self._parents, dim=1, reduce='mean')
            aux = {'modularity_loss': 0.0, 'collapse_loss': 0.0}
            return x_next, self._coarse_edge_index, self._coarse_edge_weight, aux

        # --- Full mode (or hybrid's trailing layer): learn soft cluster assignments ---
        k = _compute_pool_k(n, min_nodes, self.pool_gnn.out_channels)

        if shared_graph:
            c_raw = self.pool_gnn(x, edge_index, edge_weight=edge_weight)
        else:
            c_raw = self.pool_gnn(x, edge_index, edge_weight=edge_weight).view(batch_size, n, -1)
        c_raw = F.dropout(c_raw, p=self.assign_dropout, training=self.training)
        C = F.softmax(c_raw[:, :, :k], dim=-1)

        # Pool features: X' = C^T Z
        x_next = torch.bmm(C.transpose(1, 2), z)

        # --- pooled adjacency C^T A C: unlike DiffPoolLayer, this layer's
        # modularity loss genuinely depends on A_next_dense/degree below, so
        # autograd must differentiate through this op. Shared-graph levels
        # use `_pool_adjacency`'s batched sparse.mm trick (one graph for the
        # whole batch, ~3.2x faster per layer than a per-sample Python loop
        # on the real 3534-node/3.7M-edge level-2 graph); block-diagonal
        # levels (full mode, level >= 1, where every sample's graph
        # genuinely differs -- see changes-from-claude.md #13) use
        # `_pool_adjacency_batched`. Both fuse the degree vector into the
        # same matmul (`compute_degree=True`).
        if shared_graph:
            _, A_next_dense, degree = _pool_adjacency(
                edge_index, edge_weight, C, n, batch_size, k, compute_degree=True
            )
        else:
            _, A_next_dense, degree = _pool_adjacency_batched(
                edge_index, edge_weight, C, n, batch_size, k, compute_degree=True
            )

        # --- modularity loss ---
        # Q = (1/2m) * [Tr(C^T A C) - (1/2m) * ||C^T d||^2], where d is the
        # (weighted) degree vector and m is the total edge weight. A shared
        # graph is identical for every sample, so d and m are computed once;
        # a block-diagonal graph genuinely differs per sample, so both
        # become per-sample (batch,)/(batch, n) instead of a shared scalar/
        # (n,) vector. edge_weight is assumed to list both directions of
        # each undirected edge (as elsewhere in this codebase), hence the /2
        # when turning summed weight into m.
        # Clamped away from 0: in chained full-mode DMoN layers, edge mass
        # increasingly concentrates on the diagonal of C^T A C as upstream
        # clustering improves -- that diagonal is dropped before the next
        # level's edge_weight is built, so m can shrink toward 0 over
        # training, and it is squared in the second term's denominator below.
        if shared_graph:
            m = torch.clamp(edge_weight.sum() / 2, min=1e-8)  # scalar
            Cd = torch.einsum('bnk,n->bk', C, degree)  # (batch, k) = C^T d
        else:
            edge_sample = torch.div(edge_index[0], n, rounding_mode='trunc')
            m = torch.clamp(
                scatter(edge_weight, edge_sample, dim=0, dim_size=batch_size, reduce='sum') / 2, min=1e-8
            )  # (batch,)
            Cd = torch.einsum('bnk,bn->bk', C, degree)  # (batch, k), degree is (batch, n) here
        deg_term = (Cd ** 2).sum(dim=-1)  # (batch,)
        trace_CAC = torch.diagonal(A_next_dense, dim1=-2, dim2=-1).sum(dim=-1)  # (batch,)
        modularity = trace_CAC / (2 * m) - deg_term / (2 * m) ** 2
        modularity_loss = -modularity.mean()

        # --- collapse regularization ---
        # L_c = (sqrt(k)/n) * ||sum_nodes C||_2 - 1: 0 when cluster sizes are
        # perfectly balanced (n/k nodes each), sqrt(k)-1 in the degenerate
        # case where every node is assigned to a single cluster -- i.e. its
        # *range* grows with k. A chained full-mode schedule has a very
        # different k at every level (e.g. 2809 / 558 / 111 for a 3-level,
        # max_clusters=111 run: sqrt(k)-1 of 52.0 / 22.6 / 9.5), so one
        # shared lambda_collapse effectively regularizes whichever level has
        # the largest k and barely touches the others. Dividing by its own
        # level's range puts every level's term on the same [0, 1] scale, so
        # a single lambda applies comparably everywhere -- see
        # changes-from-claude.md #13, finding 2.
        cluster_sizes = C.sum(dim=1)  # (batch, k)
        collapse_range = max(math.sqrt(k) - 1, 1e-8)
        collapse_loss = (
            ((math.sqrt(k) / n) * cluster_sizes.norm(dim=-1) - 1) / collapse_range
        ).mean()

        aux = {
            'modularity_loss': modularity_loss,
            'collapse_loss': collapse_loss,
        }

        # --- output graph: a hybrid trailing layer or full mode's last
        # level has nothing downstream to consume one (`DiffPoolGNN.forward`
        # reshapes x_next straight to the classifier), so skip building it.
        # Otherwise (full mode, not the last level) build either each
        # sample's own pooled graph (flattening x_next to match) or the
        # cheaper batch-averaged one, per `self.per_patient_graphs`.
        if is_last_level:
            edge_index_next, edge_weight_next = None, None
        elif self.per_patient_graphs:
            edge_index_next, edge_weight_next = _build_pooled_output_graph_batched(A_next_dense, k, self.sparsify_density)
            x_next = x_next.reshape(batch_size * k, -1)
        else:
            edge_index_next, edge_weight_next = _build_pooled_output_graph(A_next_dense, k, self.sparsify_density)

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


LEARNED_POOLING_TYPES = ('diffpool', 'dmon')
# Hybrid-mode controls: the trailing layer pools with a fixed (non-learned)
# assignment instead of a learned one, so that everything else about the
# model and its training protocol is identical to Hybrid DiffPool/DMoN.
#   'hem'      -- one more precomputed HEM level (e.g. 221 -> 111 nodes at
#                 n_hybrid=6); with n_hybrid=0 this is the 1-level Fixed HEM
#                 architecture (14,133 -> 7,067 nodes).
#   'spectral' -- spectral clustering of the trailing layer's input graph
#                 into max_clusters groups: a fixed, structure-aware
#                 clustering at the same bottleneck width as the learned
#                 layer.
#   'random'   -- a fixed, balanced random partition into max_clusters
#                 groups: same bottleneck, no structure.
FIXED_POOLING_TYPES = ('hem', 'spectral', 'random')


def _fixed_trailing_parents(
    kind: str,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n: int,
    k: int,
    seed: int = 0,
) -> torch.Tensor:
    """Fixed node -> cluster mapping (length ``n``, values ``0..k'-1``) for
    the 'spectral' and 'random' trailing-layer controls. Deterministic for a
    given graph and seed, so rebuilding the model (e.g. for the final retrain
    or when reloading ``final_model.pt``) reproduces the same clusters."""
    if kind == 'random':
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        parents = torch.empty(n, dtype=torch.long)
        parents[perm] = torch.arange(n) % k
    elif kind == 'spectral':
        import scipy.sparse
        from sklearn.cluster import SpectralClustering

        ei = edge_index.cpu().numpy()
        ew = edge_weight.detach().cpu().numpy().astype(np.float64)
        A = scipy.sparse.coo_matrix((ew, (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = (A + A.T) / 2
        labels = SpectralClustering(
            n_clusters=k, affinity='precomputed', assign_labels='cluster_qr', random_state=seed,
        ).fit_predict(A)
        _, labels = np.unique(labels, return_inverse=True)  # contiguous ids
        parents = torch.from_numpy(labels).long()
    else:
        raise ValueError(f"unknown fixed trailing assignment {kind!r}")
    return parents.to(edge_index.device)


class DiffPoolGNN(nn.Module):
    """Hierarchical pooling GNN with learnable cluster assignments.

    Two modes:

    **Hybrid mode** (``full_mode=False``, default):
      The first ``n_hybrid`` layers use pre-computed HEM coarse edges +
      parent-based scatter pooling; the trailing layer switches to full
      learned DiffPool. The number of layers is ``n_hybrid + 1``.

      ``n_hybrid`` is a *level count*, not a node-count threshold -- the
      actual node count the trailing learned layer receives is whatever HEM's
      own precomputed per-level reduction happens to leave after ``n_hybrid``
      halvings (e.g. ``n_hybrid=2`` leaves ~3,500 nodes on the real
      14,133-node PPI graph, not some small fixed number). There used to be a
      ``dense_threshold`` parameter here documented as a dynamic node-count
      switch; it was never actually read anywhere in this class or in
      ``build_diffpool_model`` and has been removed -- see
      changes-from-claude.md fix #4.

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
    layer's learned-assignment branch: ``'diffpool'`` (default, link
    prediction + entropy losses) or ``'dmon'`` (Deep Modularity Networks,
    modularity + collapse-regularization losses -- see ``DMoNLayer``). Both
    apply ``assign_dropout`` to their raw assignment logits before the
    softmax (Tsitsulin et al., DMoN paper, JMLR 2023) whenever that branch
    is reached, hybrid mode's trailing layer included.

    ``pooling_type`` may also be one of ``FIXED_POOLING_TYPES`` (hybrid mode
    only): the trailing layer then pools with a fixed assignment instead of a
    learned one and contributes no auxiliary loss. These are controls for
    separating "learned vs. fixed clustering" from "how narrow the
    bottleneck is" -- see ``FIXED_POOLING_TYPES``.

    ``n_out_nodes`` is the number of nodes the last layer outputs, i.e. the
    classifier input is ``n_out_nodes * last_channels``.

    ``pool_gnn_layers`` sets every layer's assignment-head depth (default 1,
    a plain single ``ChebConv``, identical to the original code); see
    ``PoolAssignmentHead``.

    ``per_patient_graphs`` (full mode only, default True) controls whether
    each non-last level builds its own per-sample pooled graph for the next
    level (correct, but costs a dense per-edge gather tensor) or the
    original batch-averaged one (much cheaper, batch-composition-dependent
    -- see changes-from-claude.md #13/#14).
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
        K: int = 2,
        full_mode: bool = False,
        n_levels: Optional[int] = None,
        n_nodes: Optional[int] = None,
        encoder_channels: int = 16,
        encoder_layers: int = 2,
        pooling_type: str = 'diffpool',
        sparsify_density: Optional[float] = None,
        assign_dropout: float = 0.5,
        pool_gnn_layers: int = 1,
        per_patient_graphs: bool = True,
    ):
        super().__init__()
        self.max_filters = max_filters

        if pooling_type not in LEARNED_POOLING_TYPES + FIXED_POOLING_TYPES:
            raise ValueError(
                f"pooling_type must be one of {LEARNED_POOLING_TYPES + FIXED_POOLING_TYPES}, "
                f"got {pooling_type!r}"
            )
        fixed_trailing = pooling_type in FIXED_POOLING_TYPES
        if fixed_trailing and full_mode:
            raise ValueError(f"pooling_type={pooling_type!r} is a hybrid-mode control; it has no full mode")
        if fixed_trailing and (parents_list is None or len(parents_list) <= n_hybrid):
            raise ValueError(
                f"pooling_type={pooling_type!r} needs HEM parents for level {n_hybrid} "
                f"(got {0 if parents_list is None else len(parents_list)} levels)"
            )
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

        # Fixed-trailing controls reuse DiffPoolLayer's HEM-scatter branch
        # (aux losses are 0.0 there; their lambdas are 0 too).
        LayerClass = DMoNLayer if pooling_type == 'dmon' else DiffPoolLayer
        layer_extra_kwargs = {
            'assign_dropout': assign_dropout, 'pool_gnn_layers': pool_gnn_layers,
            'per_patient_graphs': per_patient_graphs,
        }
        if full_mode:
            layer_extra_kwargs['sparsify_density'] = sparsify_density

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

        if fixed_trailing:
            trailing = self.diffpool_layers[-1]
            n_in = len(parents_list[n_hybrid])
            if pooling_type == 'hem':
                parents = parents_list[n_hybrid]
            else:
                ei, ew = coarse_edges[n_hybrid]
                parents = _fixed_trailing_parents(
                    pooling_type, ei, ew, n_in, _compute_pool_k(n_in, 2, max_clusters)
                )
            # The trailing layer's output graph is never consumed downstream.
            trailing.set_coarse_edges(None, None, parents=parents)
            self.n_out_nodes = int(parents.max()) + 1
        elif full_mode:
            prev = n_nodes
            for target in cluster_schedule:
                prev = _compute_pool_k(prev, 2, target) if prev is not None else target
            self.n_out_nodes = prev
        else:
            n_in = len(parents_list[n_hybrid]) if len(parents_list) > n_hybrid else None
            self.n_out_nodes = _compute_pool_k(n_in, 2, max_clusters) if n_in is not None else max_clusters

    def forward(self, X: torch.Tensor):
        num_samples, num_features = X.shape
        H = torch.reshape(X, (num_samples, num_features, 1))

        edge_index = self.base_edge_index
        edge_weight = self.base_edge_weight

        if self.encoder is not None:
            H = self.encoder(H, edge_index, edge_weight)

        aux_records = []

        n_layers = len(self.diffpool_layers)
        for i, layer in enumerate(self.diffpool_layers):
            H, edge_index, edge_weight, aux = layer(
                H, edge_index, edge_weight, batch_size=num_samples, is_last_level=(i == n_layers - 1)
            )
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
    mlp_hidden_dim: Union[int, Tuple[int, ...]] = (256,),
    mlp_dropout: float = 0.5,
    K: int = 2,
    full_mode: bool = False,
    n_levels: Optional[int] = None,
    encoder_channels: int = 16,
    encoder_layers: int = 2,
    pooling_type: str = 'diffpool',
    sparsify_density: Optional[float] = None,
    assign_dropout: float = 0.5,
    pool_gnn_layers: int = 1,
    per_patient_graphs: bool = True,
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
        ``'diffpool'`` (default) or ``'dmon'``, or a hybrid-mode fixed-
        trailing control ``'hem'``/``'spectral'``/``'random'`` -- see
        ``DiffPoolGNN`` and ``FIXED_POOLING_TYPES``.
    sparsify_density : float, optional
        Full mode only: prune each level's pooled output adjacency to this
        fraction of edges per node instead of leaving it fully connected
        (see ``_build_pooled_output_graph``). ``None`` (default) disables
        pruning, matching prior behavior.
    assign_dropout : float
        Dropout applied to each layer's raw assignment logits before the
        softmax, whenever the learned-assignment branch is reached (both
        pooling types, hybrid mode's trailing layer included). Default 0.5
        matches Tsitsulin et al. (DMoN paper, JMLR 2023).
    pool_gnn_layers : int
        Depth of every layer's assignment head. Default 1 is a plain single
        `ChebConv`, identical to the original code; see `PoolAssignmentHead`.
    per_patient_graphs : bool
        Full mode only, default True: each non-last level's own pooled
        graph feeds the next level, instead of the cheaper but batch-
        composition-dependent batch-averaged graph. See `DiffPoolGNN`.
    """
    gnn_model = DiffPoolGNN(
        base_edge_index=base_graph.edge_index,
        base_edge_weight=base_graph.edge_weight,
        coarse_edges=coarse_edges,
        n_hybrid=n_hybrid,
        parents_list=parents_list,
        max_filters=max_filters,
        max_clusters=max_clusters,
        K=K,
        full_mode=full_mode,
        n_levels=n_levels,
        n_nodes=base_graph.num_nodes,
        encoder_channels=encoder_channels,
        encoder_layers=encoder_layers,
        pooling_type=pooling_type,
        sparsify_density=sparsify_density,
        assign_dropout=assign_dropout,
        pool_gnn_layers=pool_gnn_layers,
        per_patient_graphs=per_patient_graphs,
    )

    if full_mode:
        levels = n_levels if n_levels is not None else 3
    else:
        levels = n_hybrid + 1

    # Flatten the last layer's n_out_nodes nodes (max_clusters for the
    # learned layers at every configuration run so far; the HEM/spectral/
    # random controls' own cluster count otherwise). Channel doubling starts
    # from encoder_channels in full mode (the pre-pooling encoder's output
    # width) instead of the raw 1-dim input.
    start_channels = encoder_channels if full_mode else 1
    last_channels = min(start_channels * 2 ** levels, max_filters)
    mlp_input_dim = gnn_model.n_out_nodes * last_channels

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
    record. `build_hp_config` (diffpool_experiment.py) always zeroes the
    weight for the pair that doesn't apply to the model's pooling_type, so
    the `weight == 0` check below skips indexing into records that legitimately
    lack that key -- callers must preserve that invariant (never pass a
    nonzero weight for a key a record can't have) rather than relying on a
    silent per-key default here, so a real missing/mistyped key still raises.
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
                total = total + weight * sum(r[key] for r in records)
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

