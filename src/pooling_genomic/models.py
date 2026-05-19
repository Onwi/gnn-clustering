from functools import partial
from typing import Union, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import ChebConv
from torch_geometric.utils import from_networkx
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

