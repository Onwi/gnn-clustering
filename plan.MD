Proposed Architecture
Replace the fixed HEM cluster indices with DiffPool-style soft assignment layers. At each level, a dedicated GNN produces a soft assignment matrix S ∈ ℝ^{n × k} that clusters n nodes into k super-nodes, then pools both features and adjacency:
- Feature pooling: X' = S^T Z where Z is output of an embedding GNN
- Adjacency pooling: A' = S^T A S (dense), then converted back to edge_index/edge_weight
- Auxiliary losses: link prediction loss ||A - S S^T||_F + entropy regularization on S to encourage crisp assignments
Practical Concern
Full DiffPool adjacency pooling (S^T A S) for the initial level (thousands of genes) is O(n²) memory — intractable. Two options:
1. Hybrid: First 1-2 levels use a fixed pool ratio with sparse operations and keep the pre-computed coarse edges; once nodes are < ~500, switch to full learned adjacency pooling
2. Pool adjacency sparsely throughout: Use sparse-dense matmul (A @ S) to avoid materializing full A
Files to Change
File	Changes
src/pooling_genomic/models.py	Add DiffPoolLayer class; add build_diffpool_model() builder; optionally keep existing GNNPooling for comparison
src/pooling_genomic/networks.py	Add load_base_graph() to load only the base STRING graph (no hierarchical parents)
src/pooling_genomic/engines.py	Add link_pred_loss and entropy_loss computation to train_epoch_clf, exposed as lambda_link_pred / lambda_entropy knobs
Experiment scripts	New entrypoint or add --diffpool flag to existing ones
Key Question
For the "number of clusters learned dynamically" — the most practical approach is to parameterize the number of clusters at each level as k_l = max(min_nodes, ceil(n_l * pool_ratio)) where pool_ratio is a trainable parameter (one per level, or global). This lets the model decide how aggressively to coarsen. Alternatively, k_l could simply be a fixed hyperparameter (like n_l // 2), and "dynamic" means the assignment is learned (not the count). Which do you prefer?

###################
###################
###################
###################
###################

Here's the refined plan:
Architecture Design
1. DiffPoolLayer (models.py)
A new module that handles one level of differentiable pooling:
Input: X [n, d], edge_index [2, e], edge_weight [e]
  │
  ├── Embed GNN (ChebConv) → Z [n, d']      (feature transform)
  ├── Pool GNN (ChebConv) → S_raw [n, k_max] (raw assignments)
  │
  ├── Compute k = max(min_nodes, ceil(n × sigmoid(pool_ratio)))
  │   where pool_ratio is a nn.Parameter (per level or shared)
  │
  ├── Truncate S to S[:, :k], softmax over columns → S [n, k]
  │
  ├── X' = Sᵀ Z                              [k, d']  (pooled features)
  │
  └── If "early" mode (hybrid, n > threshold):
  │       Use pre-computed coarse edge_index (from HEM), no adjacency pooling
  │
      If "full" mode (n ≤ threshold, ~500 nodes):
  │       A = dense_adj(edge_index)           [n, n]
  │       A' = Sᵀ A S                         [k, k]
  │       edge_index' = dense_to_sparse(A')
  │       edge_weight' = ... (derived from A')
  └── Return: X', edge_index', edge_weight', auxiliary dict {S, A, link_pred_loss, entropy}
2. DiffPoolGNN — replaces GNNPooling
Input: X [batch, num_features, 1]  (gene expression)
  │
  │  Level 0: ChebConv message passing (no pooling)
  │
  ├── Level 1 (hybrid): DiffPoolLayer with pre-computed HEM edges
  ├── Level 2 (hybrid): DiffPoolLayer with pre-computed HEM edges
  │     └── threshold check → next levels use full mode
  ├── Level 3 (full):  DiffPoolLayer with learned adjacency
  ├── Level 4 (full):  DiffPoolLayer with learned adjacency
  └── ...
       │
       Flatten → MLP classifier (same as now)
3. Builder: build_diffpool_model
Similar to build_coarsening_model but:
- Takes levels_config: list of {pool_ratio_init, hybrid_mode} per level
- Still loads initial STRING graph + pre-computed coarse edges for early levels
- Initialize each DiffPoolLayer with its config, including a nn.Parameter for pool_ratio
4. Auxiliary Losses (engines.py)
train_epoch_clf gets two new knobs:
- lambda_link_pred — weight for ||A - SSᵀ||_F (encourages nearby nodes to cluster together)
- lambda_entropy — weight for 1/n Σ H(S_i) (encourages crisp assignments)
5. Network Loading (networks.py)
Add load_base_graph_and_coarse_edges():
- Same as load_graph_levels but returns a tuple: (base_graph, coarse_edges_list) where coarse edges are only for levels up to the hybrid threshold
6. Experiment Scripts
New script or add --diffpool flag to coarsening_levels.py:
- Pass --diffpool to use build_diffpool_model instead of build_coarsening_model
- Add --hybrid-levels N to set how many early levels use hybrid mode
- Add --link-pred-weight and --entropy-weight loss knobs
Trainable pool_ratio Details
Each level with DiffPool has:
self.logit_pool_ratio = nn.Parameter(torch.tensor(0.0))  # sigmoid gives 0.5
At forward: pool_ratio = torch.sigmoid(self.logit_pool_ratio) clamped to (min_ratio, max_ratio), then k = max(min_nodes, ceil(n * pool_ratio))
For the "global" option, a single logit_pool_ratio is shared across all levels.
Next step when you're ready: I'll implement this, starting with DiffPoolLayer in models.py, adding the load_base_graph_and_coarse_edges helper, updating the engine, and wiring up the experiment entrypoint.

###########################
Build
#####################3333
All changes are in place. Here's a summary of what was implemented:
Files Modified
src/pooling_genomic/networks.py
- load_coarse_edges_for_diffpool() — loads (edge_index, edge_weight) per level (no parents), used by hybrid mode levels
src/pooling_genomic/models.py
- DiffPoolLayer (models.py:393) — differentiable pooling block:
- Two ChebConv GNNs (embed + pool) operating on flattened (B*N, F) input
- Soft assignment S via truncated softmax over k columns
- Hybrid mode: k matches pre-computed coarse graph node count; fixed coarse edges passed through for next level
- Full mode: k = max(2, ceil(n * σ(pool_ratio))) where pool_ratio is a nn.Parameter; adjacency pooled via SᵀAS and converted back to sparse
- Auxiliary losses: link prediction (||A/||A|| - SSᵀ/||SSᵀ||||_F) + entropy regularization
- DiffPoolGNN (models.py:506) — stacks n_hybrid+1 DiffPoolLayers:
- Base ChebConv on full gene graph → n_hybrid hybrid layers → full-mode layers (auto-switch when nodes ≤ dense_threshold)
- Final global mean pooling → fixed (batch, hidden_channels) vector
- build_diffpool_model() — builder that creates nn.Sequential(DiffPoolGNN, FCModel)
- get_diffpool_aux_losses() — helper to extract and sum auxiliary losses from the model
src/pooling_genomic/engines.py
- train_epoch_clf and train_cohort_tumor_clf now accept lambda_link_pred and lambda_entropy — auxiliary loss is computed each step via get_diffpool_aux_losses
New File
scripts/experiments/diffpool_experiment.py
Full experiment script supporting both tuning and direct runs. Usage:
# Quick test
python scripts/experiments/diffpool_experiment.py <path_to_data> <path_to_levels> --path-network <path_to_stringdb_csv> --n-hybrid 2
# With Ray Tune
python scripts/experiments/diffpool_experiment.py <path_to_data> <path_to_levels> --path-network <path_to_stringdb_csv> --tune --n-hybrid 2 --num-samples 8 --n-holdouts 5
Key architectural choices
- Hybrid → Full transition: controlled by dense_threshold (default 500 nodes). Below this, the adjacency is pooled densely and learned; above it, pre-computed HEM coarse edges provide structure
- Fixed MLP input dim: global mean pooling over final nodes gives a constant hidden_channels-sized vector regardless of how many clusters pool_ratio produces
- Pool ratio is per-layer: each DiffPoolLayer has its own logit_pool_ratio Parameter (initialized at σ(0)=0.5)
