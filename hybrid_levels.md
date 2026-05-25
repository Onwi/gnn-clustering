# Hybrid Levels in DiffPoolGNN

A **hybrid level** is a middle ground between fixed HEM pooling and fully learned DiffPool. It uses **learned GNN feature transforms** but **fixed HEM structure** (parent mappings and coarse edges).

## How It Works

In `DiffPoolLayer.forward()`, when `self._parents is not None`, the layer operates in hybrid mode:

### Step-by-step for a hybrid level (e.g. 14K → 7K nodes):

1. **Learned GNN embedding** — message passing with **learnable weights** transforms node features:

   ```
   z = ReLU(ChebConv(x, edge_index, edge_weight))
   ```

   The ChebConv weights are trained via backpropagation — this is the "learned" part.

2. **Fixed HEM pooling** — the pre-computed HEM parent tensor maps each node to a cluster. All nodes in a cluster are mean-pooled into one supernode:

   ```
   x_next = scatter(z, parents, dim=1, reduce='mean')
   ```

   No learned assignment matrix `S` is computed. This avoids the OOM-causing `(batch, 14K, 7K)` intermediate.

3. **Fixed coarse edges** — the next level's graph structure comes from the pre-computed HEM hierarchy, loaded via `set_coarse_edges()`:

   ```
   edge_index_next ← coarse_edge_index[i+1]   (pre-loaded)
   edge_weight_next ← coarse_edge_weight[i+1]  (pre-loaded)
   ```

4. **No auxiliary losses** — link prediction and entropy losses return 0.0:

   ```
   aux = {'link_pred_loss': 0.0, 'entropy_loss': 0.0}
   ```

## Why Hybrid?

| Aspect | Why it exists |
|---|---|
| **Memory** | Learned assignment `S` would be `(batch, 14K, 7K)` at level 0 = 23 GiB OOM. Hybrid uses O(n) scatter instead of O(n²) bmm. |
| **Speed** | No `S^T @ A @ S` dense pooling needed at large sizes. |
| **Learnability** | Still has learnable GNN weights at each level to transform features before pooling. |
| **Structure** | HEM's heavy-edge matching gives a strong structural prior that the model inherits for free. |

## Transition to Full Mode

Hybrid mode continues until the graph shrinks to ≤ `dense_threshold` (default 500 nodes). At that point, `_coarse_edge_index` and `_parents` are cleared:

```python
if n_nodes <= self.dense_threshold and layer._coarse_edge_index is not None:
    layer._coarse_edge_index = None
    layer._parents = None
```

The next forward pass falls through to **full mode**: learned `S = softmax(pool_gnn(x))`, dense adjacency pooling `S^T @ A @ S`, and auxiliary losses for regularization.

## Visualization

```
Hybrid Level                        Full Level
─────────────────                   ─────────────────
x ──► ChebConv ──► z                x ──► ChebConv ──► z
     (learned)                           │
         │                               ├──► pool_gnn ──► S = softmax(…)
         ▼                               ▼
    scatter(z, parents, mean)       x_next = S^T @ z
    (fixed HEM parent tensor)       A_next = S^T @ A @ S
         │                           (fully learned)
         ▼
    edge_index from coarse_edges    edge_index from dense_to_sparse(A_next)
    (fixed HEM structure)           aux = {link_pred, entropy} ≠ 0
