# Hybrid DMoN: A Complete Technical Reference

This document explains, in full, how Hybrid DMoN is implemented in this codebase, the exact
mathematics it computes, and the reasoning behind every non-obvious design decision made while
building and correcting it. It assumes the reader knows the project's general framing (RNA-seq
gene expression classification over a PPI graph, TCGA pan-cancer, 16 classes) but not the specific
implementation.

For the overall project's other pooling methods (Fixed HEM, Full DiffPool, Full DMoN, Hybrid
DiffPool) and their comparative results, see `plan.md`, `analysis-approaches.MD`, and
`changes-from-claude.md`.

---

## 1. What Hybrid DMoN is, in one paragraph

Hybrid DMoN classifies a patient's tumor/cohort type from gene expression by hierarchically pooling
a 14,133-node protein-protein interaction (PPI) graph down to a small, fixed-size representation.
Early levels use a **fixed, pre-computed hierarchy** (Heavy Edge Matching, HEM) that requires no
learning at all. Only the **final** pooling step is learned: a Deep Modularity Network (DMoN) layer
that produces a soft cluster assignment trained jointly with the downstream classifier, using an
unsupervised graph-clustering objective (modularity + collapse regularization) as an auxiliary loss
alongside the classification cross-entropy. "Hybrid" refers to this split -- fixed early, learned
late -- as opposed to "Full" DMoN, which learns every level's assignment from scratch.

---

## 2. The full pipeline, step by step

### 2.1 Input

Each training example is one patient: a vector of 14,133 raw gene-expression values, one scalar per
gene/node. The PPI graph (STRING-DB, `stringdb_top100pc.csv`) is shared across every patient --
edges (protein-protein interactions) and edge weights (interaction confidence, `combined_score /
1000`) never change per-sample, only the node features do.

### 2.2 The pre-computed HEM hierarchy

Before any training happens, a separate offline script (Heavy Edge Matching coarsening, see
`plan.md` for the algorithm) builds a sequence of progressively coarser graphs by greedily merging
nodes connected by the strongest edges. This produces, per level `L`, three files:

- `edge_index_lvl{L}.pt` / `edge_weight_lvl{L}.pt` -- the graph's edges *at* level `L`
- `parents_lvl{L}.pt` -- a mapping from level `L`'s node indices to level `L+1`'s (which supernode
  each node merges into)

On the real graph, the per-level node counts are:

| level | nodes |
|---|---|
| 0 (base) | 14,133 |
| 1 | 7,067 |
| 2 | 3,534 |
| 3 | 1,767 |
| 4 | 884 |
| 5 | 442 |
| 6 | 221 |
| 7 | 111 |

`load_coarse_edges_for_diffpool()` (`src/pooling_genomic/networks.py`) just loads these tensors
from disk; no computation happens at training time for this part.

### 2.3 Model construction (`build_diffpool_model` / `DiffPoolGNN`, `src/pooling_genomic/models.py`)

For Hybrid mode (`full_mode=False`), the model has `n_hybrid + 1` pooling layers. Given `n_hybrid`,
`DiffPoolGNN.__init__` builds `n_hybrid + 1` `DMoNLayer` instances (for `pooling_type='dmon'`) and,
for layer `i < n_hybrid`, calls `layer.set_coarse_edges(coarse_edges[i+1], parents=parents_list[i])`
-- handing that layer the pre-computed level-`(i+1)` graph and the level-`i`-to-`(i+1)` parent
mapping. The **last** layer (index `n_hybrid`) never gets this call, so its `_parents` attribute
stays `None`.

This `_parents is None` check, inside `DMoNLayer.forward()`, is what actually decides which
pooling mechanism a layer uses -- there is no separate "hybrid layer class" vs. "learned layer
class"; `DMoNLayer` implements both branches, and which one runs is decided per-forward-call based
on whether coarse edges were ever attached to that specific layer instance.

Channel widths grow progressively across layers via `_compute_channel_list`: for Hybrid mode
(`start_channels=1`, since there's no pre-pooling encoder -- see 2.6), layer `i`'s input/output
channels are `min(2^i, max_filters)` / `min(2^(i+1), max_filters)`. E.g. for `n_hybrid=5` (6
layers total): `(1,2), (2,4), (4,8), (8,16), (16,32), (32,32)`.

Every layer's `max_clusters` bound (the width of its `pool_gnn`, hence the ceiling on how many
clusters it can produce) is set from `cluster_schedule`. In Hybrid mode this is **flat** --
`[max_clusters] * levels` (the progressive geometric schedule in `_compute_cluster_schedule` only
activates when `full_mode=True`; see §5.2 for why this matters).

### 2.4 Forward pass: the early (HEM) layers

For layer `i < n_hybrid`, `DMoNLayer.forward()` takes the `_parents is not None` branch:

```python
z = F.relu(self.embed_gnn(x, edge_index, edge_weight=edge_weight))   # ChebConv, K=2
x_next = scatter(z, self._parents, dim=1, reduce='mean')             # merge nodes -> supernodes
aux = {'modularity_loss': 0.0, 'collapse_loss': 0.0}
return x_next, self._coarse_edge_index, self._coarse_edge_weight, aux
```

`embed_gnn` is a `ChebConv` (Chebyshev spectral graph convolution, filter order `K=2`, so each
layer's output at a node depends on itself and everything within 2 hops). After message-passing,
nodes are merged into their pre-determined supernodes by averaging (`scatter(..., reduce='mean')`)
-- purely deterministic, no learned assignment, and **zero auxiliary loss contribution** (the
`0.0`s above). The edge_index/edge_weight handed to the *next* layer is the precomputed graph at
the new, smaller level -- not derived from this layer's computation at all.

### 2.5 Forward pass: the trailing learned layer

For layer `i == n_hybrid` (the last one), `_parents is None`, so `DMoNLayer.forward()` takes the
full-learned branch:

```python
k = _compute_pool_k(n, min_nodes, self.pool_gnn.out_channels)   # = max_clusters, clamped by n

c_raw = self.pool_gnn(x, edge_index, edge_weight=edge_weight)   # ChebConv, K=2
c_raw = F.dropout(c_raw, p=self.assign_dropout, training=self.training)   # see §5.4
C = F.softmax(c_raw[:, :, :k], dim=-1)                          # (batch, n, k) soft assignment

x_next = torch.bmm(C.transpose(1, 2), z)                        # X' = C^T Z, pooled features
```

`n` here is whatever node count the previous (HEM) layers left it with -- **not** a fixed "small
graph" size; see §5.1 for why this number matters enormously. `C` is the soft cluster-assignment
matrix: `C[b, i, j]` is patient `b`'s node `i`'s (soft) membership weight in cluster `j`. Every row
of `C` sums to 1 (it's a softmax over clusters). Features are pooled as `X' = C^T Z`: each output
cluster's feature vector is the assignment-weighted average of its member nodes' embeddings.

This is the **only** layer in the entire Hybrid DMoN model that learns an assignment or
contributes a non-zero clustering loss.

### 2.6 Pooled adjacency and the modularity/collapse losses (this layer only)

The trailing layer also computes a pooled adjacency, needed for the modularity loss:

```python
_, A_next_dense, degree = _pool_adjacency(edge_index, edge_weight, C, n, batch_size, k, compute_degree=True)
```

`_pool_adjacency` computes `A_next_dense[b] = C[b]^T @ A @ C[b]` (a `k x k` matrix per sample) via
one batched sparse-dense matmul (never materializing a dense `n x n` copy of `A`), plus the
(weighted) degree vector `d`, fused into the same matmul pass.

**Modularity loss.** Given the modularity matrix `B = A - dd^T/2m` (`m` = total edge weight / 2),
modularity is `Q = (1/2m) * [Tr(C^T A C) - (1/2m)||C^T d||^2]`, and the loss is `-Q` (since training
minimizes loss, and we want to *maximize* Q):

```python
m = torch.clamp(edge_weight.sum() / 2, min=1e-8)
trace_CAC = torch.diagonal(A_next_dense, dim1=-2, dim2=-1).sum(dim=-1)   # Tr(C^T A C), per sample
Cd = torch.einsum('bnk,n->bk', C, degree)                                # C^T d, per sample
deg_term = (Cd ** 2).sum(dim=-1)                                         # ||C^T d||^2, per sample
modularity = trace_CAC / (2 * m) - deg_term / (2 * m) ** 2
modularity_loss = -modularity.mean()
```

**Collapse regularization.** Penalizes uneven cluster sizes (0 when perfectly balanced, `sqrt(k)-1`
in the degenerate case where every node lands in one cluster):

```python
cluster_sizes = C.sum(dim=1)   # (batch, k), total assignment mass per cluster
collapse_loss = ((sqrt(k) / n) * cluster_sizes.norm(dim=-1) - 1).mean()
```

Both formulas were checked against the original DMoN paper (Tsitsulin, Palowitch, Perozzi, Müller,
*"Graph Clustering with Graph Neural Networks"*, JMLR 2023, arXiv:2006.16904) and match exactly,
including normalization constants -- see §5.4 for the one place this codebase's *weighting* of
these two terms deliberately departs from the paper.

### 2.7 After pooling: flatten and classify

After all `n_hybrid + 1` layers run, `DiffPoolGNN.forward()` flattens the final `(batch, k,
channels)` tensor to `(batch, k * channels)` and hands it to `FCModel`, a standard
Linear-BatchNorm-ReLU-Dropout MLP that outputs class logits. The trailing layer's *output graph*
(`edge_index_next`/`edge_weight_next`, built from `A_next_dense`) is computed but never consumed by
anything downstream, since there is no layer after it -- see §5.2 for why this makes
`sparsify_density` irrelevant to Hybrid mode specifically.

---

## 3. Training

### 3.1 Loss

Total loss = classification cross-entropy (optionally class-weighted, see §3.4) + the weighted sum
of every layer's auxiliary losses:

```python
aux_loss = lambda_modularity * sum(modularity_loss for each layer) \
         + lambda_collapse   * sum(collapse_loss   for each layer)
loss = cross_entropy_loss + aux_loss
```

(`get_diffpool_aux_losses`, `src/pooling_genomic/models.py`.) Since every early layer's
`modularity_loss`/`collapse_loss` is exactly `0.0`, this sum is really just the trailing layer's
two losses, scaled by `lambda_modularity`/`lambda_collapse`.

### 3.2 Optimizer and schedule

`AdamW`, with `CosineAnnealingWarmRestarts(T_0=1, T_mult=2, eta_min=1e-5)`. This schedule restarts
at epochs `2^n - 1`: 1, 3, 7, 15, 31, 63, 127, .... Validation/test reads are only trustworthy at
these boundaries -- mid-cycle, the learning rate is still descending from a restart spike, and loss
can be highly volatile (see §5.3 for how this bit us, twice).

Gradient clipping (`max_norm=5.0`) is applied on every step, unconditionally, for every
configuration (`engines.py::train_epoch_clf`).

### 3.3 Hyperparameter search space (`build_hp_config`, `--tune` mode)

```
lr:                 loguniform(1e-4, 1e-1)
weight_decay:       loguniform(1e-4, 1e-1)
lambda_modularity:  loguniform(1e-3, 1e1)
lambda_collapse:    loguniform(1e-3, 1e1)
lambda_link_pred:   0.0   (DiffPool-only, forced off for DMoN)
lambda_entropy:     0.0   (DiffPool-only, forced off for DMoN)
```

`lambda_modularity`/`lambda_collapse` start two orders of magnitude higher than DiffPool's
`lambda_link_pred`/`lambda_entropy` (`1e-3`-`1e1` vs. `1e-5`-`1e-3`) because modularity is bounded
in `[-0.5, 1]` and collapse in `[0, sqrt(max_clusters)-1]` -- both O(1) quantities, unlike
DiffPool's losses which operate on a different natural scale.

Tuning uses `ASHAScheduler`, with `max_t`/`grace_period` computed from `--n-cycles` so that both
the pruning checkpoint and the tuning-phase epoch budget land exactly on cosine-restart boundaries
(see §5.3.2) -- never at an arbitrary, potentially mid-cycle epoch count.

### 3.4 Class weights and holdout protocol

`--use-train-set-weights` computes per-class weights from the training split and passes them to
`nn.CrossEntropyLoss(weight=...)`, added after both pooling types showed a large gap between raw
accuracy and balanced accuracy in early runs (real class imbalance in the TCGA cohort labels).

`--n-holdouts N` repeats the entire tune-then-retrain procedure `N` times with independent random
splits (`run_holdout`, called once per rep from `main()`), each rep tuning its own hyperparameters
from scratch. Reported results are mean ± std across reps -- a single rep cannot distinguish a real
effect from run-to-run variance (see §5.3.1, where exactly this mistake produced a spurious
result).

### 3.5 Final retrain and evaluation

Once tuning finishes, `test_tuned_model` retrains a fresh model on **train+val combined**, using
`results.get_best_result(scope="all").config` (the best hyperparameters found, selected by the best
epoch across each trial's full run -- not the last epoch; see §5.3.1), for `_cosine_restart_epochs(T_0, T_mult, args.n_cycles)`
epochs (127 for `--n-cycles 7`), then evaluates on the held-out test set. Per-epoch train/test
metrics are recorded into `metrics.csv`; the number that's actually reported is the **last row**
(the final restart-boundary epoch), not any of the noisy mid-cycle rows in between.

---

## 4. What actually varies between DiffPool and DMoN

`DiffPoolGNN` is a single shared wrapper (`pooling_type='diffpool'` or `'dmon'` selects
`DiffPoolLayer` or `DMoNLayer` as `LayerClass`) -- identical channel schedule, identical optimizer
and scheduler, identical hybrid/full-mode branching logic, identical `assign_dropout`. The *only*
difference is the auxiliary objective shaping the trailing layer's assignment:

| | DiffPool | DMoN |
|---|---|---|
| Assignment matrix | `S` | `C` |
| Aux losses | link-prediction (reconstruct `A` from `SS^T`) + entropy | modularity + collapse regularization |
| Losses computed | per-sample (see `changes-from-claude.md` fix #4c) | per-sample |
| Reference | Ying et al., DiffPool (NeurIPS 2018) | Tsitsulin et al., DMoN (JMLR 2023) |

This deliberate architectural symmetry is what let this investigation isolate loss-function effects
from confounding architecture differences at every step.

---

## 5. Key decisions, and the investigation that produced them

### 5.1 `n_hybrid` is a level count, not a node-count threshold -- and this was the single biggest lever found

`DiffPoolGNN` used to accept a `dense_threshold` parameter, documented as "node count below which
hybrid mode switches to full mode" (default 500). **It was dead code** -- never read anywhere in
the model-building logic, only threaded through function signatures. The actual switch point is
`i < n_hybrid` in the layer-construction loop: a fixed *count* of HEM levels, matched against
whatever node count the precomputed hierarchy happens to have reached by then.

Every Hybrid run in this project used `n_hybrid=2` until this was found. Per the table in §2.2,
that leaves the trailing learned layer with **~3,534 nodes** -- not the "≤500" that `plan.md`,
`ARCHITECTURE.md`, and this project's own narrative consistently assumed. The learned layer was
doing a single-hop ~110:1 compression (3,534→32) on a nearly-raw, narrow-channel feature
representation, with none of Full DMoN's mitigations (no pre-pooling encoder, no progressive
cluster schedule, no adjacency sparsification -- all gated behind `full_mode=True`, which Hybrid
mode never sets). Structurally, this is the same *kind* of ill-posed problem that made untuned Full
DiffPool collapse to near-random, just softened by having 4x fewer nodes at the compression point.

`dense_threshold` was removed entirely (`changes-from-claude.md` fix #4b); `DiffPoolGNN`'s
docstring now describes the real mechanism. Testing `n_hybrid=5` (which lands at 442 nodes, per the
table -- the value actually close to the long-assumed design point) on Hybrid DiffPool produced
**83.71% ± 7.68pp test accuracy** (75.05% / 86.39% / 89.70% per rep, full 3-rep protocol), up from
65.82% ± 7.02pp at `n_hybrid=2` -- a +17.89pp jump, and the best learned-pooling result in this
entire project, ahead of even the fully-rescued Full DiffPool/DMoN numbers (71.68%/71.16%). The
matching Hybrid DMoN run at `n_hybrid=5` is prepared (`scripts/run_hybrid_n5_dmon_full3.sh`) but not
yet executed as of this writing.

### 5.2 `sparsify_density` and the progressive cluster schedule don't apply to Hybrid mode -- correctly

Two fixes built for Full DMoN's own failure modes (see `changes-from-claude.md` fixes #1 and #3):
a progressive cluster-count schedule (spreading compression across multiple full-mode levels
instead of one hop) and pooled-adjacency sparsification (pruning each level's output graph to a
realistic density instead of leaving it fully connected). Both are gated behind `full_mode=True`
in `DiffPoolGNN.__init__` and correctly **do not** apply to Hybrid mode:

- The cluster schedule only kicks in when `full_mode and levels > 1` -- Hybrid mode's
  `cluster_schedule` stays flat (`[max_clusters] * levels`) regardless of `n_hybrid`, since only
  the *last* layer ever reaches the learned branch (there's nothing to spread the schedule across).
- `sparsify_density` is only added to `layer_extra_kwargs` when `full_mode=True`. This isn't an
  oversight: Hybrid mode's trailing layer's own output graph is never consumed downstream (§2.7),
  so pruning it would have zero effect even if applied.

`assign_dropout` (§5.4), by contrast, is added to `layer_extra_kwargs` **unconditionally** --
correct, since it affects the *quality of the assignment itself* (a within-layer training-stability
concern), not the graph handed downstream.

### 5.3 Two evaluation-methodology bugs, found and fixed before trusting any Hybrid DMoN number

An early single-rep Hybrid DiffPool vs. Hybrid DMoN comparison (at `n_hybrid=2`, before the fix in
§5.1) showed DiffPool at 53.92% and DMoN appearing to win by +11.4pp. Root-caused to two bugs, not a
real DMoN advantage:

#### 5.3.1 Reading the wrong epoch

`results.get_best_result(scope="last")` was selecting each tuning trial's *final-epoch* validation
value. On the cosine-restart schedule, the final epoch of a trial isn't guaranteed to land on a
converged trough -- it can land mid-oscillation, right after a restart. Fixed to `scope="all"`
(best epoch across the whole trial).

#### 5.3.2 A scheduler that never pruned, misaligned with the training schedule it was pruning

`ASHAScheduler(grace_period=max_t)` is a no-op -- grace period equal to the max epoch budget means
no trial is ever stopped early. Worse, the epoch budget wasn't aligned to the cosine schedule's own
restart boundaries, so on the rare occasion a rung *did* apply, it could fire mid-cycle. Fixed by
setting `grace_period` to the *previous* restart boundary (§3.3): the restart sequence's
consecutive-boundary ratio (3, 2.33, 2.14, 2.07, 2.03, ...) converges to `reduction_factor=2` as
`n` grows, so the rung ASHA computes automatically (`max_t / reduction_factor`) lands within about
one epoch of a real trough.

After both fixes, the corrected `n_hybrid=2` comparison (3 reps, 16 tuning samples each) gave
Hybrid DiffPool 65.82% ± 7.02pp vs. Hybrid DMoN 62.63% ± 11.86pp -- DiffPool narrowly ahead, and
crucially, that delta (3.20pp) is *smaller than DMoN's own standard deviation* -- the honest read
is a noise-adjacent lead, not a decisive one, and a complete reversal from the original buggy
result. This episode is also why every result in this project is reported as mean ± std across
multiple holdout reps (§3.4), never a single run.

### 5.4 Reading the DMoN paper directly: one fix taken, one taken partially, two declined

Before the `n_hybrid=5` Hybrid DMoN run, the original paper (Tsitsulin et al., JMLR 2023) was read
in full and cross-checked against every file on the DMoN code path. Four findings, four different
outcomes:

**Taken: assignment-head dropout.** The paper's assignment head is `C = softmax(GCN(...))` with
**0.5 dropout on the GNN representation before the softmax**, reported to specifically prevent
"gradient descent from getting stuck" in a degenerate assignment. Neither `DiffPoolLayer` nor
`DMoNLayer` had any dropout in the pooling path before this (`FCModel`, the classifier head, was
the only place with dropout). Added as `assign_dropout` (default 0.5) to both layer classes, and
applied to **both pooling types** -- not just DMoN -- specifically to avoid introducing a new
architectural asymmetry between DiffPool and DMoN that wasn't there before (§4's whole point is
that architecture stays identical between them). Verified: `model.eval()` forward passes are
bitwise-deterministic on CPU across repeated calls (dropout correctly disabled at eval time); on
GPU, repeated eval-mode calls differ by ~`3.6e-7` (ordinary CUDA sparse-op floating-point
non-determinism, unrelated to dropout).

**Taken partially / dropped a redundant parameter.** `DMoNLayer` used to also accept a
`collapse_regularization` parameter (a fixed multiplier on `collapse_loss`, default 1.0),
originally added to mirror the paper's internal hyperparameter. It was mathematically redundant
with the already-tuned `lambda_collapse` -- both linearly scale the exact same term with nothing
else combining them, so tuning both adds no search expressiveness beyond `lambda_collapse` alone.
Removed. (It was never actually part of the Ray Tune search space in the first place -- only a
fixed CLI override, always left at its default -- so this has zero numerical effect on any run to
date; pure cleanup.)

**Declined: the paper's fixed 1:1 modularity:collapse weighting.** The paper's actual loss is
`L = -1/(2m)*Tr(C^T B C) + [√k/n·‖ΣCᵢ‖_F − 1]` -- modularity and collapse added with coefficient 1
each, no independent weight. This codebase instead exposes `lambda_modularity` and `lambda_collapse`
as *independently* tuned hyperparameters. This is a deliberate, considered divergence, not an
oversight: the paper's fixed ratio was designed for **pure unsupervised clustering**, where the
modularity+collapse loss *is* the entire training signal; here, they're auxiliary regularizers
beside a supervised cross-entropy loss already doing most of the discriminative work, so there's no
a priori reason the paper's balance should transfer. Our own tuned configs already found ratios far
from 1:1 (one run: `lambda_modularity=0.028` vs. `lambda_collapse=6.34`, ≈226:1) -- tying them
together now would risk making the best achievable config *worse*, not better, by removing
flexibility the task may genuinely need, and would confound the `n_hybrid=5` result with a second,
untested change if bundled together. Left as a candidate follow-up ablation with spare compute.

**Declined: the paper's GCN architecture.** The paper's layer is `X^(t+1) = SeLU(ÃX^(t)W +
XW_skip)` -- a GCN with a trainable skip connection and SeLU activation, vs. this codebase's plain
`ChebConv` (K=2) with ReLU (`embed_gnn`) and no activation at all before softmax (`pool_gnn`).
Explicitly declined as too large an architecture change for this pass.

Full detail, including verification steps, is in `changes-from-claude.md` fix #5.

---

## 6. Current status (as of this writing)

| Config | Test accuracy | Status |
|---|---|---|
| Hybrid DiffPool, `n_hybrid=2` (original, corrected for §5.3's bugs) | 65.82% ± 7.02pp | done |
| Hybrid DMoN, `n_hybrid=2` (original, corrected for §5.3's bugs) | 62.63% ± 11.86pp | done |
| Hybrid DiffPool, `n_hybrid=5` (§5.1 fix) | **83.71% ± 7.68pp** | done |
| Hybrid DMoN, `n_hybrid=5` (§5.1 fix + §5.4's dropout addition) | -- | prepared, not yet run (`scripts/run_hybrid_n5_dmon_full3.sh`) |

For context, Fixed HEM (the fully pre-computed, zero-learning baseline) sits at 96.05%, and
MLP-on-expression-alone at 95.46% -- the graph structure adds relatively little on top of the raw
expression signal for this task (see `analysis-approaches.MD`), which is the deeper reason none of
the learned-pooling methods have closed the full gap.

---

## 7. Open questions for the next Hybrid DMoN run

- Does DMoN benefit from `n_hybrid=5` the way DiffPool did, or does its different loss family
  respond differently once the architecture is fair? (This is the whole point of the prepared run.)
- Does the tied 1:1 modularity:collapse weighting (§5.4, declined for now) help or hurt, once tested
  as its own ablation rather than bundled into this run?
- What would the paper's skip-connection + SeLU architecture (§5.4, declined) do here, if ever
  revisited as a larger architecture change?
