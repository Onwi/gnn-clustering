# Changes Made by Claude

Log of code changes made to this repository in collaboration with Claude, most recent first.
Each entry: what changed, why, files touched, and how it was verified.

---

## 2. Sparse-dense adjacency pooling for Full DiffPool (2026-08-23)

### Problem

In `DiffPoolLayer.forward()` (full mode), the pooled adjacency `A' = Sᵀ A S` was computed by
building a dense `(n, n)` adjacency matrix (`to_dense_adj`) and batching it via
`A_dense.unsqueeze(0).expand(batch_size, -1, -1)` into a `torch.bmm`. The base graph topology
is identical for every sample in the batch — only `S` varies per sample — so, in principle,
expanding `A` per batch item is unnecessary work, and the design docs (`ARCHITECTURE.md`,
`hybrid_levels.md`, `plan.md` §6.3) attribute the project's ~23 GiB OOM at the full ~14K-node
graph to exactly this kind of dense/batched adjacency operation.

### Fix

Replaced the dense expand+bmm with a sparse-dense formulation: build a `torch.sparse_coo_tensor`
from `edge_index`/`edge_weight` once, then compute `A @ S` **per sample** via `torch.sparse.mm`
(each call is `O(nnz · k)` instead of `O(n²)`), writing directly into a preallocated
`(batch, n, k)` output tensor rather than collecting per-sample results in a Python list/`stack`
(which would keep every sample's result alive simultaneously and double peak memory for no
benefit). Only the small `(batch, k, k)` `Sᵀ @ (A @ S)` product still uses a dense `bmm`. The
dense `to_dense_adj` construction is now used **only** for `link_pred_loss` (a single, non-batched
`(n, n)` reconstruction target — unchanged from before), not for the pooling step.

Correctness was verified against the old dense implementation (see below); numerically identical
to float precision, gradients w.r.t. `S` match to `~1e-6`.

### Files changed

- `src/pooling_genomic/models.py` — `DiffPoolLayer.forward()`, full-mode adjacency-pooling block.

### Verification

Using the `pooling_genomic` conda env (note: this env has PyTorch **2.1.0**, not the
`requirements.txt`/`README.md`-pinned **1.12.1** used for the dissertation's actual experiments
— see caveat below):

1. **Correctness**: on a small synthetic graph (n=50, batch=4, k=6), the new sparse
   implementation's forward output matches the old dense implementation to `atol=1e-5`, and
   `S.grad` after `.backward()` matches to `atol=1e-5` (max diff `~9.5e-7`).
2. **End-to-end**: `build_diffpool_model(..., full_mode=True)` on a synthetic 300-node graph
   builds, runs forward + backward (including the auxiliary losses), and produces correct
   gradients on both `embed_gnn` and `pool_gnn` weights. Hybrid mode (unaffected by this change,
   since it never reaches the full-mode branch for its early levels) was re-verified to still
   work.
3. **Memory, at realistic scale** (n=14,000, ~8M edges, batch=32) on the available RTX 3060
   (12 GB): this is where the result is more nuanced than expected — see below.

### Important finding: the memory picture is more nuanced than the docs suggested

Benchmarking at n=14,000/batch=32 with **k=32** (the old, pre-fix-#1 default, i.e. every layer
capped at `max_clusters`) showed the *old* dense-expand-bmm approach only peaks at ~1.3 GB, not
~23 GiB — in PyTorch 2.1.0, `torch.bmm` on a `.expand()`-broadcast batch dimension does **not**
materialize `batch_size` separate copies of `A_dense`; cuBLAS's strided-batched GEMM handles the
zero-stride batch dimension efficiently. So at `k=32`, the new sparse implementation isn't
meaningfully cheaper (it actually used marginally *more* memory in one measurement, ~1.5 GB, due
to sparse-tensor bookkeeping overhead).

Benchmarking at the **larger `k` values fix #1 now introduces** for early full-mode layers (e.g.
`k=1844`, matching the 3-level schedule's level-0 target for a 14,000-node graph) is where things
get tight: forward-only, the new implementation peaks lower than the old one (~7.4 GB vs ~8.0 GB),
but **both** implementations exhaust the 12 GB test GPU during `.backward()` at this exact
configuration. This isn't a regression from either version of the code — it reflects that at
this scale, the dominant memory cost is the `(batch, n, k)` assignment tensor `S` and its pooled
counterpart `AS`, and their backward-pass buffers, not the adjacency multiplication method. This
matches `ARCHITECTURE.md`'s existing note that full-mode training needs ≥24 GiB (the project's
RTX 3090 Ti), which the fix does not — and cannot, by itself — change.

**What this fix actually buys**:
- Removes reliance on the batched `bmm`-with-broadcast optimization being available/efficient
  across PyTorch versions and backends (the pinned production version is 1.12.1, not the 2.1.0
  tested here — behavior may differ, and the ~23 GiB figure in the docs may originate from
  exactly that version gap).
- Strictly avoids ever materializing a dense `(n, n)` matrix for the pooling step itself (only
  for `link_pred_loss`, unchanged).
- At the (more common) later/deeper levels, where `n` has already shrunk substantially, this is
  a clear, unambiguous efficiency win with no measured downside.
- At the large-`k`, large-`n` early-level regime that fix #1 introduces, it modestly reduces
  forward-pass peak memory but does **not** solve the backward-pass memory ceiling on its own.

### Still open

- The `(batch, n, k)` scale of `S`/`AS` (and their autograd buffers) — not the adjacency
  multiply — is the real ceiling for training Full DiffPool on the full ~14K-node graph with a
  wide early-level cluster budget. Reducing this further would need e.g. gradient checkpointing
  through the pooling layers, mixed precision, a smaller batch size for early wide layers, or a
  more conservative early-level cluster schedule than fix #1's default geometric interpolation.
- `pool_gnn`/`embed_gnn` shared `K`, missing pre-pooling encoder, no gradient clipping, no
  aux-loss warmup — same as listed under fix #1 below, still not addressed.

---

## 1. Progressive cluster-count schedule for Full DiffPool (2026-08-23)

### Problem

In `DiffPoolGNN` (full mode, `full_mode=True`), every `DiffPoolLayer` was constructed with the
same global `max_clusters` value (default 32) as its `pool_gnn` output width. Since
`DiffPoolLayer.forward()` bounds the pooled cluster count `k` by `pool_gnn.out_channels`
(`k = max(min_nodes, min(k_raw, self.pool_gnn.out_channels))`), and `k_raw` (derived from the
learned `pool_ratio`) is almost always far larger than 32 for any real node count, `k` was
effectively **always capped at `max_clusters` on the very first layer**, regardless of how many
levels (`n_levels`/`n_hybrid`) were configured.

Concretely: with the default full-mode config (`n_levels=2`, `max_clusters=32`), the graph was
pooled **14,000 → ≤32 → ≤32** instead of a gradual hierarchy. This is a single-hop ~440:1
compression with no structural prior to guide it — one of the failure modes identified in
`plan.md` §6.3 ("Assignment learning is ill-posed at scale") and `ARCHITECTURE.md`, and it made
"multiple levels" configs behave almost identically to a single-level config, since all the real
work was already done in one hop by layer 0.

### Fix

Added `_compute_cluster_schedule(n_start, n_final, levels)` in `src/pooling_genomic/models.py`,
which geometrically interpolates a per-level cluster-count target between the base graph's node
count (`n_start`) and the final `max_clusters` (`n_final`), strictly decreasing, always landing
exactly on `n_final` at the last level. Example for 14,000 nodes → 32 clusters:

| levels | schedule |
|---|---|
| 1 | `[32]` (unchanged — single hop, matches prior behavior) |
| 2 | `[669, 32]` |
| 3 | `[1844, 243, 32]` |
| 5 | `[4149, 1230, 364, 108, 32]` |

`DiffPoolGNN.__init__` now accepts an `n_nodes` argument (the base graph's node count) and, when
`full_mode=True` and there is more than one level, builds each `DiffPoolLayer`'s `pool_gnn` with
its own schedule-derived `max_clusters` instead of the single global value. The schedule is
stored on the module as `self.cluster_schedule` for introspection/debugging.

`build_diffpool_model()` now passes `n_nodes=base_graph.num_nodes` through to `DiffPoolGNN`.

**Hybrid mode is unaffected** — the condition guarding the schedule requires `full_mode=True`,
so hybrid mode's single trailing full-DiffPool layer (after the HEM-coarsened early levels)
still pools directly to `max_clusters` in one hop, matching the original documented design
("last level always pools to `max_clusters` nodes").

### Files changed

- `src/pooling_genomic/models.py`
  - Added `import math`.
  - Added `_compute_cluster_schedule()`.
  - `DiffPoolGNN.__init__`: added `n_nodes` param, per-layer cluster schedule, `self.cluster_schedule`.
  - `DiffPoolGNN` docstring updated to describe the schedule.
  - `build_diffpool_model()`: passes `n_nodes=base_graph.num_nodes` to `DiffPoolGNN`; docstring updated.

### Verification

No automated test suite exists in this repo (see `AGENTS.md`), so this was verified manually
with the `pooling_genomic` conda env:

1. Unit-checked `_compute_cluster_schedule()` directly for several `(n_start, n_final, levels)`
   combinations, including the `n_start <= n_final` edge case (returns `[n_final] * levels`).
2. Built a `DiffPoolGNN`/`build_diffpool_model()` model on a small synthetic graph
   (200 nodes, `full_mode=True`, `n_levels=3`, `max_clusters=8`) and confirmed:
   - `cluster_schedule == [68, 23, 8]`
   - each layer's `pool_gnn.out_channels` matches the schedule
   - a forward + backward pass runs cleanly end-to-end (output shape correct, gradients flow).
3. Confirmed hybrid mode (`full_mode=False`) is unchanged: `cluster_schedule == [max_clusters]`
   as before, forward pass still works.

### Not addressed by this change (still open, from the original code review)

These were identified in the same review but are **out of scope** for this change:

- Full mode materialized a dense `n × n` adjacency (`to_dense_adj` + batched `bmm`) for
  `Sᵀ A S` at every level — addressed by **fix #2** above (with caveats: it removes the dense
  `(n, n)` materialization, but the `(batch, n, k)` assignment-tensor scale this fix introduces
  at early levels is itself a separate memory ceiling that fix #2 only partly mitigates).
- `pool_gnn` and `embed_gnn` still share the same Chebyshev filter order `K`; a wider receptive
  field specifically for the assignment GNN was proposed but not implemented.
- No pre-pooling feature-enrichment encoder before the first `DiffPoolLayer`.
- No gradient clipping in `engines.py::train_epoch_clf`.
- No warmup/annealing schedule for `lambda_link_pred` / `lambda_entropy`.
