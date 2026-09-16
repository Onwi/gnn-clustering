# Changes Made by Claude

Log of code changes made to this repository in collaboration with Claude, most recent first.
Each entry: what changed, why, files touched, and how it was verified.

---

## 6. Three findings from a `/code-review` pass on recent commits (2026-09-16)

Found by a code-review agent reviewing the last several commits, all independently verified
before fixing.

### 6a. ASHA pruning silently disabled again for `--n-cycles` 1 or 2

**Problem:** `run_holdout()`'s `grace_n_cycles = max(tune_n_cycles - 1, 1)` floors to 1 whenever
`tune_n_cycles` is already 1 (i.e. `--n-cycles` <= 2), making `grace_period == max_t` -- the exact
"ASHA never prunes early" no-op that this whole block's comment says it exists to avoid, silently
reintroduced via a different path than the original bug.

**Fix:** made the "no room for an earlier boundary" case explicit instead of falling through the
same `max(..., 1)` floor as the normal case: when `tune_n_cycles <= 1` (tuning budget is already a
single epoch, so there genuinely is no earlier restart boundary to prune at), skip `ASHAScheduler`
entirely and pass `scheduler=None` to `build_tuner`, with a printed warning explaining why -- so the
log says plainly that no pruning is happening at this budget, instead of constructing a scheduler
that looks active but isn't.

**Files changed:** `scripts/experiments/diffpool_experiment.py` (`run_holdout()`).

**Verification:** `python3 -m py_compile`; checked `tune_max_epochs`/`grace_period` for
`--n-cycles` in `{1, 2, 3, 5, 7}` -- `1`/`2` now hit the `scheduler=None` branch (previously both
computed `grace_period == tune_max_epochs == 1`, silently); `3`/`5`/`7` are unaffected
(`grace_period` 1/7/31 vs. `max_t` 3/15/63, matching pre-fix values exactly).

### 6b. `_config_key()` collided between Hybrid and Full-mode runs of the same pooling type

**Problem:** `scripts/analysis_v2/parsers.py`'s `_config_key()` builds `{prefix}_H{n_hybrid}_R{rep}`
using only `pooling_type`, `n_hybrid`, and `rep` -- never `full_mode`, even though the row-loading
code captures it. `diffpool_hybrid5_rep0` and `diffpool_full5_rep0` both key to `"DP_H5_R0"`;
`load_all_predictions`/`load_all_outputs` (dicts keyed by this) silently drop one run's data when
both are parsed together -- which is now a real scenario (`outputs/hybrid_n5_diffpool` alongside
`outputs/full_dmon_shrunk`).

**Fix:** added a mode letter (`"H"`/`"F"`) derived from `row["full_mode"]`, giving
`diffpool_hybrid5_rep0` -> `DP_H5_R0` and `diffpool_full5_rep0` -> `DP_F5_R0`.

**Files changed:** `scripts/analysis_v2/parsers.py` (`_config_key()`).

**Verification:** `python3 -m py_compile`; constructed synthetic hybrid/full rows with identical
`pooling_type`/`n_hybrid`/`rep` and confirmed `_config_key()` now returns distinct strings for them.

### 6c. `ARCHITECTURE.md` documented a `--collapse-regularization` flag that no longer exists

**Problem:** fix #5b (above) removed the `--collapse-regularization` CLI flag entirely, but
`ARCHITECTURE.md`'s DMoN section still told readers to tune it -- following that doc would hit an
argparse "unrecognized arguments" error.

**Fix:** corrected the sentence to describe only the two flags that actually exist
(`--lambda-modularity`/`--lambda-collapse`), with a brief note on why the removed flag was
redundant (matching the explanation already correct in `HYBRID_DMON.md` and this changelog's own
fix #5b, which the review agent didn't flag as stale).

**Files changed:** `ARCHITECTURE.md`.

**Verification:** grepped the repo's `.md`/`.MD` files for remaining `--collapse-regularization`
mentions -- only `changes-from-claude.md` (historical, correct) and `HYBRID_DMON.md` (already
correctly described as removed) remain; confirmed `--collapse-regularization` doesn't appear in
`diffpool_experiment.py --help` output.

---

## 5. Assignment-head dropout (from the DMoN paper) + remove redundant collapse_regularization (2026-09-15)

Found reading the actual DMoN paper (Tsitsulin, Palowitch, Perozzi, Muller, "Graph Clustering with
Graph Neural Networks", JMLR 2023, arXiv:2006.16904) end to end and cross-checking it against every
file on the DMoN path, ahead of the `n_hybrid=5` Hybrid DMoN run (`scripts/run_hybrid_n5_dmon_full3.sh`).
Two of several findings were judged worth acting on now; a third (the paper's fixed 1:1
modularity:collapse weighting vs. this codebase's independently-tuned `lambda_modularity`/
`lambda_collapse`) was deliberately left alone -- our own tuned configs already found ratios far
from 1:1 (e.g. one run: `lambda_modularity=0.028` vs `lambda_collapse=6.34`), and constraining the
search now, right before the matched DiffPool-vs-DMoN comparison, would both risk making the best
achievable config worse (removing flexibility this task may genuinely need, given it differs from
the paper's pure-unsupervised-clustering setting) and confound the n_hybrid=5 result with a second,
untested change. A fourth (matching the paper's trainable-skip-connection + SeLU GCN layer in place
of `ChebConv`+ReLU) was explicitly declined -- too large an architecture change for this pass.

### 5a. Added assignment-head dropout

**Problem:** the paper's assignment head is `C = softmax(GCN(...))` with **0.5 dropout on the GNN
representation before the softmax**, and states this specifically prevents "gradient descent from
getting stuck" in a degenerate assignment. Neither `DiffPoolLayer` nor `DMoNLayer` had any dropout
anywhere in the pooling path -- only the final `FCModel` classifier head had dropout.

**Fix:** added an `assign_dropout` parameter (default 0.5, matching the paper) to both layer
classes, applied via `F.dropout` to the raw assignment logits (`s_raw`/`c_raw`) immediately before
the softmax. Applied to **both** pooling types, not just DMoN, and to hybrid mode's trailing layer
too (it reaches the same learned-assignment branch) -- to avoid introducing a new architectural
asymmetry between DiffPool and DMoN that wasn't there before, consistent with this investigation's
running principle of keeping the two pooling types on identical architectural footing so any
difference in outcome isolates the loss family. Threaded through `DiffPoolGNN.__init__` (added to
`layer_extra_kwargs` unconditionally, not gated by `full_mode`, unlike `sparsify_density`) and
`build_diffpool_model()`, plus a new `--assign-dropout` CLI flag.

### 5b. Removed the redundant `collapse_regularization` parameter

**Problem:** already identified as a loose end in fix #4's era but not acted on then.
`collapse_regularization` (a fixed multiplier on `collapse_loss`, default 1.0) and the outer,
independently-tuned `lambda_collapse` both linearly scale the exact same term with nothing else
combining them -- `lambda_collapse * collapse_regularization * collapse_loss` is mathematically
just `(lambda_collapse * collapse_regularization) * collapse_loss`, a single effective scalar.
Tuning both adds no search expressiveness beyond what `lambda_collapse` alone already covers.

**Fix:** removed `collapse_regularization` entirely from `DMoNLayer`, `DiffPoolGNN.__init__`,
`build_diffpool_model()`, and the `--collapse-regularization` CLI flag. `DMoNLayer`'s
`collapse_loss` is now returned unscaled; `lambda_collapse` (already tuned) is the only remaining
knob on that term.

### Files changed

- `src/pooling_genomic/models.py`
  - `DiffPoolLayer`/`DMoNLayer`: added `assign_dropout` param + `F.dropout` call before softmax;
    removed `collapse_regularization` from `DMoNLayer` (param, `self.` attribute, and its multiply
    in the returned `aux` dict).
  - `DiffPoolGNN.__init__`, `build_diffpool_model()`: added `assign_dropout` param (unconditional,
    both modes); removed `collapse_regularization` param; docstrings updated.
- `scripts/experiments/diffpool_experiment.py`: added `--assign-dropout` CLI flag (default 0.5),
  threaded into both `build_diffpool_model(...)` call sites; removed `--collapse-regularization`
  and its two pass-throughs.

### Verification

Using the `pooling_genomic` conda env:

1. `python3 -m py_compile` on both changed files; confirmed `--collapse-regularization` no longer
   appears in `--help` output and no remaining code references to `collapse_regularization` outside
   an explanatory docstring note.
2. End-to-end, all 4 combinations of `pooling_type in {diffpool, dmon}` x `full_mode in {True,
   False}` on a synthetic 200-node graph: `model.train()` forward + backward produces no NaN
   gradients; `model.eval()` forward is fully deterministic across repeated calls on the same
   input (confirms dropout correctly disables during evaluation, so test/final metrics aren't
   corrupted by dropout noise).

### Not addressed by this change (deliberately, see above)

- The paper's fixed 1:1 modularity:collapse weighting vs. this codebase's independently-tuned
  `lambda_modularity`/`lambda_collapse` -- left as a candidate follow-up ablation with spare
  compute, not bundled into the upcoming `n_hybrid=5` Hybrid DMoN comparison.
- The paper's trainable-skip-connection + SeLU GCN layer, in place of the current `ChebConv` + ReLU
  -- explicitly declined as too large an architecture change for this pass.

---

## 4. Three small correctness/cleanliness fixes found reviewing Full DMoN and adjacent code (2026-09-10)

Found while re-reading the full Full DMoN implementation end to end, looking for further
improvements after the shrunk-scope run reached 71.16%. Three independent, low-risk fixes;
a fourth, larger finding (Hybrid mode's trailing learned layer operating on ~3,534 nodes, not
the documented <=500, since `n_hybrid` is a level count matched against pre-computed HEM files,
not a dynamic node-count check) is a behavioral question left for a separate experiment, not
bundled into this entry.

### 4a. Removed the dead `logit_pool_ratio` parameter

**Problem:** `DiffPoolLayer`/`DMoNLayer` each had a `logit_pool_ratio` `nn.Parameter`, exposed via
a `pool_ratio` property (`sigmoid(logit_pool_ratio)`), used by `_compute_pool_k` to compute
`k_raw = ceil(n * pool_ratio)` before clamping to `max_clusters`. `torch.ceil` has zero gradient
almost everywhere, so `logit_pool_ratio` never received a gradient and stayed frozen at its init
value (`sigmoid(0) = 0.5`) for the entire life of training, despite being registered as a trainable
parameter and included in the optimizer's param groups. Worse, at that permanently-fixed ratio,
`k_raw` was always far larger than `max_clusters` in every configuration this codebase actually
runs (every level's node count shrinks by more than 2x, `max_clusters` being the smaller side), so
the clamp to `max_clusters` was binding every single time regardless -- the "learnable pooling
ratio" had zero effect on the model's actual behavior, full stop.

**Fix:** Removed `logit_pool_ratio`/`pool_ratio` entirely from both layer classes. `_compute_pool_k`
now just takes `(n, min_nodes, max_clusters)` and returns `max_clusters` clamped to
`[min_nodes, n]` -- exactly what the old code always computed in practice, made explicit instead of
routed through a parameter that looked learnable but wasn't.

### 4b. Removed the dead `dense_threshold` parameter

**Problem:** `DiffPoolGNN`/`build_diffpool_model`/the CLI (`--dense-threshold`, default 500) all
carried a `dense_threshold` parameter documented as "node count below which hybrid mode switches to
full mode." Grepped the whole codebase: it was never read anywhere in the model-building logic,
only threaded through signatures. The real switch point is `i < n_hybrid` in `DiffPoolGNN.__init__`
-- a fixed *level count* against pre-computed HEM files, unrelated to `dense_threshold`'s value. A
user passing `--dense-threshold 200` (or any value) saw zero effect on the model.

**Fix:** Removed `dense_threshold` from `DiffPoolGNN.__init__`, `build_diffpool_model`, and the CLI.
Corrected `DiffPoolGNN`'s docstring to describe what actually controls the switch (`n_hybrid`, a
level count) and to flag that the trailing layer's real node count depends on HEM's own per-level
reduction rate, not a fixed target -- e.g. `n_hybrid=2` leaves ~3,534 nodes on the real 14,133-node
PPI graph, not the "<=500" figure the rest of the docs (plan.md, ARCHITECTURE.md) assume.

### 4c. Corrected DiffPool's link-prediction loss to be a true per-sample loss

**Problem:** `DiffPoolLayer`'s link-prediction loss averaged the assignment matrix `S` across the
*batch* (`S_mean = S.mean(dim=0)`) before computing `SSᵀ` and comparing it against the (shared)
adjacency `A`. Batch samples are different patients with different node features/assignments;
averaging `S` before the quadratic term collapses that per-sample structure into one
"population-average" assignment's self-similarity, which is a materially different (and weaker)
target than a true per-sample reconstruction loss. `DMoNLayer`'s modularity loss doesn't have this
issue -- it's computed per-sample from `A_next_dense` (itself batched) and averaged only at the end.

**Fix:** Rewrote the loss as an exact per-sample `MSE(A/‖A‖_F, SSᵀ/‖SSᵀ‖_F)`, averaged over the
batch, using an algebraic identity instead of ever materializing a dense `(n, n)` matrix per sample
(infeasible at this codebase's scale -- up to 14,133 nodes at full-mode level 0):
- `⟨A, SSᵀ⟩_F = trace(SᵀAS)` (cyclic trace), which is exactly `A_next_dense`'s diagonal, already
  computed by `_pool_adjacency` for the output graph -- reused rather than recomputed.
- `‖SSᵀ‖_F = ‖SᵀS‖_F` for any `S`, since `SSᵀ` is symmetric: `‖SSᵀ‖_F² = trace((SSᵀ)²) =
  trace((SᵀS)²) = ‖SᵀS‖_F²`. `SᵀS` is only `(k, k)`, cheap to form per sample via one `bmm`.
- `‖A‖_F = sqrt(Σ edge_weight²)`, computed directly from the sparse edge list -- no dense copy.

This also removes the `to_dense_adj` (n, n) materialization from `DiffPoolLayer.forward()`
entirely (it was only ever used for this loss), and reorders `_pool_adjacency` to run before the
loss instead of after, so its result is shared between the loss and the output-graph construction
rather than computed twice.

### Files changed

- `src/pooling_genomic/models.py`
  - `_compute_pool_k()`: dropped the `pool_ratio` parameter, simplified to `max(min_nodes,
    min(max_clusters, n))`.
  - `DiffPoolLayer`/`DMoNLayer`: removed `logit_pool_ratio`/`pool_ratio`; `DiffPoolLayer.forward()`
    rewritten (link-prediction loss, `_pool_adjacency` reordered before the aux-loss block, dense
    `to_dense_adj` call removed).
  - `DiffPoolGNN.__init__`, `build_diffpool_model()`: removed `dense_threshold` parameter;
    docstrings corrected.
  - Removed the now-unused `to_dense_adj` import.
- `scripts/experiments/diffpool_experiment.py`: removed `--dense-threshold` CLI flag and both
  `dense_threshold=args.dense_threshold` pass-throughs.

### Verification

Using the `pooling_genomic` conda env:

1. `_compute_pool_k`: checked directly against the values it needs to reproduce
   (`_compute_pool_k(14133, 2, 1854) == 1854`, `_compute_pool_k(243, 2, 32) == 32`,
   `_compute_pool_k(5, 2, 32) == 5`, correctly clamped by `n`).
2. Link-prediction loss: on a small synthetic graph (n=40, k=6, batch=5, de-duplicated edges),
   compared the new trace-identity formula against a brute-force per-sample dense `SSᵀ` reference
   computed independently -- matched to `5.8e-11` absolute difference (float32 precision), both in
   the batch-mean and every individual per-sample value.
3. End-to-end: `build_diffpool_model(...)` for all 4 combinations of `pooling_type in
   {diffpool, dmon}` x `full_mode in {True, False}` on a synthetic 200-node graph -- each builds,
   runs forward + backward, and produces no NaN gradients.
4. `python3 -m py_compile` on both changed files; confirmed `--dense-threshold` no longer appears
   in `--help` output and no other file in the repo references `dense_threshold`, `pool_ratio`, or
   `logit_pool_ratio`.

### Not addressed by this change (separate, larger question)

- Whether re-running the Hybrid comparison with `n_hybrid=5` (landing the trailing learned layer at
  ~442 nodes, close to the long-documented "<=500" design point, instead of the ~3,534 nodes
  `n_hybrid=2` actually produces) changes the Hybrid vs. Full DiffPool/DMoN comparison -- this is a
  behavioral experiment, not a correctness fix, and wasn't bundled into this entry.

---

## 3. Pooled-adjacency sparsification for full-mode pooling (2026-09-07)

### Problem

`_build_pooled_output_graph()` (both `DiffPoolLayer` and `DMoNLayer`'s shared full-mode output
step) means the batch's pooled adjacency, drops the diagonal, and calls PyG's `dense_to_sparse`.
But `dense_to_sparse` keeps every *numerically nonzero* entry as an edge, and since the pooled
adjacency is built from softmax assignments (`S`/`C`), essentially no entry is ever exactly zero.
So every full-mode level returns a **fully connected** `k x k` graph to the next level -- "sparse"
only in tensor representation, not in structure. This is failure mode #3 from `analysis-approaches.MD`
("the dense adjacency is a memory and optimization trap"): the original sparse, biologically
meaningful PPI graph gets replaced by spurious full connectivity at every hop, and that corrupted
structure is what the next level's `ChebConv` message-passing has to work with. Fix #1 (the
progressive cluster schedule, entry #1 below) reduces *how much* compression happens per hop, but
doesn't touch this -- every hop, regardless of schedule, still produces a fully connected output.

This surfaced while scoping a **Full DMoN** run (multi-level full-mode pooling with
`pooling_type='dmon'`, not yet attempted in this repo -- prior DMoN runs were all hybrid-mode,
HEM-coarsened down to <=500 nodes before one learned pooling hop). In a multi-level full-mode
chain the corruption compounds level over level, since each level's fully-connected output becomes
the next level's input graph; in hybrid mode's single trailing full-mode layer this was harmless in
practice, since nothing downstream consumes that layer's output graph (the classifier just flattens
its pooled features).

### Fix

`_build_pooled_output_graph()` now accepts an optional `sparsify_density`. When given, it computes
`top_k = max(1, round(sparsify_density * (k - 1)))` for that level's own width `k`, keeps only the
`top_k` strongest edges per row of the (diagonal-dropped) pooled adjacency, then ORs the resulting
mask with its transpose (an edge survives if *either* endpoint ranked it in its own top-k -- per-row
top-k masks aren't symmetric in general even though the adjacency itself is) before converting to
sparse. `None` (default) preserves the old fully-connected behavior.

`sparsify_density` is a **fraction of each level's own width**, not a fixed edge count -- an earlier
version of this fix used a fixed `sparsify_top_k` shared across all levels, but full-mode levels in
the 3-level schedule span very different widths (e.g. 1854 down to 32), and a single fixed count
can't match the same relative density at more than one of them (a count sized for 1854 columns is
tiny relative to 243 columns, or vice versa). The fraction is meant to match the base PPI graph's
*actual* density rather than an arbitrary constant: `stringdb_top100pc.csv` has ~11.9M edge rows
over 19,385 nodes, i.e. avg degree ~1232 out of ~19,385 possible -- ~4% density (average degree /
number of nodes, not edges/possible-pairs, since the CSV lists each undirected edge as a symmetric
directed pair). At `sparsify_density=0.04`, the 3-level schedule's per-level top-k works out to ~74
edges/node at level 0 (k=1854) and ~10 at level 1 (k=243, matching a level 0->1 hop rather than an
absolute edge count).

Threaded through as a constructor argument: `DiffPoolLayer`/`DMoNLayer` -> `DiffPoolGNN` (only
forwarded into `layer_extra_kwargs` when `full_mode=True`, so hybrid mode's trailing full-mode
layer -- which already ignores its own output graph -- is unaffected) -> `build_diffpool_model()`
-> new `--sparsify-density` CLI flag in `scripts/experiments/diffpool_experiment.py`.

This only fixes the *structural* corruption (full connectivity), not the compression-ratio problem
(that's fix #1) or the auxiliary-loss-dominance problem (still open, see below) -- deliberately
scoped to just these two, since a Full DMoN run's open question is specifically whether DMoN's
modularity/collapse losses behave differently from DiffPool's link-pred/entropy losses once the
architecture (schedule + real sparsity) stops confounding the comparison.

### Files changed

- `src/pooling_genomic/models.py`
  - `_build_pooled_output_graph()`: added `sparsify_density` param and the per-level top-k-per-row +
    symmetrize pruning logic (`top_k` derived from `sparsify_density * (k - 1)`).
  - `DiffPoolLayer.__init__`/`forward()`, `DMoNLayer.__init__`/`forward()`: added `sparsify_density`
    param, stored on `self`, passed through to `_build_pooled_output_graph()`.
  - `DiffPoolGNN.__init__`: added `sparsify_density` param, forwarded into `layer_extra_kwargs` only
    when `full_mode=True`.
  - `build_diffpool_model()`: added `sparsify_density` param, passed to `DiffPoolGNN`.
- `scripts/experiments/diffpool_experiment.py`: added `--sparsify-density` CLI flag, threaded into
  both `build_diffpool_model(...)` call sites (initial tuning-phase model and final-retrain model).

### Verification

Using the `pooling_genomic` conda env:

1. **Density measurement**: computed the base graph's actual degree stats directly from
   `data/string_data/data/networks/stringdb_top100pc.csv` (19,385 nodes, 11,938,498 edge rows, mean
   degree 1231.7, median 978) to ground `sparsify_density` in the real graph rather than a guess.
2. **Structural**: on synthetic small graphs, `sparsify_density=None` returns the full `k*(k-1)`
   non-diagonal edges every time (confirms the corruption is real in the current code, for both
   `DiffPoolLayer` and `DMoNLayer`); `sparsify_density=0.04` prunes this substantially (e.g. k=25:
   600 -> 48 edges). Directly confirmed the per-level top-k formula against the real 3-level
   schedule's widths: `k=1854 -> top_k=74`, `k=243 -> top_k=10`, `k=32 -> top_k=1` (this last level's
   output isn't consumed downstream, so its value doesn't matter in practice).
3. **Gradients**: backward through the sparsified output graph's edge weights plus each layer's
   auxiliary losses (`link_pred_loss`/`entropy_loss` for DiffPool, `modularity_loss`/`collapse_loss`
   for DMoN) still produces valid `pool_gnn` gradients for both pooling types.
4. **End-to-end**: `build_diffpool_model(..., full_mode=True, n_levels=3, sparsify_density=0.04)` on
   a synthetic 200-node graph builds, confirms `cluster_schedule == [68, 23, 8]` (fix #1 unaffected),
   and runs a full forward + backward pass for both `pooling_type='diffpool'` and `'dmon'`.
5. `python3 -m py_compile` on both changed files.

### Not addressed by this change (still open)

- **Auxiliary-loss dominance** (failure mode #4 in `analysis-approaches.MD`): whether DMoN's
  `lambda_modularity`/`lambda_collapse` need different tuning ranges at 14K-node full-mode scale
  than the <=500-node hybrid-mode regime they've only been tuned in so far. Deliberately left to
  the normal hyperparameter search rather than special-cased.
- No gradient clipping in `engines.py::train_epoch_clf` (still listed as open under fix #1/#2).
- A genuine **Full DMoN run has not yet been executed** -- this entry is architecture/code
  preparation for one, not a result.

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
