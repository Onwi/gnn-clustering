# Analysis of Pooling Approaches

## Context

- **Input**: RNA-seq gene expression profiles, 7709 samples × ~14K genes
- **Graph**: STRING-DB PPI network, ~14K nodes, ~8M edges (filtered to expressed genes)
- **Task**: 16-class TCGA cancer type classification
- **Data characteristic**: Gene expression features alone are highly informative — an MLP on raw 14K features achieves **95.5% test accuracy**

---

## Approach 1: Fixed HEM Coarsening — 96.05% test accuracy

### How it works

Heavy Edge Matching greedily contracts the graph by pairing nodes connected by the strongest interaction scores. Each level roughly halves the node count (14K → 7K → 3.5K → ... → 111 nodes in 8 levels). The hierarchy is **pre-computed once** and stored on disk.

At training time, each sample's 14K features are mapped to the 14K graph nodes. For each coarsening level, a ChebConv processes the graph, then `scatter(parents, sum)` pools nodes into their HEM-assigned super-nodes. The final flattened representation is fed to an MLP classifier.

### Why it succeeds

| Factor | Explanation |
|---|---|
| **Biologically meaningful clusters** | HEM pairs genes with the strongest STRING-DB interaction scores. These correspond to real protein complexes and pathways. The hierachy preserves the biological community structure of the genome. |
| **Deterministic, no learned parameters** | No clustering to learn — the hierarchy is fixed from the graph topology. The model can't discover "wrong" clusters. |
| **Sparse O(n) operations** | `scatter(parents, sum)` is O(n) and trivially parallelizable. The coarse edges are also sparse, so ChebConv remains O(\|E\|) throughout. |
| **Progressive channel growth** | Channels double at each level (1→2→4→...→32). Early levels (many nodes, few channels) do coarse feature extraction; late levels (few nodes, many channels) do fine-grained classification. |
| **Weighted pooling** | A learnable scalar per node, multiplied before scatter, allows the model to up/down-weight individual genes within each cluster. |
| **No competing losses** | Only cross-entropy on the classification task. The L1 regularization on node importances is optional and doesn't fight the main objective. |
| **Graph is a regularizer, not the signal** | Since expression features already contain the classification signal, the graph hierarchy acts as a **structured regularizer** — it encourages the model to group genes by biological function, reducing overfitting. |

### Results from our runs

```
n0_wpoolF_convsF (MLP only):        95.46%
n1_wpoolF_convsF:                    94.36%
n1_wpoolF_convsT (ChebConv):        95.66%
n1_wpoolT_convsF (weighted pool):    95.33%
n1_wpoolT_convsT (both):            96.05%  ← best
n2_wpoolF_convsF:                    94.10%
```

**Best**: 1 coarsening level + ChebConv + weighted pooling = **96.05%**

The MLP-only baseline (95.46%) nearly matches the best HEM config. This tells us the graph structure adds a small but consistent improvement (+0.6%) by regularizing the classifier, rather than providing novel discriminative signal.

---

## Approach 2: Full DiffPool — 27.7% test accuracy

### How it works

Every level learns a soft assignment matrix `S` via a separate `pool_gnn` (ChebConv). Features are pooled as `X' = S^T Z` and the adjacency as `A' = S^T A S`. Everything is learned from scratch — no pre-computed hierarchy. Two auxiliary losses are added: link prediction (reconstruct A from S) and entropy (encourage uniform assignments).

### Why it fails

#### 1. The 440:1 compression problem

Level 0 has 14,133 nodes and 1 channel per node. The `pool_gnn = ChebConv(1, 32, K=2)` must assign each node to one of 32 clusters. In one layer.

```
14,133 nodes × 1 feature → 32 clusters
```

This is an ill-posed learning problem:
- Each node has only a scalar expression value as its input feature
- The ChebConv has a receptive field of only K=2 hops (~200 neighbors on average in a graph with 8M edges and 14K nodes)
- The assignment must compress 440:1 — from 14K unique expression profiles to 32 groups
- Information theory says this is impossible without a strong prior

#### 2. No structural prior

HEM brings a strong biological prior: "genes with strong PPI interactions should stay together." Full DiffPool has no such prior — every assignment is learned from the classification gradient alone. The gradient through a soft assignment is noisy and high-variance, especially with 14K×32 = 450K assignment weights at level 0.

The model must simultaneously:
- Learn meaningful message-passing filters (ChebConv weights)
- Learn 450K cluster assignment weights  
- Learn the MLP classifier
- Balance 2 auxiliary losses against the main classification loss

#### 3. The dense adjacency is a memory and optimization trap

`A_dense = to_dense_adj(edge_index)` creates a `(14K, 14K)` = **200 million element** matrix.

```
A_dense memory: 0.74 GB (fp32)
S matrix:       0.03 GB (batch=16, k=32)
S^T A S:        creates intermediate tensors of similar size
Total level 0:  ~3-5 GB with gradients and optimizer states
```

But more importantly, `S^T A S` creates a **dense** pooled adjacency matrix. The original STRING-DB graph is sparse (~8M edges out of 200M possible). After `S^T A S`, every one of the 32 super-nodes is connected to every other super-node (32×32 = 1024 edges, all non-zero). This creates spurious connections — biological noise that propagates to deeper levels.

Level 1 then receives a fully-connected 32-node graph, even though the original biology only has ~8M meaningful interactions. The model is trying to learn from a corrupted graph structure.

#### 4. Competing auxiliary losses

```
loss = CrossEntropy + λ_link_pred × link_pred_loss + λ_entropy × entropy_loss
```

- **Link prediction loss**: Tries to reconstruct the 14K×14K adjacency matrix from a 32×32 pooled version. The pooled matrix has 1024 entries but the original has 200M — reconstruction is impossible. This loss provides a noisy gradient that fights the classification objective.
- **Entropy loss**: Encourages each node's assignment distribution to be uniform across all clusters. This actively fights the clustering objective (which wants assignments to be concentrated).
- **Classification loss**: The only signal for actual class discrimination.

The model must balance three competing objectives, and the auxiliary losses dominate early in training when the assignment matrix S is essentially random.

#### 5. Verification from experiments

```
Config          | Epochs | Test Accuracy
----------------|--------|-------------
full0 (k=32)    | 63     | 27.67%
full1 (k=32)    | 63     | 27.28%
full2 (k=32)    | 63     | 20.22%
full0 (k=256)   | 3      | 24.69%
```

- All configs hover near random (6.25% for 16 classes)
- More levels consistently **hurt** performance (20% vs 27%)
- Increasing to 256 clusters didn't help — the problem isn't the number of clusters, it's the lack of signal to learn them
- Training accuracy was also low (20-40%), indicating the model can't learn meaningful representations at any level

---

## Approach 3: Hybrid DiffPool — 70.3% best reported

### How it works

Early levels use HEM pre-computed coarse edges and `scatter(parents, mean)` pooling (identical to Fixed HEM). When the graph shrinks below `dense_threshold` (default: 500 nodes), it switches to full learned DiffPool for one final layer.

```
Level 0: 14K nodes, ChebConv(1→2), scatter(parents[0], mean) → 7K nodes
Level 1:  7K nodes, ChebConv(2→4), scatter(parents[1], mean) → 3.5K nodes  
  ... (hybrid levels continue until n ≤ 500)
Level N: ≤500 nodes, ChebConv embed, learned S, S^T Z, S^T A S → ≤32 nodes
  ↑ full DiffPool on small graph only
```

### Why it works better than Full DiffPool

| Aspect | Full DiffPool | Hybrid DiffPool |
|---|---|---|
| Level 0 graph | Dense 14K×14K (200M elements) | Sparse HEM coarse edges |
| Level 0 pooling | Learned S (14K×32) | `scatter(parents, mean)` — O(n) |
| Structural prior | None | Strong HEM hierarchy |
| Levels with aux loss | All levels (including 14K) | Only last (≤500 nodes) |
| Dense operations | On 14K×14K | On ≤500×500 (250K elements) |
| Pooled adjacency | S^T A S — always dense | Only on small graph — manageable |

The HEM pre-coarsening provides:
- **Biologically meaningful clusters** at no learning cost
- **Sparse operations** throughout the early levels
- **A warm-started small graph** for the learned DiffPool layer

The final learned layer operates on ≤500 nodes, where:
- `A_dense` is 500×500 = 250K elements (vs 200M for full mode)
- `S` is batch×500×32 = manageable
- `S^T A S` creates a 32×32 pooled adjacency — actually meaningful compression
- The link prediction loss can plausibly reconstruct a 500×500 graph from a 32×32 summary
- The entropy loss doesn't overwhelm because there are only 500 nodes to assign

### Results

- Best reported: **70.3%** (n_hybrid=2, lr=0.0099, wd=0.0344, 127 epochs)
- Our tuned run: **56.8% validation** (n_hybrid=2, lr=0.0097, wd=0.0002, 50 epochs)
- Default HPs: **~27%** (same as Full DiffPool)

Hybrid DiffPool improves substantially over Full DiffPool (70% vs 28%) but still underperforms Fixed HEM (96%). The learned DiffPool layer at the end still requires careful tuning, and the auxiliary losses add optimization difficulty. The structural prior from HEM only extends part-way; the final learned layer can still discover degenerate solutions.

---

## Comparison Summary

| Criterion | Fixed HEM | Hybrid DiffPool | Full DiffPool |
|---|---|---|---|
| **Test accuracy** | **96.05%** | 70.3% (tuned) / 56.8% (our tune) | 27.7% |
| **vs MLP baseline** | +0.6% (marginal gain) | −25.2% (hurts) | −67.8% (catastrophic) |
| **Structural prior** | Very strong (full HEM) | Strong (HEM early layers) | None |
| **Training speed** | ~1 hour for 6 configs | ~4+ hours for 12 tuned trials | ~1 hour per config |
| **GPU memory** | 9.6 GiB | 10.6 GiB (tuning with 3 workers) | 21.6 GiB |
| **Auxiliary losses** | None | Link_pred + entropy (last layer only) | Link_pred + entropy (every layer) |
| **Operations at level 0** | O(\|E\|) sparse scatter | O(\|E\|) sparse scatter | O(n²) dense 14K×14K |
| **Number of parameters learned** | Few (ChebConv + MLP) | Moderate (HEM levels free, ChebConv + MLP + last DiffPool) | Many (all assignments + ChebConvs + MLP) |
| **Pre-computation required** | 8 HEM level files | 8 HEM level files | None |
| **Key limitation** | Needs pre-computation | Needs pre-computation + still requires HP tuning | No biological prior, 440:1 compression impossible to learn |

---

## Conclusion

### Why Fixed HEM wins

The gene expression data contains the classification signal; the PPI graph provides a structured regularizer. HEM's heavy-edge matching preserves biologically meaningful gene communities (pathways, complexes), and the fixed hierarchy acts as a strong prior that prevents the model from learning spurious clusters. The MLP alone gets 95.5%; adding one HEM level + ChebConv + weighted pooling pushes this to 96.05% by regularizing the classifier. The graph helps, but marginally — the signal is in the expression levels, not the graph structure.

### Why Full DiffPool fails

Full DiffPool asks a ChebConv with 1-channel input and K=2 receptive field to partition 14K genes into 32 clusters from scratch. This is an impossible learning task. The dense 14K×14K operations create memory bottlenecks and corrupt the graph structure (`S^T A S` produces dense, biologically meaningless connections). The competing auxiliary losses (link prediction + entropy) fight the classification objective. The learned assignments at every level compound errors — deeper models perform worse than shallow ones. The approach ignores the key insight that the STRING-DB graph has genuine biological structure worth preserving.

### Why Hybrid DiffPool is intermediate

Hybrid DiffPool correctly uses HEM for early levels (sparse, biologically meaningful, free) and only applies learned pooling when the graph is small enough (≤500 nodes) for dense operations to be tractable and meaningful. This avoids the worst problems of Full DiffPool but still introduces the auxiliary losses and optimization difficulties of the final learned layer. The result (70%) is a significant improvement over Full DiffPool (28%) but still well below Fixed HEM (96%). The structural prior helps, but the learned component at the end can still discover degenerate solutions.
