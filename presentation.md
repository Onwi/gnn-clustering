# Pooling GNNs for Genomic Data Classification

## Hierarchical Graph Pooling Approaches for Cancer Type Classification from Gene Expression Data

---

## Slide 1: Problem Statement

The goal of this work is to classify cancer types from gene expression data. Given a patient's RNA-seq profile — the activity levels of thousands of genes — we want to predict which of 16 TCGA cancer types they have.

This is a high-dimensional problem. Each sample has about 14,000 gene expression values, but we only have around 7,700 samples. With more features than samples, a model can easily overfit by latching onto spurious correlations. The challenge is to find structure that generalizes.

Our key intuition is that genes do not work in isolation. They interact through complex biological pathways — proteins bind to other proteins, regulate each other, and collaborate to drive cellular functions. This interaction structure is well-studied and publicly available through databases like STRING-DB, which catalog millions of known protein-protein interactions.

Our idea is to use this interaction network as a graph prior: we place each gene as a node in a graph, connect genes that physically interact, and then apply hierarchical graph pooling techniques to learn representations that respect this structure. The question is whether incorporating this prior knowledge helps or hurts classification compared to treating each gene as an independent feature.

---

## Slide 2: Input Data

We work with RNA-seq data from The Cancer Genome Atlas (TCGA), accessed through the Xena platform. The raw data is upper-quartile FPKM — a normalized measure of gene expression — which we log-transform so that the features follow an approximately Gaussian distribution, making them more suitable for neural network training.

From the 33 available TCGA cohorts, we selected 16 that had enough samples (at least 10 tumor and 10 normal tissue each), ensuring we could split into training, validation, and test sets meaningfully. We also removed genes and samples with more than 20% missing values to avoid imputation artifacts.

To connect gene expression with protein interactions, we mapped each ENSG identifier from the TCGA data to its best-matching ENSP identifier using the STRING API. This mapping reduced the feature space from roughly 60,000 transcripts to 14,133 protein-coding genes — a substantial dimensionality reduction that also links each gene directly to its place in the protein interaction network.

The graph itself comes from STRING-DB v11.5 (Homo sapiens). It contains 14,133 nodes (one per mapped gene) connected by approximately 8 million edges. Edge weights represent combined confidence scores from STRING, normalized to [0, 1]. We removed 15 singleton nodes that had no connections, leaving a single large connected component. Importantly, this graph structure is fixed — it is the same for every sample. Only the node features (the expression values) change from one sample to the next.

For hierarchical pooling approaches, we pre-compute 8 levels of graph coarsening using Heavy Edge Matching:

| Level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Nodes | 14K | 7K | 3.5K | 1.8K | 884 | 442 | 221 | 111 |

Each level roughly halves the number of nodes by merging functionally related genes into super-nodes. These pre-computed levels are only needed by the Fixed HEM and Hybrid DiffPool models; the Full DiffPool learns its own hierarchy from scratch and ignores them entirely.

---

## Slide 3: Why Graph Coarsening?

Before we discuss specific approaches, it is worth asking why we need graph coarsening at all. There are three reasons.

First, computational cost. A graph neural network on 14,000 nodes with 8 million edges is expensive. Every message-passing step propagates information along every edge, which means each forward pass touches millions of connections. Doing this repeatedly over hundreds of training epochs adds up quickly. Coarsening reduces the graph size aggressively — by a factor of 2 at each level — so later levels are dramatically cheaper.

Second, biological plausibility. Biological systems are inherently hierarchical. Pathways contain sub-pathways, which contain individual protein complexes, which contain individual genes. A flat graph with 14,000 nodes ignores this structure. Coarsening mirrors the hierarchical organization of biology: nearby genes in the interaction graph tend to participate in shared functions, so merging them into super-nodes is a natural way to aggregate information.

Third, representation learning. At the coarsest level, each super-node aggregates information from many genes, producing a compact, high-level representation of the sample. This is analogous to how convolutional networks pool pixels into increasingly abstract features — the hierarchy builds a multi-scale description of the input.

This leads to a central design question: should the coarsening hierarchy be fixed in advance, or should it be learned jointly with the classifier? The first approach is faster and simpler; the second is more flexible but harder to train. We explore both.

---

## Slide 4: Heavy Edge Matching (HEM) — The Algorithm

For the fixed coarsening approach, we use Heavy Edge Matching, a classic algorithm from the graph partitioning literature (adapted from the GraCLUS algorithm used in multilevel mesh partitioning).

The intuition is straightforward. We want to pair up nodes that are connected by heavy edges — edges with high STRING-DB confidence scores — because a heavy edge suggests a strong functional relationship between those two genes. By merging them into a single super-node, we preserve the most important structural information while reducing the graph size.

The algorithm proceeds level by level. At each level, we first sort nodes by degree so that highly-connected nodes get processed first. Then we iterate over each unmarked node and look at its neighbors to find the one with the heaviest edge weight. If that neighbor exists and is also unmarked, we pair them together — both are assigned to the same new cluster. If no suitable neighbor exists, the node forms its own cluster as a singleton. This greedy matching produces a set of clusters that roughly halves the number of nodes at each level.

Once the clusters are determined, we construct the coarser graph. The new edge weights between super-nodes are computed as the sum of the original edges connecting their constituent genes. We also record a parent mapping that tells us, for each original node, which super-node it belongs to — this is what we use later for pooling.

The result is a hierarchy of graphs: starting from 14,000 nodes, we get levels with approximately 7,000, 3,500, 1,800, and so on down to 111 nodes at the coarsest level. Each level preserves the connectivity structure of the original graph while progressively merging related nodes.

---

## Slide 5: HEM Pre-computation Pipeline

The HEM hierarchy is computed once, before any training happens, and saved to disk for reuse across experiments. This is important because the coarsening itself is deterministic and does not depend on the model or the labels — it only depends on the graph structure.

The pre-computation script works as follows. First, it loads the full gene list from the TCGA dataset and the STRING-DB network, filtering the network to only keep genes that actually appear in our expression data. This step ensures that every node in the graph corresponds to a gene we have measurements for. The filtered network is then converted into a sparse adjacency matrix.

This adjacency matrix is fed into the HEM algorithm, which produces 8 successively coarser graphs along with their parent mappings. Each coarser graph represents a different resolution of the original interaction network, from the full 14,000-gene level down to a compact 111-super-node level.

Each level is saved as three tensor files: the edge indices (which nodes are connected), the edge weights (the normalized connection strengths), and the parents tensor (which maps each node at this level to its cluster at the next level). In total, 24 files are generated. These are loaded at training time by the Fixed HEM and Hybrid DiffPool models. The Full DiffPool model ignores them entirely, since it learns its own hierarchy from scratch.

---

## Slide 6: Fixed HEM — Model Architecture

With the hierarchy pre-computed, the model itself is surprisingly simple. At training time, we load the pre-computed graph levels and build a sequential pooling architecture.

Each sample starts as a flat vector of 14,000 gene expression values. We reshape this into a 2D tensor of shape (14,000, 1) — treating each gene as a node with a 1-dimensional feature (its expression level). This is passed through a series of coarsening levels.

At each level, we optionally apply a Chebyshev convolution (ChebConv with K=2) followed by a ReLU activation. The ChebConv lets each node aggregate information from its neighbors within a 2-hop radius, smoothing the expression values across the graph. Then we use the HEM parent mapping to pool nodes into clusters using scatter addition: all nodes belonging to the same cluster have their features summed to produce the feature of the super-node at the next level.

As we move through the hierarchy, the number of channels grows progressively: 1 channel at the first level, then 2, 4, 8, and so on up to a maximum of 32. The intuition is that as the graph gets smaller, each super-node should have a richer representation — it needs more dimensions to encode the information from the many genes it aggregates.

There is also an optional weighted pooling mechanism where the model learns a scalar importance weight for each node. Before pooling, each node's features are multiplied by its learned weight, allowing the model to downweight noisy or irrelevant genes. This is regularized with an L1 penalty to encourage sparsity — the model should learn to ignore most genes and focus on a few important ones.

At the end of the hierarchy, we flatten the remaining nodes into a single vector and pass it through a standard fully-connected classifier: a hidden layer of 256 units with batch normalization and dropout, followed by a linear layer that outputs logits for the 16 cancer types.

---

## Slide 7: Fixed HEM — How Pooling Works

To make the pooling operation concrete, consider what happens at the first level. The HEM algorithm produces a parents tensor for level 0: an array of length 14,000 where each entry is a cluster ID between 0 and 6,999. Nodes 0 and 1 might both have cluster ID 0, meaning they are paired together into the same super-node. Nodes 2 and 3 belong to cluster 1, and so on. Singleton nodes form their own cluster.

When we pool, we use a scatter operation. For each cluster, we gather all the node features assigned to it and sum them together. The result is a new tensor where each row corresponds to a super-node and its feature is the sum of its constituent genes' features. This reduces the graph from 14,000 nodes to 7,000 nodes in one step.

We chose sum pooling rather than mean pooling for a specific reason. Summing preserves the total signal magnitude — if two genes are both highly expressed, their super-node will have a high value. With mean pooling, a cluster containing one highly-expressed gene and one lowly-expressed gene would produce a moderate value, potentially drowning out the strong signal. In cancer classification, it is often the presence of over-expressed oncogenes or under-expressed tumor suppressors that matters, and sum pooling preserves these extremes more faithfully.

This same operation repeats at each level, using the parents tensor for that level to determine how nodes merge into the next coarser graph.

---

## Slide 8: Fixed HEM — Results

The results of the Fixed HEM experiments are striking, and perhaps counter-intuitive. The best performing configuration is not a hierarchical model at all: it is a simple MLP applied directly to the raw 14,000 gene expression values, with no graph structure and no pooling. This achieves 92.3% test accuracy.

Every configuration that uses graph coarsening performs worse. With one level of pooling (14,000 to 7,000 nodes), accuracy drops to 86.8%. With two levels and convolutional filters, it drops further to 86.6%. With three levels, 85.1%. And at the full 7-level hierarchy all the way down to 111 nodes, accuracy falls to around 80%.

This pattern is consistent and unambiguous: adding graph structure hurts performance. The more we coarsen, the worse the model becomes.

We have three hypotheses for why this happens. The first is that pooling discards discriminative signal. If a gene that is highly predictive of cancer type gets merged with a neighbor that is not, the model can no longer distinguish that gene's contribution. The second is that HEM clusters genes by physical binding affinity — which is what STRING-DB measures — not by classification relevance. Two proteins that physically interact may have completely different expression patterns across cancer types. The third is simply that the MLP is already very effective: with 14,000 parameters in its first layer, it has enough capacity to learn which genes matter independently. The graph structure constrains this flexibility rather than enhancing it.

This result sets the stage for our next question: if a fixed hierarchy hurts, can a learned hierarchy do better?

---

## Slide 9: Hybrid DiffPool (Learned Pooling)

The Fixed HEM results suggest that the HEM hierarchy, while structurally meaningful, may be suboptimal for classification. This motivates our second approach: differentiable pooling, or DiffPool, where the clustering is learned jointly with the classifier rather than fixed in advance.

The idea behind DiffPool is simple. Instead of using HEM's fixed parent assignments to pool nodes, we learn a soft assignment matrix S at each level. Each row of S corresponds to a node, each column to a cluster, and the entries indicate how strongly that node belongs to that cluster. By making S a differentiable function of the node features, the model can learn which genes should be grouped together for the purpose of classification.

However, there is a practical problem. At the first level, S would be a 14,000 by 7,000 matrix — nearly 100 million entries per sample — and computing S^T @ A @ S requires materializing a dense 14,000 by 14,000 adjacency. This is prohibitively expensive: approximately 23 gigabytes of memory for a single batch, causing out-of-memory errors on standard GPUs.

To work around this, we use a hybrid approach. The early levels — where the graph is large — use the fixed HEM parents for pooling via scatter, exactly like the Fixed HEM model. This is efficient because scatter is O(n) in the number of nodes and requires no dense matrices. Only the last level, where the graph has been coarsened to a manageable size, uses the full learned DiffPool with a soft assignment matrix and learned adjacency pooling.

Specifically, with n_hybrid=2, the first two levels use HEM scatter pooling to reduce from 14,000 to 7,000 to 3,500 nodes. The third and final level then learns a soft assignment that pools from 3,500 down to at most 32 super-nodes, including auxiliary losses for link prediction and entropy regularization to encourage meaningful clustering.

---

## Slide 10: Hybrid DiffPool — The OOM Problem

To make the memory issue concrete, consider what happens if we try to apply full DiffPool at the first level. The assignment matrix S is computed by a separate pooling GNN that outputs a score for each node-cluster pair. With 14,000 nodes and 7,000 target clusters, this produces a 14,000 by 7,000 matrix per sample — nearly 100 million floating point values. With a batch size of 16, that is 1.6 billion values just for S.

The real problem comes when we pool the adjacency. DiffPool pools the adjacency as S^T @ A @ S, where A is the original 14,000 by 14,000 dense adjacency matrix. Even though the original graph is sparse (8 million edges out of 196 million possible), converting it to dense form materializes all 196 million entries. The subsequent matrix multiplications add more intermediates on top.

In total, the first DiffPool layer alone consumes approximately 23 gigabytes of GPU memory for a batch of 16. On a 24-gigabyte GPU, this causes an immediate out-of-memory error — there is no room for the rest of the model, the optimizer states, or even the data itself.

The hybrid approach solves this by using HEM scatter pooling for the first two levels, which are O(n) in both time and memory. No dense adjacency is ever materialized. The learned DiffPool is only applied at the last level, where the graph has been reduced to at most 3,500 nodes — small enough that the dense operations fit comfortably within memory.

---

## Slide 11: Hybrid DiffPool — Training Dynamics

Training the Hybrid DiffPool model revealed an interesting dynamic that is worth examining closely.

We use cosine annealing warm restarts as our learning rate scheduler, with an initial period of 1 epoch that doubles after each restart (T_0=1, T_mult=2). This means the first cycle is just 1 epoch, the second cycle is 2 epochs, then 4, 8, and so on. The total training length depends on the number of cycles: with 7 cycles, we get 127 epochs.

The validation accuracy across these cycles shows a clear pattern. In the first cycle — just one epoch — the model reaches 48% validation accuracy. When the learning rate resets for the second cycle, accuracy drops — the large learning rate disrupts the parameters — but it recovers to 54% by the end of the cycle. The third cycle reaches 56%. By the fourth cycle, with 8 epochs, accuracy peaks at 72%.

This staircase pattern is typical of warm restarts: each reset temporarily hurts performance, but the model ultimately reaches a higher plateau each time. The important lesson is that training for only a few cycles produces a misleadingly low result. With 1 cycle, we would report 48%. With 4 cycles, we get 72%. Understanding this dynamic required running the full 7 cycles.

Another critical finding was that the default hyperparameters — a learning rate of 0.05 and auxiliary loss weights of 0.001 — were severely over-regularizing the model. The auxiliary losses for link prediction and entropy, meant to encourage meaningful cluster assignments, were dominating the classifier loss and preventing the model from learning to classify. Tuning reduced these weights by an order of magnitude.

---

## Slide 12: Hybrid DiffPool — Hyperparameter Tuning

We used Ray Tune with an ASHA scheduler to search for better hyperparameters, running 8 trials over 31 epochs each. The search space covered the learning rate, weight decay, and the two auxiliary loss weights.

The default hyperparameters were a learning rate of 0.05, weight decay of 0.01, and auxiliary loss weights of 0.001 for both link prediction and entropy. The tuned values tell an instructive story.

The learning rate was reduced from 0.05 to 0.0099 — about a factor of 5 lower. The original rate was too aggressive, causing the optimization to skip over good minima. Weight decay actually increased from 0.01 to 0.0344, suggesting that the model benefits from stronger L2 regularization to prevent overfitting on the 14,000-dimensional input.

The most dramatic change was in the auxiliary loss weights. The link prediction weight dropped from 0.001 to 0.000166, and the entropy weight dropped to 0.0000198 — both roughly 10 to 50 times lower than the defaults. This tells us that the original auxiliary losses were far too strong. The link prediction loss, which tries to make S^T @ S approximate the original adjacency, was forcing the model to preserve graph structure at the expense of classification performance. The entropy loss, which encourages soft assignments to be deterministic, was similarly overwhelming the classifier gradient.

The practical takeaway is that the classifier loss must dominate. The auxiliary losses are regularizers — they shape the clustering but should not compete with the primary objective. Setting them too high prevents the model from learning anything useful about cancer types.

---

## Slide 13: Hybrid DiffPool — Results

After hyperparameter tuning and extended training, the Hybrid DiffPool model reached 70.3% test accuracy with the tuned hyperparameters and 127 epochs (7 cycles). This is a substantial improvement over the default hyperparameter configuration, which barely exceeded 30% — the tuning alone was worth approximately 40 percentage points.

However, 70.3% still falls well short of the 92.3% achieved by the simple MLP baseline. The gap is roughly 22 percentage points, and it persists even with extended training. Running for more epochs helps — accuracy rises from 59.5% at 31 epochs to 70.3% at 127 epochs — but the improvement slows, and there is no indication that further training would close the gap entirely.

Why does the learned pooling still underperform? One possibility is that the hybrid nature of the model — using HEM for the early levels — constrains the clustering in ways that the final learned level cannot compensate for. The early levels aggressively pool genes using HEM's fixed assignments, potentially discarding information that the last level never gets back. Another possibility is that the pooled 32-dimensional representation is simply not rich enough to capture the information present in the 14,000-dimensional original, even after transformation through ChebConv layers.

This result motivated our final approach: removing HEM entirely and learning the full hierarchy from scratch, from the very first level. This requires a GPU with enough memory to handle the dense operations at the 14,000-node level.

---

## Slide 14: Full DiffPool — No HEM At All

The Full DiffPool approach removes the HEM crutch entirely. Instead of using fixed parent assignments for early levels and learned assignments only at the end, every single level learns its own soft clustering from scratch.

The architecture works as follows. At each level, we run two parallel graph convolutions on the current node features: one to produce embeddings Z, and one to produce the assignment logits that become the soft assignment matrix S after softmax. The assignment matrix S is then used in two ways. First, we pool the node features: X' = S^T @ Z, where each row of X' corresponds to a super-node whose features are a weighted combination of the original nodes. Second, we pool the adjacency: A' = S^T @ A @ S, which produces a dense adjacency matrix for the new super-nodes. We then extract sparse edges from A' for the next level.

Each level also produces two auxiliary losses. The link prediction loss encourages S^T @ S to approximate the original adjacency A — this pushes the model to preserve graph structure in its clustering. The entropy loss encourages each node's assignment to be close to a one-hot vector — this prevents the model from spreading a node's membership across many clusters.

The crucial difference from the hybrid approach is that level 0 also learns its assignment. There is no free pass with HEM parents. This means we need to materialize the dense 14,000 by 14,000 adjacency and compute S^T @ A @ S at the very first level. With batch size 16, this consumes approximately 19 gigabytes of GPU memory — too much for a 24-gigabyte card, but feasible on a 40 or 80-gigabyte A100.

The trade-off is clear: we trade memory and compute for flexibility. The model is no longer constrained by HEM's fixed clustering. It can learn which genes to group together for the purpose of classification, even if those groupings do not match the protein-protein interaction structure captured by STRING-DB.

---

## Slide 15: Approach Comparison

Across three approaches, we see a clear trade-off between structural prior, flexibility, and computational cost.

The Fixed HEM model is the simplest and fastest. It uses pre-computed parent assignments for pooling, requires no dense matrix operations, and converges in a single epoch. Its structural prior is very strong — the clustering is entirely determined by STRING-DB's protein-protein interaction scores, with no influence from the classification task. Despite this rigidity — or perhaps because of it — it achieves the best accuracy at 92.3% when using no pooling at all.

The Hybrid DiffPool occupies the middle ground. It inherits HEM's structural prior for the first two levels, keeping memory costs low, but introduces a learned assignment at the final level. This adds auxiliary losses that must be carefully tuned, and requires substantially more training — 127 epochs versus 1. The result is 70.3% accuracy, which is better than the HEM-coarsened models but worse than the no-pooling MLP baseline.

The Full DiffPool removes all structural priors. Every level learns its own clustering from scratch. This requires a GPU with at least 32 gigabytes of memory to handle the dense operations at level 0, and the training time is likely to be even longer than the hybrid model. Its accuracy is not yet evaluated, but the hope is that removing HEM's constraints will allow the model to discover more classification-relevant groupings.

The comparison highlights a central tension in this work: the inductive bias provided by the graph structure appears useful in theory but harmful in practice. The best model so far is the one that ignores the graph entirely.

---

## Slide 16: Key Lessons

Several important lessons emerged from this work that extend beyond the specific models we tested.

The first and most striking lesson is that the MLP baseline — a simple fully-connected network with no graph structure at all — outperforms every graph-based approach. This suggests that for cancer type classification from expression data, the individual gene expression values are already highly discriminative. Adding graph structure, whether fixed or learned, appears to constrain the model in ways that hurt rather than help. The graph may be capturing the wrong kind of structure: protein-protein interactions reflect physical binding, not necessarily co-expression patterns that distinguish cancer types.

The second lesson is about hyperparameter sensitivity. The DiffPool auxiliary losses, when set to their default values, completely dominated the classifier loss. The model spent its gradient updates trying to predict edges and minimize assignment entropy rather than classifying cancer types. Reducing the auxiliary loss weights by an order of magnitude was the single most important factor in improving performance. This is a cautionary tale for any multi-task or regularized approach: the primary objective must dominate.

The third lesson concerns training dynamics with warm restarts. The validation accuracy follows a distinctive staircase pattern, collapsing at each restart and recovering to a higher peak. Stopping early — say after 1 or 2 cycles — would have led us to believe the model was much worse than it actually was. Understanding this dynamic required running the full schedule of 7 cycles, and it underscores the importance of letting warm restart schedules play out completely.

Fourth, scatter pooling proved to be an effective memory-saving technique. By using pre-computed HEM parent assignments instead of learned assignment matrices, we reduced the memory footprint at large graph sizes from O(n²) to O(n). This is a practical technique that could be useful in any graph learning setting where the graph is too large for dense operations.

Finally, the Full DiffPool approach — removing HEM entirely — may or may not close the gap with the MLP baseline. It removes the structural prior that constrained the hybrid model, but it also loses the regularization that prior provided. Testing it requires hardware that we currently lack, but it represents a natural next step in this line of investigation.

---

## Slide 17: Future Work

This work opens several directions for future investigation.

The most immediate next step is to evaluate the Full DiffPool model on hardware capable of running it. The dense operations at level 0 require at least 32 gigabytes of GPU memory, ideally 40 or 80. Running the full experiment would tell us whether removing HEM's constraints closes the gap with the MLP baseline, or whether learned pooling is inherently limited on this type of data.

If the gap persists even in the full model, it would be worth investigating why graph structure hurts. One concrete hypothesis is that pooling inevitably merges discriminative genes with non-informative neighbors, diluting the signal. This could be tested with an ablation study that selectively pools only subsets of genes, or by analyzing which genes the learned assignment matrices actually group together.

Another direction is to replace the STRING-DB network with a different graph construction. Co-expression networks, where edges connect genes whose expression is correlated across samples, might be more relevant for classification than protein-protein interaction networks. Correlation-based graphs would capture the statistical relationships in the data itself rather than relying on prior biological knowledge that may not align with the classification task.

Gene-level interpretability is another important direction. The Fixed HEM model's weighted pooling mechanism learns an importance weight for each gene, and the Full DiffPool model produces soft assignment matrices that reveal which genes the model chooses to group together. Analyzing these weights could yield biological insights about which genes and pathways are most discriminative for each cancer type.

Finally, the models could be extended to multi-task learning, jointly predicting both the cancer type and whether the sample is tumor or normal tissue. The codebase already includes infrastructure for this (CohortAndTumorClassifier) that was not explored in the current experiments.

Each of these directions would help answer the central question raised by this work: when does graph structure help, and when does it hurt, in genomic classification tasks?
