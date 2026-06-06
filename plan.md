# A Comparative Study between Fixed and Dynamic Graph Clustering in GNNs for Cancer Classification using RNA-seq and PPI Networks
## Chapter 1: Introduction

### 1.1 Motivation
- Cancer genomics: RNA-seq enables tumor classification from expression profiles
- Problem: high-dimensional (~14K genes), small-sample (7,709 patients) regime
- Prior knowledge: protein-protein interaction (PPI) networks capture functional relationships
- Prior thesis baseline: Thomas Vaitses Fontanari, Hierarchical Pooling and Explainability in Graph Neural Networks for Tumor and Tissue-of-Origin Classification Using RNA-seq Data, uses a pre-computed clustering approach based on Heavy Edge Matching (HEM)
- Central question in this dissertation: compared with that pre-computed HEM baseline, can learnable clustering with DiffPool (Full DiffPool and Hybrid DiffPool) provide better or more robust performance for the same RNA-seq classification task?

### 1.2 Research Questions
1. Can learned node assignment via Full DiffPool match or exceed the 96.05% accuracy of Fontanari's pre-computed HEM clustering on the same TCGA pan-cancer task?
2. Why does Full DiffPool degrade to near-random performance (~27.7%) under default settings on a 14K-node PPI graph, and which specific mitigations (pre-pooling encoder, low learning rate, low auxiliary loss weights, gradient clipping) are necessary to recover meaningful performance (71.68%)?
3. Does a Hybrid DiffPool strategy â using Fontanari's HEM coarsening for the large early levels and learned DiffPool only once the graph is below 500 nodes â narrow the 24-point gap between Full DiffPool and Fixed HEM?
4. What is the fundamental limit of learned pooling on RNA-seq data: is the bottleneck the pooling method, the graph topology, or the nature of the expression signal itself?

### 1.3 Contributions
- Reproduces and contextualises Fontanari's Fixed HEM result (96.05% test accuracy) as a controlled reference baseline on the same TCGA pan-cancer task and graph
- Demonstrates that Full DiffPool, under default settings, collapses to ~27.7% accuracy on this task due to three compounding failure modes: (1) 440:1 node-to-cluster compression without any structural prior, (2) dense `S^T A S` adjacency operations that corrupt graph topology (~200M elements at 14K nodes), and (3) auxiliary losses (link prediction + entropy) that dominate the classification gradient
- Identifies that a pre-pooling GNN encoder (1D-Conv â 16-channel ChebConv), combined with low learning rate (0.0008), low auxiliary loss weights (~0.0005), and gradient clipping, rescues Full DiffPool to 71.68% â still 24 points below Fixed HEM but 44 points above the untuned default
- Shows that Hybrid DiffPool (HEM coarsening for early levels, learned DiffPool below 500 nodes) reaches ~70.3%, confirming that reducing the compression ratio is necessary but not sufficient to match pre-computed biological priors
- Establishes that for RNA-seq classification the PPI graph acts as a regularizer, not a discriminative signal source: expression features alone achieve 95.46% and the graph adds at most +0.6%; learned pooling that cannot exploit a biological prior yields substantially worse results

---

## Chapter 2: Background and Related Work

### 2.1 Biological Background
- RNA-seq gene expression quantification
- TCGA pan-cancer cohort structure (16 tumor types, 7,709 samples)
- STRING-DB PPI network: 14K genes, 8.2M edges, confidence scores

### 2.2 Graph Neural Networks
- Message passing paradigm (Gilmer et al., 2017)
- ChebNet / ChebConv (Defferrard et al., 2016) - spectral graph convolutions
- Why ChebConv over GCN: flexible receptive field K, numerically stable on large graphs

### 2.3 Graph Pooling
- Coarsening-based: Heavy Edge Matching (HEM), Graclus
- Learned soft assignment: DiffPool (Ying et al., 2018), MinCut (Bianchi et al., 2020)
- Fixed vs learned trade-offs: computational cost, structural priors, optimization difficulty

### 2.4 GNNs for Genomic Data
- Previous work: Thomas Vaitses Fontanari, Hierarchical Pooling and Explainability in Graph Neural Networks for Tumor and Tissue-of-Origin Classification Using RNA-seq Data
- Gap: no systematic comparison of pooling strategies on the same genomic task

---

## Chapter 3: Methodology

**Contribution boundary for this dissertation:** Fixed HEM is reused from prior thesis work as a reference baseline. The methodological proposals investigated here are Full DiffPool and Hybrid DiffPool.

### 3.1 Problem Formulation
- Input: gene expression vector x â R^d (d â 14,133), PPI graph G = (V, E, w)
- Task: classify into k = 16 cancer types
- Setting: fixed graph shared across all samples, node features vary per sample

### 3.2 Fixed HEM (Heavy Edge Matching, Reference Baseline)

#### 3.2.1 Hierarchical pre-computation
- HEM algorithm: greedy matching by edge weight, contract matching pairs
- Level generation: 14K -> ~7K -> ~3.5K -> ... -> 111 nodes (8 levels)
- Storage: parent assignment vectors per level
- Provenance: methodology inherited from prior thesis; included here as a fixed reference, not a novel method contribution

#### 3.2.2 Model architecture
- ChebConv + LayerNorm at each level
- `scatter(parents, sum)` pooling with optional learnable node weights
- Progressive channel doubling (1, 2, 4, ..., 32)
- Final MLP classifier on the flattened representation

#### 3.2.3 Training objective
- Cross-entropy loss only
- Optional L1 regularization on node importance weights
- Role in this dissertation: benchmark for comparison against proposed learned pooling approaches

### 3.3 Full DiffPool (Learned Soft Assignment, Proposed in This Dissertation)

#### 3.3.1 Architecture
- Pre-pooling GNN encoder (1D-Conv -> ChebConv, enriches node features to `encoder_channels` dims)
- Pooling GNN for soft assignment matrix S
- Feature pool: X' = S^T Z (Z from embedding GNN)
- Adjacency pool: A' = S^T A S
- Followed by ChebConv on pooled graph

#### 3.3.2 Auxiliary losses
- Link prediction loss: reconstruct A from S (||A - SS^T||_F)
- Entropy loss: encourage uniform assignments (-mean(H(S)))
- Combined: L = L_cls + lambda_link*L_link + lambda_ent*L_ent

#### 3.3.3 Key hyperparameters
- Number of clusters per level (k), encoder channels, encoder layers
- ChebConv K (receptive field), auxiliary loss weights
- Learning rate: critical - default 0.05 fails, tuned 0.0008 succeeds

### 3.4 Hybrid DiffPool (Proposed in This Dissertation)

#### 3.4.1 Architecture
- Early levels: HEM pre-computed coarsening (identical to Fixed HEM)
- Switch to learned DiffPool when nodes <= `dense_threshold` (default: 500)
- Combines structural prior (HEM) with flexible assignment (DiffPool)

#### 3.4.2 Advantages
- Avoids 440:1 compression problem (HEM handles it)
- Sparse operations at early levels
- Dense operations only on manageable 500x500 adjacency
- Explicit objective: retain prior biological structure early while learning task-adaptive assignments late

### 3.5 Implementation Details
- PyTorch 1.12.1, PyTorch Geometric 2.1.0, CUDA 11.6, RTX 3090 Ti
- Cosine annealing warm restarts (T_0=1, T_mult=2)
- Gradient clipping (max_norm=5.0 for DiffPool)
- Fixed holdout seeds for reproducibility (np.random.default_rng(123))
- Reporting convention: Fixed HEM results are marked as prior-thesis baseline or reproduced baseline; Full/Hybrid DiffPool results are this dissertation's primary experimental outputs

---

## Chapter 4: Experimental Setup

### 4.1 Dataset Description
- TCGA pan-cancer: 33 cancer types mapped to 16 super-classes
- Gene expression: log2(TPM+1) normalized, 14,133 protein-coding genes
- Train/val/test split: 60/20/20 stratified
- Class balance statistics (Figure)

### 4.2 Graph Pre-processing
- STRING-DB v11.5: confidence scores > 700, top 100 interactions per gene
- Result: 14,133 nodes, 8,248,194 edges
- Graph-level pre-computation for HEM: 8 levels generated via `generate_graph_levels.py`

### 4.3 Hyperparameter Configurations

**Fixed HEM hyperparameters (prior-thesis baseline protocol):**
- n_levels â {0, 1, 2}
- Use ChebConv: {True, False}
- Use weighted pooling: {True, False}
- lr=0.05, weight_decay=0.01 (fixed)

**Full DiffPool hyperparameters (tuned):**
- n_levels â {1, 2}, max_clusters â {32, 64, 128, 256}
- encoder_channels â {8, 16, 32}, encoder_layers â {2, 3}
- K â {2, 3}, lr â [1e-4, 1e-1], weight_decay â [1e-4, 1e-1]
- lambda_link â [1e-5, 1e-3], lambda_ent â [1e-5, 1e-3]

**Hybrid DiffPool hyperparameters:**
- n_hybrid â {1, 2}, dense_threshold=500
- lr â [1e-4, 1e-1], weight_decay â [1e-4, 1e-1]

### 4.4 Training Protocol
- Multiple holdout repetitions: 5 random seeds per config
- Early stopping via validation loss (patience: 20 epochs)
- Cosine annealing T_0=1, T_mult=2, total epochs from n_cycles

### 4.5 Evaluation Metrics
- Test accuracy, balanced accuracy (macro-averaged recall)
- Confusion matrices, per-class F1
- Training curves (loss, accuracy vs epoch)

---

## Chapter 5: Results

### 5.1 MLP Baseline
- 95.46% test accuracy, 93.66% balanced accuracy
- Demonstrates: expression features alone are highly discriminative
- Implication: the graph must provide *additional* signal, not just the same information

### 5.2 Fixed HEM

Note: This section reports the inherited Fixed HEM reference baseline, used for direct comparison with the learned pooling methods evaluated in this dissertation.

#### 5.2.1 Effect of pooling levels
- n_levels=0 (no pool): 95.46%
- n_levels=1 (one coarsening): 94.36-96.05% (depends on weighted/convs)
- n_levels=2 (two coarsenings): 94.10%
- Diminishing returns - deeper pooling loses spatial resolution

#### 5.2.2 Effect of weighted pooling and ChebConv
- No conv, no weighted: 95.46% (same as MLP)
- Conv, no weighted: 95.66%
- Weighted, no conv: 95.33%
- Conv + weighted: 96.05% - best config
- Conclusion: both components contribute small additive improvements

#### 5.2.3 Best model analysis
- Confusion matrix shows near-diagonal performance
- Errors concentrated on histologically similar cancer types

### 5.3 Full DiffPool

#### 5.3.1 Default configuration (untuned)
- Test accuracy: ~27.7% (near random 6.25% for 16 classes)
- Training accuracy: 20-40% - model fails to learn
- Multiple levels consistently hurt (20% for 2-level vs 27% for 1-level)

#### 5.3.2 Tuned configuration
- Best: 71.68% test accuracy, 65.70% balanced accuracy (epoch 20)
- Key factors: pre-pooling encoder (16ch, 2 layers), low LR (0.0008), low aux weights
- Still lags MLP baseline by 24 percentage points

#### 5.3.3 Ablation: encoder impact
- No encoder: 27.7% (untuned) / 58.3% (tuned)
- With encoder: 71.68%
- Encoder is essential - node features need enrichment before assignment

#### 5.3.4 Failure mode analysis
- 440:1 compression impossible without prior
- Dense S^T A S corrupts graph structure
- Auxiliary losses dominate early training
- Assignment entropy collapses (most nodes assigned to 1-2 clusters)

### 5.4 Hybrid DiffPool

#### 5.4.1 Tuned configuration
- Best reported: 70.3% (n_hybrid=2, lr=0.0099, wd=0.0344, 127 epochs)
- Our run: 56.8% validation (50 epochs)
- Default: ~27% (untuned)

#### 5.4.2 Comparison with Full DiffPool
- +42.6% improvement over Full DiffPool default
- But still -25.6% below Fixed HEM
- HEM prior helps but learned final layer still problematic

### 5.5 Quantitative Summary

**Table 5.1: Overall comparison across all configurations**

| Model | Test Accuracy | Balanced Acc | Params | GPU Mem | Training Time |
|---|---|---|---|---|---|
| MLP baseline | 95.46% | 93.66% | 452K | 2.1 GiB | ~5 min |
| Fixed HEM (best) | 96.05% | 93.53% | 70K | 9.6 GiB | ~10 min |
| Full DiffPool (untuned) | 27.67% | 17.5% | 10.6M | 21.6 GiB | ~60 min |
| Full DiffPool (tuned) | 71.68% | 65.70% | 19.4M | 15.2 GiB | ~45 min |
| Hybrid DiffPool (tuned) | 70.3% | - | ~1.5M | 10.6 GiB | ~4 hrs |

Source convention: Fixed HEM rows are prior-thesis reference/reproduced baseline; Full DiffPool and Hybrid DiffPool rows correspond to this dissertation's proposed learned pooling experiments.

---

## Chapter 6: Analysis and Discussion

### 6.1 Why the Graph Helps Only Marginally
- Expression data is inherently discriminative - tumor types have distinct transcriptional programs
- The PPI graph adds functional regularization (grouping co-expressed genes)
- But the signal exists at the individual gene level, not just the pathway level
- Implication: for gene expression, graph structure is a *regularizer*, not a *source of signal*

### 6.2 Why the Fixed HEM Baseline Outperforms Learned Pooling
- Strong biological prior vs no prior
- Deterministic, no auxiliary objectives
- Sparse operations throughout
- Channel doubling matches information bottleneck intuition

### 6.3 Why Full DiffPool Fails
**Three fundamental problems:**

1. **Assignment learning is ill-posed at scale**
   - 14K nodes x 32 clusters = 450K free parameters per level
   - Each node has only a scalar feature - insufficient signal to learn 32-dimensional assignment
   - ChebConv's K=2 receptive field covers only local neighborhood (~200 edges), can't capture global community structure

2. **Dense adjacency destroys graph structure**
   - `A_dense = to_dense_adj(edge_index)` creates 200M elements
   - `S^T A S` produces a dense pooled adjacency - all super-nodes become connected
   - This propagates spurious connections to deeper levels

3. **Auxiliary losses compete with classification**
   - Link prediction: impossible to reconstruct 200M-element adjacency from 1024-element summary
   - Entropy: encourages uniform assignments, directly opposing clustering
   - Both dominate early gradients when S is random

### 6.4 What Mitigations Work
- Pre-pooling encoder (1D-Conv + ChebConv): enriches node features from 1 -> 16 dimensions before assignment, making the clustering problem tractable
- Low learning rate (0.0008 vs 0.05): prevents early collapse of softmax assignment
- Low auxiliary loss weights (0.0005 vs 0.001): prevents aux losses from dominating
- Gradient clipping (max_norm=5.0): stabilizes assignment gradient

### 6.5 Generalizability
- On datasets where graph structure *is* the signal (molecules, social networks), learned pooling methods may outperform fixed coarsening
- On genomic data where feature values dominate, simple pooling with strong priors is preferred
- The encoder+DiffPool approach may transfer to other high-dimensional bioinformatics tasks

### 6.6 Computational Cost vs Accuracy Trade-off
- Fixed HEM: best accuracy, lowest cost, requires pre-computation
- Full DiffPool: worst accuracy, highest cost in both memory and time
- Hybrid: intermediate but adds pre-computation dependency
- Practical recommendation for genomic tasks: start with Fixed HEM

---

## Chapter 7: Conclusion

### 7.1 Summary of Findings
1. The inherited Fixed HEM reference baseline achieves 96.05% test accuracy, marginally outperforming an MLP (95.46%) by providing structured regularization
2. Full DiffPool, proposed and evaluated in this dissertation, fails catastrophically under default settings (27.7%) and requires careful tuning plus an encoder to reach 71.68%
3. Hybrid DiffPool, proposed and evaluated in this dissertation, improves on Full DiffPool (70.3%) using HEM priors but still trails the Fixed HEM reference baseline
4. The primary bottleneck is not the GNN architecture but the nature of gene expression data - features are already highly discriminative; the graph provides marginal additional signal

### 7.2 Limitations
- Single dataset (TCGA pan-cancer) - may not generalize to other genomic tasks
- Fixed graph topology (STRING-DB) - PPI confidence thresholds may affect results
- Fixed HEM is inherited from prior thesis work, so direct comparability depends on faithful protocol reproduction and consistent data splits

### 7.3 Future Work
- Use other pooling methods (MinCut, SAGPool, ASAPool)
- Test on datasets where graph signal dominates (spatial transcriptomics, Hi-C chromatin interaction)
- Explore fixed hierarchies beyond HEM: Leiden community detection, pathway databases (KEGG, Reactome)
- Learn edge weights jointly with pooling (attentive pooling)
- Apply to larger RNA-seq datasets (GTEx, ARCHS4) to test generalization