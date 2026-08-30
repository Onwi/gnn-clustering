---
title: Apresentação — Arquitetura e Modos de Pooling
author: Equipe de Pesquisa
date: 2026-07-12
---

# Apresentação: Arquitetura e Modos de Pooling

---

## Agenda

- Visão geral da arquitetura
- Modos do HEM (Heavy Edge Matching)
- Modos do Full DiffPool
- Modos do Hybrid DiffPool
- Exemplos práticos e configurações de experimento
- Conclusões e referências

---

## Visão Geral da Arquitetura

- Objetivo: classificar dados genômicos (TCGA) usando GNNs hierárquicas.
- Fluxo geral:
  - Rede de interação gênica (STRING) → grafo inicial
  - GNNs convolucionais espaciais + camadas de pooling hierárquico
  - Classificador final (MLP / FCModel)
- Implementação chave: pre-computação de níveis de coarsening e modelos compostos em `src/pooling_genomic`.

---

## Motivação para Pooling em GNNs

- Reduzir a escala do grafo mantendo estrutura relevante.
- Capturar padrões em múltiplas escalas (local → global).
- Melhorar eficiência e geralização em tarefas de classificação de amostras.

---

## HEM — Heavy Edge Matching (Resumo)

- Método de coarsening heurístico baseado em emparelhar vértices conectados por arestas "pesadas".
- Objetivo: agrupar nós fortemente ligados para formar super-nós.
- Características:
  - Determinístico/estocástico por ciclo de emparelhamento
  - Rápido, escalável para pré-processamento

---

## HEM — Modos e Variantes

- Modo Single-level: aplicar HEM uma vez para reduzir tamanho (
  ideal para ensaios rápidos).
- Modo Multi-level: repetir HEM iterativamente para gerar hierarquia de níveis (usado em experimentos com N níveis).
- Modo Fixed-supernodes: gerar níveis off-line e fixá-los para todo o treino/avaliação (útil para comparações justas e para Hybrid setups).
- Parâmetros importantes: critério de seleção de arestas (peso), razão de colapso (pooling ratio), número de ciclos/replicações.

---

## Full DiffPool — Conceito

- Pooling aprendido differentiable: a cada camada, aprende-se uma matriz de atribuição $S \in \mathbb{R}^{n \times k}$.
- Operação de pooling:
 - Operação de pooling:
 $$
 X' = S^{T} X
 $$

 $$
 A' = S^{T} A S
 $$
- O modelo treina $S$ junto com camadas GNN para otimizar a tarefa alvo, possivelmente com regularizadores adicionais.

---

## Full DiffPool — Modos e Componentes

- Modo Padrão (Full DiffPool): todas as camadas de pooling são aprendidas (atribuição via GNN), com $k$ especificado por camada.
- Modos de perda e regulação:
  - Loss de reconstrução de adjacência (link prediction): $L_{link}=\|A - S S^{T}\|_F$ (ajuda preservar estrutura)
  - Entropy / balance regularization: $L_{ent} = -\sum_{i,j} S_{ij} \log S_{ij}$ para evitar colapsos triviais
- Configurações práticas: escolher $k$ (número de clusters), normalização de $S$, otimização de memória (batching ou níveis pré-calculados).

---

## Full DiffPool — Vantagens e Limitações

- Vantagens:
  - Aprendizado adaptativo de agrupamentos específicos da tarefa
  - Pode capturar estruturas não triviais que heurísticas ignoram
- Limitações:
  - Custo computacional e memória (matrizes densas $S$)
  - Risco de overfitting ou colapso sem regularização

---

## Hybrid DiffPool — Conceito

- Combina pooling pré-computado (fixo, ex.: HEM) e pooling aprendido (DiffPool) em diferentes níveis.
- Ideia: usar níveis estruturais robustos (HEM) para reduzir escala onde DiffPool é caro, e aplicar DiffPool onde há sinal útil para aprender.

---

## Hybrid DiffPool — Modos Comuns

- Modo Híbrido Sequencial: primeiros níveis (coaresing grosseiro) via HEM (fixed), níveis superiores via DiffPool aprendido.
- Modo Concatenado: features agregadas de níveis HEM e saídas DiffPool são concatenadas para o classificador.
- Modo Regularizado: usar S aprendida mas forçar proximidade com agrupamentos HEM (termo de perda que penaliza diferença entre atribuições).
- Parâmetros: quais níveis são fixos vs aprendidos, como combinar features, pesos relativos das perdas.

---

## Implementação prática (no repositório)

- Nossos scripts chave:
  - Geração de níveis: `scripts/generate_graph_levels.py`
  - Modelos: `src/pooling_genomic/models.py` (funções `build_coarsening_model` / `build_fixed_supernodes_coarsening_model`)
  - Coarsening HEM: `src/pooling_genomic/coarsening.py`
  - Carregamento de níveis: `src/pooling_genomic/networks.py`

---

## Configurações de Experimento (recomendadas)

- Para comparar modos:
  - Dataset: TCGA (uso de cohorts em `data/tcga_brca_subtypes_classification`)
  - Métricas: Acurácia, AUC por classe, curva de treinamento, confusão
  - Varie: número de níveis, usar HEM fixo vs DiffPool total vs Hybrid
  - Regularizadores: incluir $L_{link}$ e $L_{ent}$ para DiffPool

---

## Exemplo de Pipeline Experimental

1. Gerar níveis HEM: `python scripts/generate_graph_levels.py` (parâmetros: n-levels)
2. Rodar experimento Fixed HEM: `python scripts/experiments/coarsening_levels.py <data> <levels> --max-n-levels 3 --n-cycles 1`
3. Rodar Full DiffPool: `python scripts/experiments/diffpool_experiment.py --use-diffpool --levels 3`
4. Rodar Hybrid: combinar `--fixed-levels 1` + `--learned-levels 2` (ex.: `fixed_supernodes_coarsening.py` helper)

---

## Boas Práticas e Dicas

- Pre-compute níveis pesados (HEM) para experimentos repetidos — economiza tempo e garante comparabilidade.
- Use regularização de DiffPool quando o número de clusters for pequeno/rápido.
- Monitorar distribuição de tamanhos de clusters e entropia de `S` durante treino.

---

## Conclusões

- HEM é eficiente e robusto para coarsening off-line; DiffPool oferece pooling adaptativo, porém mais caro; Hybrid une forças: escalabilidade + adaptabilidade.
- Recomenda-se comparar modos em condições controladas (mesmos splits, seeds, níveis pré-computados) para avaliar ganhos reais.

---

## Referências

- Ying, R. et al., "Hierarchical Graph Representation Learning with Differentiable Pooling" (DiffPool)
- Métodos Heurísticos de Coarsening: Heavy Edge Matching (literatura clássica em graph coarsening)
- Código e scripts do repositório: ver `src/pooling_genomic` e `scripts/experiments`.

---

## Fim

- Posso ajustar o conteúdo, adicionar diagramas (SVG/PNG) ou exportar para PDF/PowerPoint — deseja que eu gere imagens explicativas também?
