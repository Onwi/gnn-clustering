#!/bin/bash

# cohorts=("brca" "kirc" "luad" "lusc" "esca" "kich")
# cohorts=("brca")
# cohorts=("prad")

cohorts=("lihc" "thca" "ucec" "kirp" "brca" "lusc" "luad" "read" "kirc" "coad" "hnsc" "stad" "esca" "prad" "blca" "kich")
for cohort in "${cohorts[@]}"; do
    echo "Current cohort: ${cohort}"
    python scripts/experiments/coarsening_levels.py  \
            /home/thomas/Documents/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification  \
            /home/thomas/Documents/PoolingGenomicGNNs/data/networks/levels --n-cycles 1 --max-n-levels 1 \
            --path-indices /home/thomas/Documents/PoolingGenomicGNNs/data/unseen_cohort_data/indices --num-samples 1 \
            --cohort-indices ${cohort} --n-holdouts 1 --metadata-column sample_type --tune \
            --path-output /home/thomas/Documents/PoolingGenomicGNNs/outputs_test/results_unseen_cohort_${cohort}_tumor_prediction \
            --use-train-set-weights
done
