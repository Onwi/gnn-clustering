#!/bin/bash

# cohorts=("brca" "kirc" "luad" "lusc" "esca" "kich")
# cohorts=("brca")
cohorts=("prad")
for cohort in "${cohorts[@]}"; do
    echo "Current cohort: ${cohort}"
    python scripts/experiments/coarsening_levels.py  \
            /home/thomas/Documents/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification  \
            /home/thomas/Documents/PoolingGenomicGNNs/data/networks/levels --n-cycles 1 --max-n-levels 1 \
            --path-indices /home/thomas/Documents/PoolingGenomicGNNs/data/cs_vs_pc_data/indices --num-samples 8 \
            --cohort-indices ${cohort} --n-holdouts 5 --metadata-column sample_type --tune \
            --path-output /home/thomas/Documents/PoolingGenomicGNNs/outputs_test/results_coarsening_pc_${cohort}_tumor_prediction
done
