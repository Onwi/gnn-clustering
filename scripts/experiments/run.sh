python scripts/experiments/fixed_supernodes_coarsening.py \
       /scratch/tvfontanari/PoolingGenomicGNNsData/data/tcga_cohort_classification/ \
       /scratch/tvfontanari/PoolingGenomicGNNsData/data/networks/levels/ \
       --n-cycles 5 --max-n-levels 7 --num-samples 8 --device cuda --tune \
       --path-output /scratch/tvfontanari/PoolingGenomicGNNsData/results

python scripts/experiments/coarsening_levels.py \
       /scratch/tvfontanari/PoolingGenomicGNNsData/data/tcga_cohort_classification/ \
       /scratch/tvfontanari/PoolingGenomicGNNsData/data/networks/levels/ \
       --n-cycles 5 --max-n-levels 7 --num-samples 8 --device cuda --tune \
       --path-output /scratch/tvfontanari/PoolingGenomicGNNsData/results_coarsening_levels
