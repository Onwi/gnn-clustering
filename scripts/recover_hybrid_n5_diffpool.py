"""One-off recovery script: the hybrid_n5_diffpool run's tuning phase (all 14
trials) completed successfully before a machine reboot killed the process
during the final-retrain phase. Restores the completed Ray Tune experiment
from disk and re-runs just the final retrain + test evaluation, instead of
re-running the ~12h tuning search from scratch.
"""
import sys
sys.path.insert(0, "/home/lab203/workspace/gnn-clustering/scripts/experiments")
sys.argv = [
    "diffpool_experiment.py",
    "/home/lab203/workspace/gnn-clustering/data/string_data/data/tcga_cohorts_and_tumor_classification",
    "/home/lab203/workspace/gnn-clustering/data/string_data/data/networks/levels",
    "--path-network", "/home/lab203/workspace/gnn-clustering/data/string_data/data/networks/stringdb_top100pc.csv",
    "--pooling-type", "diffpool",
    "--n-hybrid", "5", "--n-hybrid-start", "5",
    "--tune", "--num-samples", "14", "--n-cycles", "7",
    "--batch-size", "96", "--device", "cuda", "--gpu-per-trial", "1", "--cpu-per-trial", "8",
    "--n-holdouts", "1", "--use-train-set-weights",
    "--path-output", "/home/lab203/workspace/gnn-clustering/outputs/hybrid_n5_diffpool",
]

from pathlib import Path
from ray import tune
import diffpool_experiment as de

args = de.parse_args()
n_hybrid = 5
rep = 0
random_state = 7  # printed in the original run's log before the reboot

path_experiment = Path(args.path_output) / f"diffpool_hybrid{n_hybrid}_rep{rep}"
ray_experiment_dir = path_experiment / "ray_results" / "train_and_validate_model_2026-09-10_23-26-26"

print(f"Restoring tuner from: {ray_experiment_dir}")
restored_tuner = tune.Tuner.restore(path=str(ray_experiment_dir))
results = restored_tuner.get_results()
print(f"Restored {len(results)} completed trials.")

best = results.get_best_result(scope="all")
print("Best config:", best.config)

print("Running final retrain + test evaluation ...")
de.test_tuned_model(
    results, n_hybrid=n_hybrid, args=args,
    path_experiment=path_experiment, random_state=random_state,
)
print("Done.")
