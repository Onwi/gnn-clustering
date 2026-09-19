"""Ensemble predictions across holdout reps that share the same test split.

Only valid when the reps were run with --shared-test-split (see
changes-from-claude.md fix #8) -- otherwise each rep's outputs.csv contains a
different set of test patients and there's nothing to average across reps.

Reads <path-output>/{pooling_type}_hybrid{n_hybrid}_rep{rep}/final_model_results/
outputs.csv for each rep (one column per class, raw logits, plus a "labels"
column -- written by scripts/experiments/diffpool_experiment.py's
analyze_final_model_results). Averages per-rep softmax probabilities (not raw
logits -- softmax-then-average is the standard, scale-robust ensembling
choice) and reports the ensembled accuracy/balanced accuracy alongside each
individual rep's own numbers.

Usage:
    python scripts/analysis/ensemble_predictions.py \
        --path-output outputs/hybrid_n6_diffpool_full3 --pooling-type diffpool \
        --n-hybrid 6 --n-holdouts 3 \
        --out outputs/hybrid_n6_diffpool_full3/ensemble.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def load_rep_outputs(path_output: Path, pooling_type: str, n_hybrid: int, rep: int):
    run_dir = path_output / f"{pooling_type}_hybrid{n_hybrid}_rep{rep}" / "final_model_results"
    outputs_path = run_dir / "outputs.csv"
    if not outputs_path.exists():
        return None
    df = pd.read_csv(outputs_path, index_col=0)
    classes = [c for c in df.columns if c != "labels"]
    logits = df[classes].to_numpy(dtype=np.float64)
    labels = df["labels"].to_numpy()
    return {"classes": classes, "logits": logits, "labels": labels, "run_dir": run_dir}


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def build_report(reps, ensemble_acc, ensemble_bal_acc, n_hybrid, pooling_type, n_holdouts):
    lines = []
    lines.append(f"# Prediction ensemble -- {pooling_type} hybrid{n_hybrid}\n")
    lines.append(
        f"{len(reps)}/{n_holdouts} reps loaded, all sharing the same test-set patients "
        "(requires --shared-test-split at run time -- see changes-from-claude.md fix #8). "
        "Ensembled by averaging per-rep softmax probabilities, then argmax.\n"
    )
    lines.append("## Individual reps\n")
    lines.append("| Rep | Test accuracy | Test balanced accuracy |")
    lines.append("|---|---|---|")
    for r in reps:
        acc = accuracy_score(r["labels"], r["classes_arr"][r["logits"].argmax(axis=1)])
        bal = balanced_accuracy_score(r["labels"], r["classes_arr"][r["logits"].argmax(axis=1)])
        lines.append(f"| {r['rep']} | {acc*100:.2f}% | {bal*100:.2f}% |")
    lines.append("")
    lines.append("## Ensemble\n")
    lines.append(f"- **Test accuracy: {ensemble_acc*100:.2f}%**")
    lines.append(f"- **Test balanced accuracy: {ensemble_bal_acc*100:.2f}%**")
    lines.append(f"- N test samples: {len(reps[0]['labels'])}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-output", type=str, required=True)
    parser.add_argument("--pooling-type", type=str, required=True, choices=["diffpool", "dmon"])
    parser.add_argument("--n-hybrid", type=int, required=True)
    parser.add_argument("--n-holdouts", type=int, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    path_output = Path(args.path_output)
    reps = []
    for rep in range(args.n_holdouts):
        r = load_rep_outputs(path_output, args.pooling_type, args.n_hybrid, rep)
        if r is not None:
            r["rep"] = rep
            reps.append(r)

    if not reps:
        raise SystemExit(f"No rep outputs.csv found under {path_output}")

    # Sanity check: every rep must share the same test-set labels in the same
    # order (shuffle=False + --shared-test-split), or averaging is meaningless.
    ref_labels = reps[0]["labels"]
    for r in reps[1:]:
        if len(r["labels"]) != len(ref_labels) or not (r["labels"] == ref_labels).all():
            raise SystemExit(
                f"rep{r['rep']}'s test labels don't match rep{reps[0]['rep']}'s -- these reps "
                "don't share a test split (was --shared-test-split used at run time?). "
                "Ensembling requires identical, identically-ordered test sets across reps."
            )
    ref_classes = reps[0]["classes"]
    for r in reps[1:]:
        if r["classes"] != ref_classes:
            raise SystemExit(f"rep{r['rep']}'s class columns don't match rep{reps[0]['rep']}'s.")

    for r in reps:
        r["classes_arr"] = np.array(r["classes"])

    probs = np.mean([softmax(r["logits"]) for r in reps], axis=0)
    ensemble_pred_idx = probs.argmax(axis=1)
    ensemble_pred = ref_classes and np.array(ref_classes)[ensemble_pred_idx]

    ensemble_acc = accuracy_score(ref_labels, ensemble_pred)
    ensemble_bal_acc = balanced_accuracy_score(ref_labels, ensemble_pred)

    report = build_report(reps, ensemble_acc, ensemble_bal_acc, args.n_hybrid, args.pooling_type, args.n_holdouts)
    out_path = Path(args.out) if args.out else path_output / "ensemble.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote ensemble report to {out_path}")
    print(f"Ensemble accuracy: {ensemble_acc*100:.2f}%, balanced: {ensemble_bal_acc*100:.2f}%")


if __name__ == "__main__":
    main()
