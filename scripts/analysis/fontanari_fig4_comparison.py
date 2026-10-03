"""Compare our cohort-of-origin runs with Fontanari & Recamonde-Mendoza (arXiv:2601.06381), Fig. 4a.

Fig. 4a (cohort confusion matrix of their multi-task model) was transcribed by hand from the PDF.
Its row totals equal the per-class counts of our shared test split (seed 7, 1,543 samples), and its
tumour/normal matrix (Fig. 4c) has the same 1,418/125 split, so both were evaluated on the same
test samples. Run from the repo root; writes outputs/fontanari_comparison/summary.txt."""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ORDER = ["blca", "thca", "lusc", "kirp", "brca", "luad", "coad", "lihc",
         "prad", "esca", "ucec", "stad", "kirc", "hnsc", "read", "kich"]
CM = np.array([  # rows = label, cols = prediction, both in ORDER
    [80, 0, 1, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 92, 0, 2, 14, 0, 0, 1, 0, 1, 0, 0, 6, 0, 0],
    [0, 0, 0, 58, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 4],
    [2, 0, 0, 0, 245, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0],
    [0, 0, 7, 0, 0, 104, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 83, 0, 0, 0, 0, 0, 0, 0, 17, 0],
    [0, 0, 1, 1, 1, 0, 0, 73, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 3, 0, 1, 0, 0],
    [0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 120, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 8, 0, 57, 0, 0, 0, 0],
    [1, 0, 0, 7, 0, 1, 0, 0, 0, 0, 0, 0, 119, 0, 0, 2],
    [5, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0],
    [0, 0, 0, 0, 0, 1, 21, 0, 0, 0, 0, 0, 0, 0, 10, 0],
    [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20],
])
y_true = np.repeat(np.repeat(ORDER, 16), CM.ravel())
y_pred = np.tile(ORDER, 16).repeat(CM.ravel())

# check the transcription against our shared test split
m = pd.read_csv("data/string_data/data/tcga_cohorts_and_tumor_classification/sample_metadata.csv",
                index_col=0).sample(frac=1, random_state=7).iloc[6166:]
ours = m.cohort.value_counts()
assert all(ours[c] == CM[i].sum() for i, c in enumerate(ORDER)), "row totals differ from our test split"

out = []
def report(name, yt, yp):
    out.append(f"{name:34s} acc {accuracy_score(yt, yp)*100:6.2f}  bal acc {balanced_accuracy_score(yt, yp)*100:6.2f}"
               f"  F1-macro {f1_score(yt, yp, average='macro'):.3f}")
report("Fontanari Fig. 4a (multi-task)", y_true, y_pred)

base = "outputs/hem_vs_dmon111_n6/"
for name, pat in [("Fixed HEM", "hem_hybrid6_rep{}"), ("Learned DMoN (111)", "k111/dmon_hybrid6_rep{}")]:
    probs, accs, bals, f1s = [], [], [], []
    for r in range(3):
        d = base + pat.format(r) + "/final_model_results/"
        p = pd.read_csv(d + "predictions.csv", index_col=0)
        accs.append(accuracy_score(p.labels, p.predictions)); bals.append(balanced_accuracy_score(p.labels, p.predictions))
        f1s.append(f1_score(p.labels, p.predictions, average="macro"))
        o = pd.read_csv(d + "outputs.csv", index_col=0); cls = [c for c in o.columns if c != "labels"]
        z = o[cls].to_numpy(); e = np.exp(z - z.max(1, keepdims=True)); probs.append(e / e.sum(1, keepdims=True))
    out.append(f"{name + ' (mean of 3 reps)':34s} acc {np.mean(accs)*100:6.2f}  bal acc {np.mean(bals)*100:6.2f}"
               f"  F1-macro {np.mean(f1s):.3f} +/- {np.std(f1s, ddof=1):.3f}")
    report(name + " (ensemble)", o["labels"], np.array(cls)[np.mean(probs, 0).argmax(1)])

f_his = f1_score(y_true, y_pred, labels=ORDER, average=None)
out.append("\nper-class F1, Fontanari Fig. 4a: " + ", ".join(f"{c} {f:.3f}" for c, f in zip(ORDER, f_his)))
print("\n".join(out))
open("outputs/fontanari_comparison/summary.txt", "w").write("\n".join(out) + "\n")
