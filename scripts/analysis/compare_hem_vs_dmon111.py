"""HEM vs learned DMoN (111 clusters) at n_hybrid=6: per-rep/mean/ensemble accuracy, paired sign
tests and per-class F1 (changes-from-claude.md #12). Run from outputs/hem_vs_dmon111_n6/."""
import json, numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
runs = {"HEM": "hem_hybrid6_rep{}", "DMoN": "k111/dmon_hybrid6_rep{}"}
def load(m, r):
    d = runs[m].format(r) + "/final_model_results/"
    o = pd.read_csv(d + "outputs.csv", index_col=0)
    cls = [c for c in o.columns if c != "labels"]
    z = o[cls].to_numpy(); p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return np.array(cls), p, o["labels"].to_numpy(), json.load(open(d + "model_configs.json"))
R = {m: [load(m, r) for r in range(3)] for m in runs}
labels = R["HEM"][0][2]
for m in runs:
    for cls, p, y, _ in R[m]: assert (y == labels).all() and (cls == R["HEM"][0][0]).all()
cls = R["HEM"][0][0]
pred = {m: [cls[p.argmax(1)] for _, p, _, _ in R[m]] for m in runs}
ens = {m: cls[np.mean([p for _, p, _, _ in R[m]], 0).argmax(1)] for m in runs}
print("== per rep / mean / ensemble (test, 1,543 samples)")
for m in runs:
    acc = [accuracy_score(labels, q) for q in pred[m]]; bal = [balanced_accuracy_score(labels, q) for q in pred[m]]
    print(f"{m:5s} reps acc " + " / ".join(f"{a*100:.2f}" for a in acc) + f" | mean {np.mean(acc)*100:.2f} +/- {np.std(acc, ddof=1)*100:.2f}"
          f" | bal mean {np.mean(bal)*100:.2f} | ensemble {accuracy_score(labels, ens[m])*100:.2f} (bal {balanced_accuracy_score(labels, ens[m])*100:.2f})")
print("== paired sign tests (disagreements: only-HEM-correct vs only-DMoN-correct)")
for r in range(3):
    h, d = pred["HEM"][r] == labels, pred["DMoN"][r] == labels
    a, b = int((h & ~d).sum()), int((~h & d).sum())
    print(f"rep{r}: {a} vs {b}, p = {binomtest(a, a+b).pvalue:.1e}")
h, d = ens["HEM"] == labels, ens["DMoN"] == labels
a, b = int((h & ~d).sum()), int((~h & d).sum()); print(f"ensembles: {a} vs {b}, p = {binomtest(a, a+b).pvalue:.1e}")
print("== per-class F1 (ensembles)")
fh, fd = f1_score(labels, ens["HEM"], labels=cls, average=None), f1_score(labels, ens["DMoN"], labels=cls, average=None)
df = pd.DataFrame({"n": pd.Series(labels).value_counts().reindex(cls).values, "F1_HEM": fh.round(3), "F1_DMoN": fd.round(3)}, index=cls)
df["diff"] = (df.F1_HEM - df.F1_DMoN).round(3); print(df.sort_values("diff", ascending=False).to_string())
print("== tuned DMoN hyperparameters")
for r, (_, _, _, c) in enumerate(R["DMoN"]): print(f"rep{r}: lr={c['lr']:.3g} wd={c['weight_decay']:.3g} lambda_mod={c['lambda_modularity']:.3g} lambda_col={c['lambda_collapse']:.3g}")
