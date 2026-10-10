"""E4 analysis exactly as fixed in the study protocol (H5-H7 + descriptive). Inputs: raw_e4.csv (tuned selectors) and
e1/raw_results.csv (MHMG, ALL on the same folds). Output: e4_summary.csv and printed report."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
HERE = Path(__file__).resolve().parent
TUNED = ["L1-cv", "F-cv", "FCQ-cv", "ReliefF-cv", "RFE-cv", "NRS-cv"]

e4 = pd.read_csv(HERE / "raw_e4.csv")
e1 = pd.read_csv(HERE.parent / "e1" / "raw_results.csv")
e1 = e1[e1.method.isin(["MHMG", "ALL"])]
assert len(e4) == 17 * 15 * len(TUNED), f"incomplete: {len(e4)} rows"
d = pd.concat([e1, e4], ignore_index=True)

acc = d.pivot_table(index="dataset", columns="method", values="knn_acc")
svm = d.pivot_table(index="dataset", columns="method", values="svm_acc")
size = d.pivot_table(index="dataset", columns="method", values="size")
tim = d.groupby("method").time_s.median()


def nogueira(sets, m):
    Z = np.zeros((len(sets), m))
    for i, s in enumerate(sets):
        Z[i, s] = 1
    kbar = Z.sum(1).mean()
    return np.nan if kbar in (0, m) else 1 - Z.var(0, ddof=1).mean() / (kbar / m * (1 - kbar / m))


stab = {}
for (ds, meth), g in d[d.method != "ALL"].groupby(["dataset", "method"]):
    stab[(ds, meth)] = nogueira([list(map(int, str(r).split("|"))) for r in g.reduct], int(g.n_attr.iloc[0]))
stab = pd.Series(stab).unstack()

# H5/H6: median difference >= -0.02; two-sided Wilcoxon, Holm over the six comparisons
rows, ps = [], {}
for c in TUNED:
    diff = acc.MHMG - acc[c]
    ps[c] = wilcoxon(acc.MHMG, acc[c], zero_method="zsplit").pvalue
    rows.append(dict(method=c, med_diff=diff.median(), wins=int((diff > 0).sum()), ties=int((diff == 0).sum()),
                     losses=int((diff < 0).sum()), p=ps[c], noninferior=diff.median() >= -0.02))
order = sorted(ps, key=ps.get); holm, run = {}, 0.0
for i, c in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * ps[c])); holm[c] = run
res = pd.DataFrame(rows).set_index("method"); res["p_holm"] = pd.Series(holm)
res["mean_acc"] = acc[TUNED].mean(); res["mean_svm"] = svm[TUNED].mean(); res["med_size"] = size[TUNED].median()
res["stab"] = stab[TUNED].median(); res["med_time"] = tim[TUNED]
# H7: MHMG smaller than L1-cv (one-sided)
p7 = wilcoxon(size.MHMG, size["L1-cv"], alternative="less", zero_method="zsplit").pvalue
# Friedman + Nemenyi over MHMG, tuned selectors and ALL
cols = ["MHMG"] + TUNED + ["ALL"]
fr = friedmanchisquare(*[acc[c] for c in cols]).pvalue
ranks = pd.DataFrame([rankdata(-acc.loc[i, cols]) for i in acc.index], columns=cols).mean()
cd = 3.031 * np.sqrt(len(cols) * (len(cols) + 1) / (6 * len(acc)))      # q_0.05 for 8 methods (Demsar 2006)
res.to_csv(HERE / "e4_summary.csv")
print(res.round(3).to_string())
print(f"\nMHMG: mean 3-NN {acc.MHMG.mean():.3f}, SVM {svm.MHMG.mean():.3f}, median size {size.MHMG.median():.1f}, "
      f"stab {stab.MHMG.median():.3f}, median time {tim.MHMG:.2f}s; ALL 3-NN {acc.ALL.mean():.3f}")
print(f"H5 (vs L1-cv, median diff >= -0.02): {res.loc['L1-cv', 'noninferior']}")
print("H6 non-inferior:", res.loc[TUNED[1:], "noninferior"].to_dict())
print(f"H7 (MHMG smaller than L1-cv): p = {p7:.4f} -> {p7 < 0.05}")
print(f"Friedman p = {fr:.2e}; mean ranks: {ranks.round(2).to_dict()}; CD = {cd:.3f}")
print("chosen configs:\n", e4.groupby("method").config.value_counts().to_string())
