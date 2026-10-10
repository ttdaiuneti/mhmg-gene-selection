"""E5 analysis (bootstrap CI; descriptive Wilcoxon/Holm, size, stability, time, SVM). Inputs: raw_e5.csv and e1/raw_results.csv (MHMG). Output: e5_summary.csv and printed report."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = Path(__file__).resolve().parent
TUNED = ["L1-cv10", "F-cv10", "FCQ-cv10", "ReliefF-cv10", "RFE-cv10"]

e5 = pd.read_csv(HERE / "raw_e5.csv")
assert len(e5) == 17 * 15 * len(TUNED), f"incomplete: {len(e5)} rows"
e1 = pd.read_csv(HERE.parent / "e1" / "raw_results.csv"); e1 = e1[e1.method == "MHMG"]
d = pd.concat([e1, e5], ignore_index=True)
acc = d.pivot_table(index="dataset", columns="method", values="knn_acc")
svm = d.pivot_table(index="dataset", columns="method", values="svm_acc")
size = d.pivot_table(index="dataset", columns="method", values="size")
tim = d.groupby(["dataset", "method"]).time_s.mean().unstack().median()


def nogueira(sets, m):
    Z = np.zeros((len(sets), m))
    for i, s in enumerate(sets):
        Z[i, s] = 1
    kb = Z.sum(1).mean()
    return 1 - Z.var(0, ddof=1).mean() / (kb / m * (1 - kb / m))


stab = pd.Series({(ds, mt): nogueira([list(map(int, str(r).split("|"))) for r in g.reduct], int(g.n_attr.iloc[0]))
                  for (ds, mt), g in d.groupby(["dataset", "method"])}).unstack().median()
rng = np.random.default_rng(0)
idx = rng.integers(0, len(acc), size=(10000, len(acc)))
rows, ps = [], {}
for c in TUNED:
    diff = (acc.MHMG - acc[c]).values
    boot = diff[idx].mean(1); lo, hi = np.percentile(boot, [2.5, 97.5])
    ps[c] = wilcoxon(acc.MHMG, acc[c], zero_method="zsplit").pvalue
    rows.append(dict(method=c, mean_diff=diff.mean(), ci_lo=lo, ci_hi=hi, noninferior=lo > -0.02, med_diff=np.median(diff),
                     wins=int((diff > 0).sum()), ties=int((diff == 0).sum()), losses=int((diff < 0).sum()), p=ps[c]))
order = sorted(ps, key=ps.get); holm, run = {}, 0.0
for i, c in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * ps[c])); holm[c] = run
res = pd.DataFrame(rows).set_index("method"); res["p_holm"] = pd.Series(holm)
res["mean_acc"] = acc[TUNED].mean(); res["mean_svm"] = svm[TUNED].mean(); res["med_size"] = size[TUNED].median()
res["stab"] = stab[TUNED]; res["time"] = tim[TUNED]
res.to_csv(HERE / "e5_summary.csv")
print(res.round(3).to_string())
print(f"\nMHMG: mean 3-NN {acc.MHMG.mean():.3f}, SVM {svm.MHMG.mean():.3f}, median size {size.MHMG.median():.1f}, "
      f"stab {stab.MHMG:.3f}, time {tim.MHMG:.2f}s")
print("H8 (vs L1-cv10, CI lower bound > -0.02):", bool(res.loc["L1-cv10", "noninferior"]))
print("H9:", res.loc[TUNED[1:], "noninferior"].to_dict())
print("chosen q:\n", e5.groupby("method").config.describe()[["mean", "50%"]].to_string())
