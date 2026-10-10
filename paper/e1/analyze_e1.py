"""Analysis of the main comparison: paired differences, stability and Friedman tests. Reads raw_results.csv only."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DA  # noqa: E402

d = pd.read_csv(HERE / "raw_results.csv")
assert d.groupby("dataset").size().eq(15 * 11).all(), d.groupby("dataset").size()
mean = d.pivot_table(index="dataset", columns="method", values="knn_acc")
svm = d.pivot_table(index="dataset", columns="method", values="svm_acc")
size = d.pivot_table(index="dataset", columns="method", values="size")
nrs_cols = ["NRS_0.1", "NRS_0.2", "NRS_0.3"]
best = mean[nrs_cols].idxmax(axis=1)
mean["NRS*"] = [mean.loc[i, c] for i, c in best.items()]
svm["NRS*"] = [svm.loc[i, c] for i, c in best.items()]
size["NRS*"] = [size.loc[i, c] for i, c in best.items()]
out = []
P = lambda s: out.append(s) or print(s)

# H1
dA = mean.MHMG - mean.ALL
h1a = (dA >= -0.02).mean(); h1b = (dA[DA.NEW] >= -0.02).sum()
P(f"H1: MHMG>=ALL-0.02 on {(dA >= -0.02).sum()}/17 ({h1a:.0%}; need >=70%) and {h1b}/9 NEW (need >=6) -> "
  f"{'PASS' if h1a >= 0.7 and h1b >= 6 else 'FAIL'}")
# H2
same_k = ["F", "FCQ", "ReliefF", "L1", "RFE"]
ps = {c: wilcoxon(mean.MHMG, mean[c], zero_method="zsplit").pvalue for c in same_k}
order = sorted(ps, key=ps.get); holm = {}
for i, c in enumerate(order):
    holm[c] = min(1.0, max([ps[o] * (len(order) - j) for j, o in enumerate(order[:i + 1])]))
for c in same_k:
    P(f"H2 vs {c}: median diff {np.median(mean.MHMG - mean[c]):+.4f} (all 17), Holm p={holm[c]:.3f}, "
      f"wins {(mean.MHMG > mean[c]).sum()}/17")
h2 = np.median((mean.MHMG - mean.FCQ)[DA.NEW])
P(f"H2-main: median MHMG-FCQ on 9 NEW = {h2:+.4f} -> {'PASS' if h2 >= 0 else 'FAIL'}")
# H3
for c in ("NRS*", "FNRS"):
    p = wilcoxon(size.MHMG, size[c], alternative="less", zero_method="zsplit").pvalue
    md = np.median(mean.MHMG - mean[c])
    P(f"H3 vs {c}: median |R| {size.MHMG.median():.1f} vs {size[c].median():.1f}, Wilcoxon(less) p={p:.4f}; "
      f"median acc diff {md:+.4f} -> {'PASS' if p < 0.05 and md >= -0.02 else 'FAIL'}")


# stability (Nogueira et al. 2018 JMLR), per method/dataset over 15 folds
def nogueira(sets, m):
    Z = np.zeros((len(sets), m))
    for i, s in enumerate(sets):
        Z[i, s] = 1
    kbar = Z.sum(1).mean()
    if kbar in (0, m):
        return np.nan
    return 1 - Z.var(0, ddof=1).mean() / (kbar / m * (1 - kbar / m))


st = {}
for (ds, meth), g in d[d.method != "ALL"].groupby(["dataset", "method"]):
    sets = [list(map(int, str(r).split("|"))) if isinstance(r, str) and r else [] for r in g.reduct]
    st[(ds, meth)] = nogueira(sets, int(g.n_attr.iloc[0]))
st = pd.Series(st).unstack()
P("Stability (median over datasets): " + ", ".join(f"{c} {st[c].median():.3f}" for c in st.columns))
# Friedman over same-k methods + MHMG + ALL
cols = ["MHMG"] + same_k + ["NRS*", "FNRS", "ALL"]
P(f"Friedman ({len(cols)} methods, 17 datasets): p={friedmanchisquare(*[mean[c] for c in cols]).pvalue:.4f}")
rk = pd.DataFrame(np.vstack([rankdata(-mean.loc[i, cols]) for i in mean.index]), columns=cols, index=mean.index)
P("Mean ranks: " + ", ".join(f"{c} {rk[c].mean():.2f}" for c in rk.mean().sort_values().index))
tab = pd.concat({"3NN": mean[cols], "size": size[cols]}, axis=1).round(3)
tab.to_csv(HERE / "summary_by_dataset.csv"); svm[cols].round(3).to_csv(HERE / "summary_svm.csv")
st.round(3).to_csv(HERE / "stability.csv")
(HERE / "analysis.txt").write_text("\n".join(out) + "\n")
print(tab.to_string())
