"""E3 analysis (fixed before results, the study protocol). rho = 0 rows from e1/raw_results.csv (same folds and code)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e1"))
import data as DA  # noqa: E402
METH = ["MHMG", "F", "FCQ", "ReliefF", "L1", "ALL"]
n3 = pd.read_csv(HERE / "raw_noise.csv")
assert n3.groupby(["dataset", "rho"]).size().eq(15 * len(METH)).all()
e1 = pd.read_csv(HERE.parent / "e1" / "raw_results.csv"); e1 = e1[e1.method.isin(METH)].assign(rho=0.0)
d = pd.concat([e1, n3], ignore_index=True)
acc = d.groupby(["dataset", "rho", "method"]).knn_acc.mean().unstack()
out = []
P = lambda s: out.append(s) or print(s)
for rho in (0.0, 0.1, 0.2):
    a = acc.xs(rho, level="rho")
    P(f"rho={rho}: mean 3-NN " + ", ".join(f"{m} {a[m].mean():.3f}" for m in METH))
drop = acc.xs(0.2, level="rho") - acc.xs(0.0, level="rho")
P("mean drop 0->0.2: " + ", ".join(f"{m} {drop[m].mean():+.3f}" for m in METH))
ps = {}
for x in ("L1", "FCQ"):
    diff = drop.MHMG - drop[x]
    ps[x] = wilcoxon(drop.MHMG, drop[x], alternative="less", zero_method="zsplit").pvalue
    P(f"H4 vs {x}: median (drop_MHMG - drop_{x}) {diff.median():+.4f}, MHMG drops more on {(diff < 0).sum()}/17, one-sided p {ps[x]:.4f}")
order = sorted(ps, key=ps.get)
holm = {c: min(1, max(ps[o] * (len(order) - j) for j, o in enumerate(order[:i + 1]))) for i, c in enumerate(order)}
ok = all(holm[x] < 0.05 and np.median(drop.MHMG - drop[x]) < 0 for x in ("L1", "FCQ"))
P(f"H4 (Holm): {holm} -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
sz = d[d.method == "MHMG"].groupby(["dataset", "rho"])["size"].mean().unstack()
P("MHMG |R| median by rho: " + ", ".join(f"{c} {sz[c].median():.1f}" for c in sz.columns))
mech = n3[n3.method == "MHMG"].groupby(["dataset", "rho"])[["ne_flipped", "share_flipped"]].mean()
P("mechanism (median over datasets): " + mech.groupby(level="rho").median().round(3).to_string().replace("\n", " | "))


def nogueira(sets, m):
    Z = np.zeros((len(sets), m))
    for i, s in enumerate(sets):
        Z[i, s] = 1
    kb = Z.sum(1).mean()
    return np.nan if kb in (0, m) else 1 - Z.var(0, ddof=1).mean() / (kb / m * (1 - kb / m))


st = {}
for (ds, rho, meth), g in d[d.method != "ALL"].groupby(["dataset", "rho", "method"]):
    sets = [list(map(int, str(r).split("|"))) if isinstance(r, str) and r else [] for r in g.reduct]
    st[(ds, rho, meth)] = nogueira(sets, int(g.n_attr.iloc[0]))
st = pd.Series(st).unstack()
P("Nogueira median by rho:\n" + st.groupby(level=1).median().round(3).to_string())
acc.round(4).to_csv(HERE / "acc_by_rho.csv"); (HERE / "analysis_e3.txt").write_text("\n".join(out) + "\n")
