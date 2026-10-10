"""Check: gamma_N(C) on every training fold at eps 0.75, 1.0 and the data-scaled radius,
and whether non-empty NRS reducts at eps = 1.0 coincide with scaled attribute ranges above 1 (rounding)."""
import sys, numpy as np, pandas as pd
sys.path[:0] = ["../e1", "."]
import data as DA, run_e8 as R
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
e8 = pd.read_csv("raw_e8.csv"); A = e8[(e8.part == "A") & (e8.method == "NRS_1.0")].set_index(["dataset", "rep", "fold"])
rows = []
for name in DA.NAMES:
    X, y = DA.load(name)
    for rep in (0, 1, 2):
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y), 1):
            Xtr = MinMaxScaler().fit(X[tr]).transform(X[tr]); ytr = y[tr]; n, m = Xtr.shape
            enemy = ytr[:, None] != ytr[None, :]; full = np.zeros((n, n))
            for a in range(m):
                full += R.col(Xtr, a)
            em = R.eps_med(Xtr); r = dict(dataset=name, rep=rep, fold=fold)
            for lab, e in (("0.75", 0.75), ("1.0", 1.0), ("med", em)):
                r[f"gC_{lab}"] = (n - ((full <= e * e) & enemy).any(1).sum()) / n
            rng = Xtr.max(0) - Xtr.min(0); r["max_range_minus1"] = float(rng.max() - 1.0)
            r["nrs1_nonempty"] = int(A.loc[(name, rep, fold), "size"] > 0)
            rows.append(r)
d = pd.DataFrame(rows); d.to_csv("check_gamma_full.csv", index=False)
print(d[["gC_0.75", "gC_1.0", "gC_med"]].describe().loc[["min", "mean"]])
print("folds with gC<1:", {c: int((d[c] < 1).sum()) for c in ["gC_0.75", "gC_1.0", "gC_med"]})
print("nonempty@1.0:", int(d.nrs1_nonempty.sum()), "of which range>1:", int(((d.max_range_minus1 > 0) & (d.nrs1_nonempty == 1)).sum()),
      "| range>1 folds total:", int((d.max_range_minus1 > 0).sum()))
