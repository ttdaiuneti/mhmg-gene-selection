"""Numerical check of Proposition 2: removing the test fold S changes delta_B(x) for a retained x iff every nearest
enemy of x (on the full data) lies in S. MHMG reducts of E1 (rep 0); distances on the reduct, full-data scaling."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
E1 = Path(__file__).resolve().parent.parent / "e1"; sys.path.insert(0, str(E1))
import data as DA  # noqa: E402
res = pd.read_csv(E1 / "raw_results.csv"); rows = []
for name in DA.NAMES:
    X, y = DA.load(name); Xs = MinMaxScaler().fit_transform(X)
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y), 1):
        B = [int(v) for v in str(res[(res.dataset == name) & (res.rep == 0) & (res.fold == fold) & (res.method == "MHMG")].reduct.iloc[0]).split("|")]
        Z = Xs[:, B]; n = len(y)
        D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(2)
        en = y[:, None] != y[None, :]
        dfull = np.where(en, D, np.inf).min(1)
        S = np.zeros(n, bool); S[te] = True
        dtr = np.where(en & ~S[None, :], D, np.inf).min(1)
        nearest = en & (D == dfull[:, None])
        predicted = (nearest & ~S[None, :]).sum(1) == 0          # all nearest enemies removed
        changed = dtr != dfull
        keep = ~S
        rows.append(dict(dataset=name, fold=fold, agree=bool((predicted[keep] == changed[keep]).all()),
                         frac_changed=changed[keep].mean(), frac_unique=(nearest.sum(1)[keep] == 1).mean()))
d = pd.DataFrame(rows); d.to_csv(Path(__file__).with_name("check_prop2.csv"), index=False)
print("rows", len(d), "| iff holds on all:", d.agree.all(), "| median fraction of radii changed:",
      round(d.frac_changed.median(), 3), "| median fraction with unique nearest enemy:", round(d.frac_unique.median(), 3))
