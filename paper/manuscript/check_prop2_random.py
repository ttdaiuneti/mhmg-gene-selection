"""Control for check_prop2.py: same folds, B = random gene sets of the same size as the MHMG reduct (seed 0)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
E1 = Path(__file__).resolve().parent.parent / "e1"; sys.path.insert(0, str(E1))
import data as DA  # noqa: E402
res = pd.read_csv(E1 / "raw_results.csv"); rows = []; rng = np.random.default_rng(0)
for name in DA.NAMES:
    X, y = DA.load(name); Xs = MinMaxScaler().fit_transform(X)
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y), 1):
        k = len(str(res[(res.dataset == name) & (res.rep == 0) & (res.fold == fold) & (res.method == "MHMG")].reduct.iloc[0]).split("|"))
        for rnd in range(5):
            Z = Xs[:, rng.choice(X.shape[1], k, replace=False)]; n = len(y)
            D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(2); en = y[:, None] != y[None, :]
            S = np.zeros(n, bool); S[te] = True
            changed = np.where(en & ~S[None, :], D, np.inf).min(1) != np.where(en, D, np.inf).min(1)
            rows.append(dict(dataset=name, fold=fold, frac_changed=changed[~S].mean(), s_over=len(te) / (n - 1)))
d = pd.DataFrame(rows); d.to_csv(Path(__file__).with_name("check_prop2_random.csv"), index=False)
print("random B: median fraction changed", round(d.frac_changed.median(), 3), "| median s/(n-1)", round(d.s_over.median(), 3))
