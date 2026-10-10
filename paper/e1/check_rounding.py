"""Sensitivity of MHMG to its stopping tolerance (review round 1, S5): the published search rounds gamma to 2 decimals;
here the same search with 3 decimals on the 85 folds of the first repetition. Compares subset size and 3-NN accuracy
with the reported 2-decimal runs (raw_results.csv). Exploratory."""
import os
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external"))
import run_pilot as P  # noqa: E402
import data as DA  # noqa: E402


def forward(X, y, digits):
    """Copy of run_pilot.forward (crit 'orig') with the rounding precision as a parameter."""
    fn = P.CRIT["orig"]; n, m = X.shape; D2 = np.zeros((n, n)); reduct, g_best = [], 0.0; remaining = list(range(m))
    while remaining:
        g_loc, best = -1.0, None
        for f in remaining:
            d = X[:, f][:, None] - X[:, f][None, :]
            g = round(fn(P.granule_sizes(D2 + d * d, y), y), digits)
            if g > g_loc:
                g_loc, best = g, f
        if not g_loc > g_best:
            break
        d = X[:, best][:, None] - X[:, best][None, :]; D2 = D2 + d * d
        reduct.append(best); remaining.remove(best); g_best = g_loc
    return reduct


def job(args):
    warnings.filterwarnings("ignore")
    name, fold, tr, te = args
    X, y = DA.load(name); sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    B = forward(Xtr, y[tr], 3) or list(range(X.shape[1]))
    acc = KNeighborsClassifier(3).fit(Xtr[:, B], y[tr]).score(Xte[:, B], y[te])
    return dict(dataset=name, fold=fold, size3=len(B), acc3=acc)


if __name__ == "__main__":
    jobs = []
    for name in DA.NAMES:
        X, y = DA.load(name)
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y), 1):
            jobs.append((name, fold, tr, te))
    with ProcessPoolExecutor(5) as ex:
        res = pd.DataFrame(list(ex.map(job, jobs)))
    raw = pd.read_csv("raw_results.csv"); raw = raw[(raw.method == "MHMG") & (raw.rep == 0)][["dataset", "fold", "size", "knn_acc"]]
    d = res.merge(raw, on=["dataset", "fold"]); d.to_csv("check_rounding.csv", index=False)
    ds = d.groupby("dataset").mean(numeric_only=True)
    print(f"median size: 2 decimals {ds['size'].median():.1f}, 3 decimals {ds.size3.median():.1f}")
    print(f"mean 3-NN: 2 decimals {ds.knn_acc.mean():.3f}, 3 decimals {ds.acc3.mean():.3f}")
