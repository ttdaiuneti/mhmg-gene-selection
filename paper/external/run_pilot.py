"""MHMG-RS criterion vs class-balanced MHMG criterion, same forward search.

The search reproduces the published MHMG-RS reducer exactly
(MinMax on the training data, Euclidean distance, gamma rounded to 2 decimals, first strictly-better
candidate wins, stop when the rounded gamma does not increase); only the criterion differs.
Distances are updated incrementally (squared distances), which does not change any granule.
Output: pilot/results.csv (one row per dataset x fold x method)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import recall_score

HERE = Path(__file__).resolve().parent
NAMES = ("ecoli", "yeast_me2", "libras_move", "oil", "spectrometer", "us_crime", "ozone_level",
         "wine_quality", "abalone", "yeast_ml8")


def granule_sizes(D2, y):
    """|G_B(x)| for every x: points strictly closer than the nearest enemy (squared distances)."""
    n = len(y)
    sizes = np.empty(n)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        enemy = np.where(y != c)[0]
        d2e = D2[np.ix_(idx, enemy)].min(axis=1)
        sizes[idx] = (D2[idx] < d2e[:, None]).sum(axis=1)
    return sizes


def gamma_orig(sizes, y):
    _, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    return np.log2(sizes + 1).sum() / np.log2(cnt[inv] + 1).sum()


def gamma_bal(sizes, y):
    vals = []
    for c in np.unique(y):
        k = y == c
        vals.append(np.log2(sizes[k] + 1).sum() / (k.sum() * np.log2(k.sum() + 1)))
    return float(np.mean(vals))


CRIT = {"orig": gamma_orig, "bal": gamma_bal}


def forward(X, y, crit, min_path=0):
    """MHMG-RS forward search. Returns (reduct at the stopping rule, greedy path continued, ignoring the
    stopping rule, until it has at least `min_path` attributes or all attributes are used)."""
    fn = CRIT[crit]
    n, m = X.shape
    D2 = np.zeros((n, n))
    reduct, path, g_best, stopped = [], [], 0.0, None
    remaining = list(range(m))
    while remaining:
        if stopped is not None and len(path) >= min_path:
            break
        g_loc, best = -1.0, None
        for f in remaining:
            d = X[:, f][:, None] - X[:, f][None, :]
            g = round(fn(granule_sizes(D2 + d * d, y), y), 2)
            if g > g_loc:
                g_loc, best = g, f
        if stopped is None and not g_loc > g_best:
            stopped = list(reduct)
            if len(path) >= min_path:
                break
        d = X[:, best][:, None] - X[:, best][None, :]
        D2 = D2 + d * d
        path.append(best)
        remaining.remove(best)
        if stopped is None:
            reduct.append(best)
            g_best = g_loc
    return (stopped if stopped is not None else reduct), path


def metrics(yt, yp):
    rec = recall_score(yt, yp, average=None, labels=np.unique(yt), zero_division=0)
    return float(np.sqrt(np.prod(rec)) if len(rec) == 2 else np.prod(rec) ** (1 / len(rec))), float(rec.min()), float(np.mean(yt == yp))


def evaluate(Xtr, ytr, Xte, yte, B):
    out = {}
    for nm, clf in (("knn", KNeighborsClassifier(n_neighbors=3)), ("svm", SVC(kernel="rbf"))):
        p = clf.fit(Xtr[:, B], ytr).predict(Xte[:, B])
        g, w, a = metrics(yte, p)
        out.update({f"{nm}_gmean": g, f"{nm}_worst": w, f"{nm}_acc": a})
    return out


def run_fold(name, fold, X, y, tr, te):
    sc = MinMaxScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    rows = []
    t0 = time.perf_counter()
    Rb, _ = forward(Xtr, y[tr], "bal")
    tb = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ro, path = forward(Xtr, y[tr], "orig", min_path=len(Rb))
    to = time.perf_counter() - t0
    Rk = path[:len(Rb)]
    for meth, B, t in (("MHMG", Ro, to), ("MHMG-bal", Rb, tb), ("MHMG@k", Rk, np.nan)):
        if len(B) == 0:
            B = list(range(X.shape[1]))
        r = dict(dataset=name, fold=fold, method=meth, size=len(B), reduct="|".join(map(str, B)), time_s=t,
                 n_attr=X.shape[1])
        r.update(evaluate(Xtr, y[tr], Xte, y[te], B))
        rows.append(r)
    return rows


def main():
    from imblearn.datasets import fetch_datasets
    D = fetch_datasets(filter_data=NAMES)
    jobs = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
        for name in NAMES:
            X, y = D[name].data.astype(float), D[name].target
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y), 1):
                jobs.append(ex.submit(run_fold, name, fold, X, y, tr, te))
        rows = []
        for j in as_completed(jobs):
            rows += j.result()
            print(rows[-1]["dataset"], rows[-1]["fold"], flush=True)
    pd.DataFrame(rows).sort_values(["dataset", "fold", "method"]).to_csv(HERE / "results.csv", index=False)


if __name__ == "__main__":
    main()
