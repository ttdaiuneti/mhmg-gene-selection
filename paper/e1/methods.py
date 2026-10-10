"""Selectors of the main comparison. MHMG = published greedy search (src/mhmg.py); NRS and FNRS = vectorised
re-implementations of the reference NRS and FNRS code (src/nrs_fnrs.py implements the same two)."""
import sys
import numpy as np
from sklearn.feature_selection import RFE, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import mhmg as _MH  # noqa: E402  (self-contained reference MHMG, see ../../src/mhmg.py)


def mhmg(X, y):
    return _MH.select_features(X, y)


def top_f(X, y, k):
    return list(np.argsort(-np.nan_to_num(f_classif(X, y)[0]), kind="stable")[:k])


def fcq(X, y, k):
    rel = np.nan_to_num(f_classif(X, y)[0])
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    sel = [int(np.argmax(rel))]
    red = np.zeros(X.shape[1])
    while len(sel) < k:
        red += np.abs(Z.T @ Z[:, sel[-1]]) / len(y)
        sc = rel / np.maximum(red / len(sel), 1e-12)
        sc[sel] = -np.inf
        sel.append(int(np.argmax(sc)))
    return sel


def relieff(X, y, k, nn=10):
    n, m = X.shape
    D = np.zeros((n, n))
    for j in range(0, m, 2000):
        D += np.abs(X[:, None, j:j + 2000] - X[None, :, j:j + 2000]).sum(2)
    np.fill_diagonal(D, np.inf)
    cls, cnt = np.unique(y, return_counts=True)
    prior = dict(zip(cls, cnt / n))
    W = np.zeros(m)
    for i in range(n):
        same = np.where(y == y[i])[0]
        same = same[same != i]
        h = same[np.argsort(D[i, same])[:nn]]
        if len(h):
            W -= np.abs(X[h] - X[i]).mean(0) / n
        for c in cls:
            if c == y[i]:
                continue
            oth = np.where(y == c)[0]
            mm = oth[np.argsort(D[i, oth])[:nn]]
            W += prior[c] / (1 - prior[y[i]]) * np.abs(X[mm] - X[i]).mean(0) / n
    return list(np.argsort(-W, kind="stable")[:k])


def l1(X, y, k):
    c = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000).fit(X, y)
    return list(np.argsort(-np.abs(c.coef_).max(0), kind="stable")[:k])


def rfe(X, y, k):
    r = RFE(LinearSVC(max_iter=5000), n_features_to_select=k, step=0.1).fit(X, y)
    return list(np.flatnonzero(r.support_))


def _col(X, a):
    d = X[:, a][:, None] - X[:, a][None, :]
    return d * d


def nrs(X, y, delta):
    """Same algorithm as ijar-nrs NRS_Reduction.fit_reduction (positive region with dist <= delta, greedy until
    gamma(C), then backward pruning), on incrementally accumulated squared distances."""
    n, m = X.shape
    enemy = y[:, None] != y[None, :]
    d2 = delta * delta

    def pos(D2):
        return int(n - ((D2 <= d2) & enemy).any(1).sum())

    full = np.zeros((n, n))
    for a in range(m):
        full += _col(X, a)
    target = pos(full)
    D2, reduct, cur, remaining = np.zeros((n, n)), [], 0, list(range(m))
    while cur < target and remaining:
        best, bg = -1, -1
        for f in remaining:
            g = pos(D2 + _col(X, f))
            if g > bg:
                bg, best = g, f
        if best != -1 and bg > cur:
            D2 += _col(X, best); reduct.append(best); remaining.remove(best); cur = bg
        else:
            break
    final = list(reduct)
    for f in reduct:
        temp = [x for x in final if x != f]
        if temp:
            Dt = np.zeros((n, n))
            for a in temp:
                Dt += _col(X, a)
            if pos(Dt) == cur:
                final.remove(f)
    return final


def fnrs(X, y, delta_frac=0.3, theta_ratio=0.95, cap=50):
    """Same algorithm as fcfp FNRS_Reduction (triangular fuzzy relation, dependency = mean_x min_{enemy} (1 - R))."""
    n, m = X.shape
    enemy = y[:, None] != y[None, :]
    full = np.zeros((n, n))
    for a in range(m):
        full += _col(X, a)
    iu = np.triu_indices(n, 1)
    delta = max(delta_frac * float(np.median(np.sqrt(full[iu]))), 1e-6)

    def dep(D2):
        v = np.where(enemy, np.minimum(1.0, np.sqrt(D2) / delta), np.inf).min(1)
        v[~np.isfinite(v)] = 1.0
        return float(v.mean())

    theta = theta_ratio * dep(full)
    D2, B = np.zeros((n, n)), []
    cur = 0.0
    while cur < theta - 1e-9 and len(B) < cap:
        cand = [j for j in range(m) if j not in B]
        if not cand:
            break
        sc = [dep(D2 + _col(X, j)) for j in cand]
        bl = int(np.argmax(sc))
        if sc[bl] <= cur + 1e-12:
            break
        B.append(cand[bl]); D2 += _col(X, cand[bl]); cur = sc[bl]
    return sorted(B)
