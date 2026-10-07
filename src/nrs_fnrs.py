"""nrs-fnrs-fast v1 (research-commons canonical, 2026-10-06). Source: NCS-Tram/MHMG-Gene/e1/methods.py (nrs, fnrs) and
e7size/run_size.py (fnrs_path). Vectorised re-implementations of ijar-nrs NRS_Reduction and fcfp FNRS_Reduction. X min-max
scaled. See README.md.
"""
import numpy as np


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


def fnrs_path(X, y, delta_frac=0.3, theta_ratio=0.95, cap=50):
    """methods.fnrs with the selection order kept (same computations)."""
    n, m = X.shape; enemy = y[:, None] != y[None, :]; full = np.zeros((n, n))
    for a in range(m):
        full += _col(X, a)
    iu = np.triu_indices(n, 1); delta = max(delta_frac * float(np.median(np.sqrt(full[iu]))), 1e-6)
    def dep(D2):
        v = np.where(enemy, np.minimum(1.0, np.sqrt(D2) / delta), np.inf).min(1); v[~np.isfinite(v)] = 1.0; return float(v.mean())
    theta = theta_ratio * dep(full); D2, B, cur = np.zeros((n, n)), [], 0.0
    while cur < theta - 1e-9 and len(B) < cap:
        cand = [j for j in range(m) if j not in B]
        if not cand: break
        sc = [dep(D2 + _col(X, j)) for j in cand]; bl = int(np.argmax(sc))
        if sc[bl] <= cur + 1e-12: break
        B.append(cand[bl]); D2 += _col(X, cand[bl]); cur = sc[bl]
    return B
