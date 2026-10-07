"""mhmg-exact v1 (research-commons canonical, 2026-10-06). Source: NCS-Tram/MHMG-AttrChange/v2/attrinc2.py.
See README.md for the verification record.
"""
"""P0 greedy MHMG with a recorded path, and exact updates when attributes are added or deleted.
Greedy (P0): gamma = E / K_d rounded to `digits`; candidates in column order; first maximum wins; stop unless strict gain.
Columns are identified by integer ids; new columns get ids larger than all old ones (appended at the end)."""
import numpy as np


def kd(y):
    _, c = np.unique(y, return_counts=True); return float((c * np.log2(c + 1)).sum())


class Evaluator:
    def __init__(self, X, y, digits=2, chunk=256):
        self.X, self.y, self.digits, self.chunk = X, y, digits, chunk
        self.en = y[:, None] != y[None, :]; self.K = kd(y); self.evals = 0

    def D(self, R):
        D = np.zeros((len(self.y),) * 2)
        for a in R:
            d = self.X[:, a][:, None] - self.X[:, a][None, :]; D += d * d
        return D

    def scores(self, D, cols):
        """Rounded gamma of R + {a} for each a in cols (vectorised in chunks)."""
        out = np.empty(len(cols)); cols = np.asarray(cols, int)
        for s in range(0, len(cols), self.chunk):
            c = cols[s:s + self.chunk]; V = self.X[:, c].T                     # (k, n)
            T = np.round(D[None] + (V[:, :, None] - V[:, None, :]) ** 2, 12)   # (k, n, n); tie-safe
            r = np.where(self.en[None], T, np.inf).min(2)
            E = np.log2((T < r[:, :, None]).sum(2) + 1).sum(1)
            out[s:s + len(c)] = np.round(E / self.K, self.digits)
        self.evals += len(cols); return out


def greedy(ev, cols, R0=(), g0=0.0):
    """P0 greedy over candidate ids `cols` (sorted), starting from prefix R0 with score g0.
    Returns path: list of (winner, g) and final record (g_stop_best) = best rounded score at the stopping prefix."""
    R, g_best, rem, path = list(R0), g0, sorted(set(cols) - set(R0)), []
    while rem:
        s = ev.scores(ev.D(R), rem); i = int(np.argmax(s))                    # argmax = first maximum
        if not s[i] > g_best:
            return path, float(s[i])
        R.append(rem[i]); path.append((rem[i], float(s[i]))); g_best = float(s[i]); rem.pop(i)
    return path, -1.0


def add_attrs(ev, cols_old, path, stop_best, new):
    """Exact update when columns `new` (ids > all of cols_old) are added. Returns (path, stop_best)."""
    new = sorted(new); R = []
    for j, (w, g) in enumerate(path):
        s = ev.scores(ev.D(R), new); i = int(np.argmax(s))
        if s[i] > g:                                                          # a new column beats the recorded winner
            R2 = R + [new[i]]; p2, sb = greedy(ev, sorted(cols_old + new), R2, float(s[i]))
            return path[:j] + [(new[i], float(s[i]))] + p2, sb
        R.append(w)
    g_last = path[-1][1] if path else 0.0
    s = ev.scores(ev.D(R), new); i = int(np.argmax(s))
    if s[i] > g_last:                                                         # stopping rule no longer holds
        p2, sb = greedy(ev, sorted(cols_old + new), R + [new[i]], float(s[i]))
        return path + [(new[i], float(s[i]))] + p2, sb
    return path, max(stop_best, float(s[i]))


def del_attrs(ev, cols_old, path, stop_best, gone):
    """Exact update when columns `gone` are deleted."""
    gone = set(gone); cols = sorted(set(cols_old) - gone)
    for j, (w, g) in enumerate(path):
        if w in gone:
            R = [a for a, _ in path[:j]]; g0 = path[j - 1][1] if j else 0.0
            p2, sb = greedy(ev, cols, R, g0); return path[:j] + p2, sb
    return path, None                                                         # unchanged; stop_best may only drop


def select_features(X, y, digits=2):
    """MHMG-RS reduct (P0): X min-max scaled to [0, 1]. Returns the selected column indices in selection order."""
    ev = Evaluator(np.asarray(X, float), np.asarray(y), digits=digits, chunk=max(1, int(2e7 / len(y) ** 2)))
    return [w for w, _ in greedy(ev, list(range(X.shape[1])))[0]]


def continued_path(X, y, q, digits=2):
    """Greedy P0 path continued past the stopping rule until q attributes (as used for accuracy curves)."""
    ev = Evaluator(np.asarray(X, float), np.asarray(y), digits=digits, chunk=max(1, int(2e7 / len(y) ** 2)))
    R, rem = [], list(range(X.shape[1]))
    for _ in range(min(q, X.shape[1])):
        s = ev.scores(ev.D(R), rem); i = int(np.argmax(s)); R.append(rem.pop(i))
    return R
