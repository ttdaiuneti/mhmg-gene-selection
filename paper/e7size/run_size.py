import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "e1")); OUT = HERE / "raw_size.csv"; Q = (10, 20, 50)


def fnrs_path(X, y, delta_frac=0.3, theta_ratio=0.95, cap=50):
    """methods.fnrs with the selection order kept (same computations)."""
    import methods as M
    n, m = X.shape; enemy = y[:, None] != y[None, :]; full = np.zeros((n, n))
    for a in range(m):
        full += M._col(X, a)
    iu = np.triu_indices(n, 1); delta = max(delta_frac * float(np.median(np.sqrt(full[iu]))), 1e-6)
    def dep(D2):
        v = np.where(enemy, np.minimum(1.0, np.sqrt(D2) / delta), np.inf).min(1); v[~np.isfinite(v)] = 1.0; return float(v.mean())
    theta = theta_ratio * dep(full); D2, B, cur = np.zeros((n, n)), [], 0.0
    while cur < theta - 1e-9 and len(B) < cap:
        cand = [j for j in range(m) if j not in B]
        if not cand: break
        sc = [dep(D2 + M._col(X, j)) for j in cand]; bl = int(np.argmax(sc))
        if sc[bl] <= cur + 1e-12: break
        B.append(cand[bl]); D2 += M._col(X, cand[bl]); cur = sc[bl]
    return B


def job(name, rep, fold):
    warnings.filterwarnings("ignore")
    import data as DA, methods as M
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    X, y = DA.load(name); tr, te = list(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y))[fold - 1]
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2); ytr, yte = y[tr], y[te]
    t0 = time.perf_counter(); stop, mp = M.P.forward(Xtr, ytr, "orig", min_path=max(Q)); tm = time.perf_counter() - t0
    t0 = time.perf_counter(); fp = fnrs_path(Xtr, ytr); tf = time.perf_counter() - t0
    rows = []
    for meth, path, t in (("MHMG", mp, tm), ("FNRS", fp, tf)):
        for q in Q:
            B = list(path[:q])
            r = dict(dataset=name, rep=rep, fold=fold, method=meth, q=q, size=len(B), path_len=len(path), time_s=t)
            for nm, clf in (("knn", KNeighborsClassifier(3)), ("svm", SVC())):
                r[f"{nm}_acc"] = float(np.mean(clf.fit(Xtr[:, B], ytr).predict(Xte[:, B]) == yte))
            rows.append(r)
    return rows


if __name__ == "__main__":
    import data as DA
    allrows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["rep"], r["fold"]) for r in allrows}
    jobs = [(n, r, f) for n in DA.NAMES for r in (0, 1, 2) for f in range(1, 6) if (n, r, f) not in done]
    with ProcessPoolExecutor(6) as ex:
        futs = [ex.submit(job, *j) for j in jobs]; print("jobs", len(futs), flush=True)
        for fu in as_completed(futs):
            rr = fu.result(); allrows += rr; pd.DataFrame(allrows).to_csv(OUT, index=False)
            print(time.strftime("%H:%M:%S"), rr[0]["dataset"], rr[0]["rep"], rr[0]["fold"], round(rr[0]["time_s"]), flush=True)
