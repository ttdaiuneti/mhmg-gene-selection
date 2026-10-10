"""E8-rev : additional runs (radius grid, equal size, controls, rounding, imputation). Resumable; one row per (dataset, rep, fold, method).
Parts: A wide NRS radius grid + NRS-cv8, B MHMG truncated to NRS size, C random baselines + KNN-NRS (Euclidean),
D MHMG with 3-decimal rounding, E Lymphoma with in-fold mean imputation."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, warnings, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; G = HERE.parent
sys.path[:0] = [str(G / "e1"), str(G / "external")]
OUT = HERE / "raw_e8.csv"; CAP = 1800.0
EPS_NEW = (0.4, 0.5, 0.75, 1.0, "med"); EPS_ALL = (0.1, 0.2, 0.3) + EPS_NEW
E1 = pd.read_csv(G / "e1" / "raw_results.csv"); ERS = pd.read_csv(G / "e6rs" / "raw_ers.csv")


class Timeout(Exception):
    pass


def col(X, a):
    d = X[:, a][:, None] - X[:, a][None, :]
    return d * d


def eps_med(X):
    n, m = X.shape; full = np.zeros((n, n))
    for a in range(m):
        full += col(X, a)
    return 0.3 * float(np.median(np.sqrt(full[np.triu_indices(n, 1)])))


def nrs_t(X, y, delta, t0):
    """methods.nrs (same algorithm) with a deadline; raises Timeout carrying the partial reduct."""
    n, m = X.shape; enemy = y[:, None] != y[None, :]; d2 = delta * delta
    pos = lambda D2: int(n - ((D2 <= d2) & enemy).any(1).sum())
    full = np.zeros((n, n))
    for a in range(m):
        full += col(X, a)
    target = pos(full); D2, reduct, cur, remaining = np.zeros((n, n)), [], 0, list(range(m))
    while cur < target and remaining:
        best, bg = -1, -1
        for f in remaining:
            if time.perf_counter() - t0 > CAP:
                raise Timeout(list(reduct))
            g = pos(D2 + col(X, f))
            if g > bg:
                bg, best = g, f
        if best != -1 and bg > cur:
            D2 += col(X, best); reduct.append(best); remaining.remove(best); cur = bg
        else:
            break
    final = list(reduct)
    for f in reduct:
        temp = [x for x in final if x != f]
        if temp:
            Dt = np.zeros((n, n))
            for a in temp:
                Dt += col(X, a)
            if pos(Dt) == cur:
                final.remove(f)
    return final


def acc(Xtr, ytr, Xte, yte, B):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    B = list(B) or list(range(Xtr.shape[1]))
    return tuple(float(np.mean(c.fit(Xtr[:, B], ytr).predict(Xte[:, B]) == yte)) for c in (KNeighborsClassifier(3), SVC()))


def knn_dep_euclid(X, y, B, k=3):
    """wang_knn_nrs.knn_dependency with the Euclidean metric in place of Chebyshev (part C)."""
    n = X.shape[0]
    if len(B) == 0:
        return 1.0 if len(set(y)) == 1 else 0.0
    Xb = X[:, B]; s = (Xb * Xb).sum(1); D = s[:, None] + s[None, :] - 2 * Xb @ Xb.T
    np.fill_diagonal(D, np.inf); nn = np.argpartition(D, k, axis=1)[:, :k]
    return float(np.mean([np.all(y[nn[i]] == y[i]) for i in range(n)]))


def greedy_target(score, m, target, t0):
    """Forward greedy of e6rs/run_ers.greedy: add argmax while it strictly increases, stop at the full-set value."""
    S, cur = [], score([])
    while cur < target - 1e-12:
        best, bv = None, -np.inf
        for j in range(m):
            if j in S:
                continue
            if time.perf_counter() - t0 > CAP:
                raise Timeout(list(S))
            v = score(S + [j])
            if v > bv:
                bv, best = v, j
        if best is None or bv <= cur + 1e-12:
            break
        S.append(best); cur = bv
    return S


def split(name, rep, fold, X, y):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y))[fold - 1]


def scaled(X, tr, te):
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler().fit(X[tr]); return sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)


def row(name, rep, fold, part, meth, status, B, t, a, **kw):
    return dict(dataset=name, rep=rep, fold=fold, part=part, method=meth, status=status, size=len(B), time_s=t,
                knn_acc=a[0], svm_acc=a[1], reduct="|".join(map(str, B)), **kw)


def job(name, rep, fold, part):
    warnings.filterwarnings("ignore")
    import data as DA, run_pilot as P
    X, y = DA.load(name); tr, te = split(name, rep, fold, X, y); Xtr, Xte = scaled(X, tr, te); ytr, yte = y[tr], y[te]
    e1 = E1[(E1.dataset == name) & (E1.rep == rep) & (E1.fold == fold)].set_index("method")
    rows = []
    if part == "A":
        em = eps_med(Xtr)
        for e in EPS_NEW:
            d = em if e == "med" else e; t0 = time.perf_counter(); st = "ok"
            try:
                B = nrs_t(Xtr, ytr, d, t0)
            except Timeout as ex:
                B, st = ex.args[0], "timeout"
            st = "empty" if (st == "ok" and not B) else st
            rows.append(row(name, rep, fold, "A", f"NRS_{e}", st, B, time.perf_counter() - t0,
                            acc(Xtr, ytr, Xte, yte, B) if B else (np.nan, np.nan), eps=d))
        # NRS-cv8: inner stratified 3-fold CV on the training part, 3-NN criterion, ties -> smaller radius
        from sklearn.model_selection import StratifiedKFold
        t0 = time.perf_counter(); sc = {e: [] for e in EPS_ALL}; st = "ok"
        try:
            for itr, iva in StratifiedKFold(3, shuffle=True, random_state=0).split(Xtr, ytr):
                Xi, Xv = scaled(Xtr, itr, iva); emi = eps_med(Xi)
                for e in EPS_ALL:
                    B = nrs_t(Xi, ytr[itr], emi if e == "med" else e, t0)
                    sc[e].append(acc(Xi, ytr[itr], Xv, ytr[iva], B)[0] if B else np.nan)
        except Timeout:
            st = "timeout"
        if st == "ok":
            ok = [e for e in EPS_ALL if not np.isnan(sc[e]).any()]
            ech = max(ok, key=lambda e: (np.mean(sc[e]), -EPS_ALL.index(e)))
            src = [r for r in rows if r["method"] == f"NRS_{ech}"]
            B = src[0]["reduct"].split("|") if src else e1.loc[f"NRS_{ech}", "reduct"].split("|")
            B = [int(b) for b in B if str(b) != "" and str(b) != "nan"]
            rows.append(row(name, rep, fold, "A", "NRS-cv8", st, B, time.perf_counter() - t0, acc(Xtr, ytr, Xte, yte, B), eps_choice=str(ech)))
        else:
            rows.append(row(name, rep, fold, "A", "NRS-cv8", st, [], time.perf_counter() - t0, (np.nan, np.nan)))
    elif part == "B":
        qs = {e: int(e1.loc[f"NRS_{e}", "size"]) for e in (0.1, 0.2, 0.3)}
        t0 = time.perf_counter(); _, path = P.forward(Xtr, ytr, "orig", min_path=max(qs.values())); t = time.perf_counter() - t0
        for e, q in qs.items():
            B = list(path[:q]); st = "ok" if q > 0 else "q0"  # E1 has no empty NRS reduct
            rows.append(row(name, rep, fold, "B", f"MHMG@NRS_{e}", st, B, t, acc(Xtr, ytr, Xte, yte, B) if q > 0 else (np.nan, np.nan), q=q))
    elif part == "C":
        rng = np.random.default_rng(zlib.crc32(f"{name}|{rep}|{fold}".encode()))
        q = max(1, int(e1.loc["MHMG", "size"])); a = np.mean([acc(Xtr, ytr, Xte, yte, rng.choice(Xtr.shape[1], q, replace=False)) for _ in range(10)], 0)
        rows.append(row(name, rep, fold, "C", "Random@MHMG", "ok", list(range(q)), 0.0, tuple(a), q=q))
        er = ERS[(ERS.dataset == name) & (ERS.rep == rep) & (ERS.fold == fold) & (ERS.method == "MHMG-pf")]
        q80 = max(1, int(er["size"].iloc[0])); top = np.argsort(-Xtr.var(0), kind="stable")[:80]
        a = np.mean([acc(Xtr, ytr, Xte, yte, rng.choice(top, q80, replace=False)) for _ in range(10)], 0)
        rows.append(row(name, rep, fold, "C", "Random80@MHMG-80", "ok", list(range(q80)), 0.0, tuple(a), q=q80))
        t0 = time.perf_counter(); m = Xtr.shape[1]; st = "ok"
        try:
            tgt = knn_dep_euclid(Xtr, ytr, list(range(m)))
            B = greedy_target(lambda S: knn_dep_euclid(Xtr, ytr, S), m, tgt, t0)
        except Timeout as ex:
            B, st = ex.args[0], "timeout"
        rows.append(row(name, rep, fold, "C", "KNN-NRS-E", st, B, time.perf_counter() - t0, acc(Xtr, ytr, Xte, yte, B)))
    elif part == "D":
        import check_rounding as CR
        t0 = time.perf_counter(); B = CR.forward(Xtr, ytr, 3)
        rows.append(row(name, rep, fold, "D", "MHMG-3dp", "ok", B, time.perf_counter() - t0, acc(Xtr, ytr, Xte, yte, B)))
    elif part == "E":
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=1101, as_frame=True, parser="auto")
        Xr = d.data.apply(pd.to_numeric, errors="coerce"); Xr = Xr.loc[:, Xr.nunique() > 1].values.astype(float)
        mu = np.nanmean(Xr[tr], 0); mu = np.where(np.isfinite(mu), mu, 0.0)
        Xf = np.where(np.isnan(Xr), mu[None, :], Xr); keep = np.ptp(Xf[tr], 0) > 0
        Xtr2, Xte2 = scaled(Xf[:, keep], tr, te)
        t0 = time.perf_counter(); B = P.forward(Xtr2, ytr, "orig")[0]
        rows.append(row(name, rep, fold, "E", "MHMG-inF", "ok", B, time.perf_counter() - t0, acc(Xtr2, ytr, Xte2, yte, B)))
        for e in (0.1, 0.2, 0.3):
            t0 = time.perf_counter(); B = nrs_t(Xtr2, ytr, e, t0)
            rows.append(row(name, rep, fold, "E", f"NRS_{e}-inF", "ok", B, time.perf_counter() - t0, acc(Xtr2, ytr, Xte2, yte, B)))
    return rows


if __name__ == "__main__":
    import data as DA
    parts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A", "B", "C", "D", "E"]
    allrows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["rep"], r["fold"], r["part"]) for r in allrows}
    jobs = [(n, r, f, p) for p in parts for n in DA.NAMES for r in (0, 1, 2) for f in range(1, 6)
            if (n, r, f, p) not in done and (p != "E" or n == "lymphoma_2classes")]
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as ex:
        futs = [ex.submit(job, *j) for j in jobs]; print("jobs", len(futs), flush=True)
        for fu in as_completed(futs):
            rr = fu.result(); allrows += rr; pd.DataFrame(allrows).to_csv(OUT, index=False)
            print(time.strftime("%H:%M:%S"), rr[0]["part"], rr[0]["dataset"], rr[0]["rep"], rr[0]["fold"], flush=True)
