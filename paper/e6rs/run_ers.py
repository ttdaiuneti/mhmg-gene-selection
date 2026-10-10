import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; G = HERE.parent; X = G / "external"   # re-implementations of the compared models (see external/README.md)
sys.path[:0] = [str(G / "e1"), str(X), str(X / "wknrs")]
OUT = HERE / "raw_ers.csv"; CAP = 1800.0
E1 = pd.read_csv(G / "e1" / "raw_results.csv")


class Timeout(Exception):
    pass


def greedy(score, m, target=None, sig_positive=True, t0=None):
    """Forward greedy: add argmax score while it strictly increases; stop at target if given."""
    R, cur = [], 0.0
    while len(R) < m:
        if target is not None and R and cur >= target - 1e-12:
            break
        best, bg = None, -1.0
        for a in range(m):
            if a in R:
                continue
            if time.perf_counter() - t0 > CAP:
                raise Timeout(R)
            g = score(R + [a])
            if g > bg:
                bg, best = g, a
        if best is None or bg <= cur + 1e-12:
            break
        R.append(best); cur = bg
    return R


def job(name, rep, fold, meth):
    warnings.filterwarnings("ignore")
    import data as DA, methods as M
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    X, y = DA.load(name)
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y))[fold - 1]
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2); ytr, yte = y[tr], y[te]
    m = X.shape[1]; q = int(E1[(E1.dataset == name) & (E1.rep == rep) & (E1.fold == fold) & (E1.method == "MHMG")]["size"].iloc[0])
    pool = np.sort(np.argsort(Xtr.var(0))[::-1][:80])
    t0 = time.perf_counter(); status = "ok"
    try:
        if meth == "FNRS-q":
            R = list(M.fnrs(Xtr, ytr, cap=max(1, q)))
        elif meth == "KNN-NRS":
            import wang_knn_nrs as K
            tgt = K.knn_dependency(Xtr, ytr, np.arange(m), 3)
            R = greedy(lambda S: K.knn_dependency(Xtr, ytr, np.array(S), 3), m, target=tgt, t0=t0)
        elif meth == "WKNRS":
            import impl as W
            R = greedy(lambda S: W._wknrs_pos_fraction(Xtr[:, S], ytr, 5, 0.01), m, t0=t0) or [0]
        elif meth == "ANRS":
            import adaptive_radius_nrs as A
            R = [int(a) for a in A.AdaptiveNRSReducer().fit_reduction(X[tr], ytr)[0]]
        elif meth == "GBNRS":
            import gbnrs as GB
            Xp = Xtr[:, pool]; tgt = GB.compute_gb_dependency(Xp, ytr, list(range(80)), 0.9)[0]
            Rp = greedy(lambda S: GB.compute_gb_dependency(Xp, ytr, S, 0.9)[0], 80, target=tgt, t0=t0)
            R = [int(pool[a]) for a in Rp]
        elif meth == "MHMG-pf":
            R = [int(pool[a]) for a in M.mhmg(Xtr[:, pool], ytr)]
    except Timeout as e:
        R, status = list(e.args[0]), "timeout"
    t = time.perf_counter() - t0
    Be = R or list(range(m))
    k = float(np.mean(KNeighborsClassifier(3).fit(Xtr[:, Be], ytr).predict(Xte[:, Be]) == yte))
    s = float(np.mean(SVC().fit(Xtr[:, Be], ytr).predict(Xte[:, Be]) == yte))
    return dict(dataset=name, rep=rep, fold=fold, method=meth, status=status, size=len(R), q_mhmg=q, time_s=t, knn_acc=k, svm_acc=s,
                reduct="|".join(map(str, R)))


if __name__ == "__main__":
    import data as DA
    meths = sys.argv[1].split(",") if len(sys.argv) > 1 else ["FNRS-q", "KNN-NRS", "WKNRS", "ANRS", "GBNRS", "MHMG-pf"]
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["rep"], r["fold"], r["method"]) for r in rows}
    jobs = [(n, r, f, mt) for mt in meths for n in DA.NAMES for r in (0, 1, 2) for f in range(1, 6) if (n, r, f, mt) not in done]
    with ProcessPoolExecutor(6) as ex:
        futs = [ex.submit(job, *j) for j in jobs]; print("jobs", len(futs), flush=True)
        for fu in as_completed(futs):
            rr = fu.result(); rows.append(rr); pd.DataFrame(rows).to_csv(OUT, index=False)
            print(time.strftime("%H:%M:%S"), rr["method"], rr["dataset"], rr["rep"], rr["fold"], rr["status"], round(rr["time_s"], 1), flush=True)
