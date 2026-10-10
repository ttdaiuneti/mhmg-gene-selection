import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; G = HERE.parent; X = G / "external"
sys.path[:0] = [str(G / "e1"), str(X)]
OUT = HERE / "raw_ers_gbfix.csv"; CAP = 1800.0


def job(name, rep, fold):
    warnings.filterwarnings("ignore")
    import data as DA, gbnrs as GB
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    X, y = DA.load(name); tr, te = list(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y))[fold - 1]
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2); ytr, yte = y[tr], y[te]
    pool = np.sort(np.argsort(Xtr.var(0))[::-1][:80]); Xp = Xtr[:, pool]; C = list(range(80))
    t0 = time.perf_counter(); status = "ok"
    gamma_C = GB.compute_gb_dependency(Xp, ytr, C, 0.9)[0]; R = []
    while True:                                   # reference loop of gbnrs_reduction (no stall stop)
        if R and abs(GB.compute_gb_dependency(Xp, ytr, R, 0.9)[0] - gamma_C) < 1e-10:
            break
        best, bg = None, -1.0
        for a in C:
            if a in R: continue
            if time.perf_counter() - t0 > CAP: status = "timeout"; break
            g = GB.compute_gb_dependency(Xp, ytr, R + [a], 0.9)[0]
            if g > bg: bg, best = g, a
        if status == "timeout" or best is None: break
        R.append(best)
    t = time.perf_counter() - t0; Rg = [int(pool[a]) for a in R]; Be = Rg or list(range(X.shape[1]))
    k = float(np.mean(KNeighborsClassifier(3).fit(Xtr[:, Be], ytr).predict(Xte[:, Be]) == yte))
    s = float(np.mean(SVC().fit(Xtr[:, Be], ytr).predict(Xte[:, Be]) == yte))
    return dict(dataset=name, rep=rep, fold=fold, method="GBNRS", status=status, size=len(Rg), time_s=t, knn_acc=k, svm_acc=s, reduct="|".join(map(str, Rg)))


if __name__ == "__main__":
    import data as DA
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["rep"], r["fold"]) for r in rows}
    jobs = [(n, r, f) for n in DA.NAMES for r in (0, 1, 2) for f in range(1, 6) if (n, r, f) not in done]
    with ProcessPoolExecutor(6) as ex:
        futs = [ex.submit(job, *j) for j in jobs]; print("jobs", len(futs), flush=True)
        for fu in as_completed(futs):
            rr = fu.result(); rows.append(rr); pd.DataFrame(rows).to_csv(OUT, index=False)
            print(time.strftime("%H:%M:%S"), rr["dataset"], rr["rep"], rr["fold"], rr["status"], round(rr["time_s"], 1), rr["size"], flush=True)
