"""E3 amendment 4 : FNRS (E1 settings, cap 50) under label noise; resumable."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e1")); sys.path.insert(0, str(HERE))
import data as DA  # noqa: E402
import methods as M  # noqa: E402
from run_e3 import flip  # noqa: E402
OUT = HERE / "raw_noise_fnrs.csv"
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
e1 = pd.read_csv(HERE.parent / "e1" / "raw_results.csv")
acc = e1[e1.method.str.startswith("NRS_")].groupby(["dataset", "method"]).knn_acc.mean().unstack()
DSTAR = {ds: float(c.split("_")[1]) for ds, c in acc.idxmax(axis=1).items()}


def run(name, rep, fold, tr, te, rho):
    warnings.filterwarnings("ignore")
    X, y = DA.load(name)
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    yn, fl = flip(y[tr], rho, 1000 * rep + 10 * fold + int(100 * rho))
    t0 = time.perf_counter(); B = M.fnrs(Xtr, yn); t = time.perf_counter() - t0
    Be = list(B) or list(range(X.shape[1]))
    r = dict(dataset=name, rep=rep, fold=fold, rho=rho, method="FNRS", size=len(B), time_s=t,
             n_attr=X.shape[1], n_flipped=len(fl), reduct="|".join(map(str, B)))
    for nm, clf in (("knn", KNeighborsClassifier(n_neighbors=3)), ("svm", SVC(kernel="rbf"))):
        r[f"{nm}_acc"] = float(np.mean(clf.fit(Xtr[:, Be], yn).predict(Xte[:, Be]) == y[te]))
    return r


def main():
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["rep"], r["fold"], r["rho"]) for r in rows}
    jobs = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 5) as ex:
        for name in sorted(DA.NAMES, key=lambda n: -np.prod(DA.load(n)[0].shape)):
            X, y = DA.load(name)
            for rep in (0, 1, 2):
                for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y), 1):
                    if (name, rep, fold, RHO) not in done:
                        jobs.append(ex.submit(run, name, rep, fold, tr, te, RHO))
        print("jobs", len(jobs), "delta*", DSTAR, flush=True)
        for j in as_completed(jobs):
            rows.append(j.result())
            print(time.strftime("%H:%M:%S"), rows[-1]["dataset"], rows[-1]["rep"], rows[-1]["fold"], flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
