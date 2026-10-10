"""E2 : accuracy vs number of selected genes, k = 1..20."""
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
sys.path.insert(0, str(HERE.parent / "e1"))
import data as DA  # noqa: E402
import methods as M  # noqa: E402
KMAX, REPS = 20, (0, 1, 2)
OUT = HERE / "raw_curves.csv"


def run(name, rep, fold, tr, te):
    warnings.filterwarnings("ignore")
    X, y = DA.load(name)
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    ytr = y[tr]
    stop, path = M.P.forward(Xtr, ytr, "orig", min_path=KMAX)
    orders = {"MHMG": path[:KMAX], "F": M.top_f(Xtr, ytr, KMAX), "FCQ": M.fcq(Xtr, ytr, KMAX),
              "ReliefF": M.relieff(Xtr, ytr, KMAX), "L1": M.l1(Xtr, ytr, KMAX)}
    rows = []
    for meth, order in orders.items():
        for k in range(1, len(order) + 1):
            B = list(order[:k]); r = dict(dataset=name, rep=rep, fold=fold, method=meth, k=k, mhmg_stop=len(stop))
            for nm, clf in (("knn", KNeighborsClassifier(n_neighbors=3)), ("svm", SVC(kernel="rbf"))):
                r[f"{nm}_acc"] = float(np.mean(clf.fit(Xtr[:, B], ytr).predict(Xte[:, B]) == y[te]))
            rows.append(r)
    return rows


def main():
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["rep"], r["fold"]) for r in rows}
    jobs = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 5) as ex:
        for name in sorted(DA.NAMES, key=lambda n: -np.prod(DA.load(n)[0].shape)):
            X, y = DA.load(name)
            for rep in REPS:
                for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y), 1):
                    if (name, rep, fold) not in done:
                        jobs.append(ex.submit(run, name, rep, fold, tr, te))
        print("jobs", len(jobs), flush=True)
        for j in as_completed(jobs):
            rows += j.result()
            print(time.strftime("%H:%M:%S"), rows[-1]["dataset"], rows[-1]["rep"], rows[-1]["fold"], flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
