"""E1 : 17 microarray datasets x 3 repeats x 5 folds; resumable, one row per (dataset, rep, fold, method)."""
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
sys.path.insert(0, str(HERE))
import data as DA  # noqa: E402
import methods as M  # noqa: E402
REPS = (0, 1, 2)
OUT = HERE / "raw_results.csv"


def run(name, rep, fold, tr, te):
    warnings.filterwarnings("ignore")
    X, y = DA.load(name)
    sc = MinMaxScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    ytr, m = y[tr], X.shape[1]
    sel = {}

    def timed(meth, fn):
        t0 = time.perf_counter(); B = fn(); sel[meth] = (list(map(int, B)), time.perf_counter() - t0)

    timed("MHMG", lambda: M.mhmg(Xtr, ytr))
    k = max(1, len(sel["MHMG"][0]))
    for meth, fn in (("F", M.top_f), ("FCQ", M.fcq), ("ReliefF", M.relieff), ("L1", M.l1), ("RFE", M.rfe)):
        timed(meth, lambda fn=fn: fn(Xtr, ytr, k))
    for d in (0.1, 0.2, 0.3):
        timed(f"NRS_{d}", lambda d=d: M.nrs(Xtr, ytr, d))
    timed("FNRS", lambda: M.fnrs(Xtr, ytr))
    sel["ALL"] = (list(range(m)), 0.0)
    rows = []
    for meth, (B, t) in sel.items():
        Be = B or list(range(m))
        r = dict(dataset=name, rep=rep, fold=fold, method=meth, size=len(B), time_s=t, n=len(y), n_attr=m,
                 k_mhmg=k, reduct="|".join(map(str, B)) if meth != "ALL" else "")
        for nm, clf in (("knn", KNeighborsClassifier(n_neighbors=3)), ("svm", SVC(kernel="rbf"))):
            r[f"{nm}_acc"] = float(np.mean(clf.fit(Xtr[:, Be], ytr).predict(Xte[:, Be]) == y[te]))
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
            pd.DataFrame(rows).to_csv(OUT, index=False)        # append-only checkpoint (finished folds never rerun)


if __name__ == "__main__":
    main()
