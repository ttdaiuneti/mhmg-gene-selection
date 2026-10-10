"""E3 amendment 1 : NRS (three radii) and FNRS under the same label noise as run_e3.py; resumable."""
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
from run_e3 import flip  # noqa: E402  (same noise as the main E3 run)
OUT = HERE / "raw_noise_rs.csv"


def run(name, rep, fold, tr, te, rho):
    warnings.filterwarnings("ignore")
    X, y = DA.load(name)
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    yn, fl = flip(y[tr], rho, 1000 * rep + 10 * fold + int(100 * rho))
    m = X.shape[1]; sel = {}
    for dl in (0.1, 0.2, 0.3):
        t0 = time.perf_counter(); sel[f"NRS_{dl}"] = (M.nrs(Xtr, yn, dl), time.perf_counter() - t0)
    t0 = time.perf_counter(); sel["FNRS"] = (M.fnrs(Xtr, yn), time.perf_counter() - t0)
    rows = []
    for meth, (B, t) in sel.items():
        Be = list(B) or list(range(m))
        r = dict(dataset=name, rep=rep, fold=fold, rho=rho, method=meth, size=len(B), time_s=t, n_attr=m,
                 n_flipped=len(fl), reduct="|".join(map(str, B)))
        for nm, clf in (("knn", KNeighborsClassifier(n_neighbors=3)), ("svm", SVC(kernel="rbf"))):
            r[f"{nm}_acc"] = float(np.mean(clf.fit(Xtr[:, Be], yn).predict(Xte[:, Be]) == y[te]))
        rows.append(r)
    return rows


def main():
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["rep"], r["fold"], r["rho"]) for r in rows}
    jobs = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 5) as ex:
        for name in sorted(DA.NAMES, key=lambda n: -np.prod(DA.load(n)[0].shape)):
            X, y = DA.load(name)
            for rep in (0, 1, 2):
                for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y), 1):
                    for rho in (0.1, 0.2):
                        if (name, rep, fold, rho) not in done:
                            jobs.append(ex.submit(run, name, rep, fold, tr, te, rho))
        print("jobs", len(jobs), flush=True)
        for j in as_completed(jobs):
            rows += j.result()
            print(time.strftime("%H:%M:%S"), rows[-1]["dataset"], rows[-1]["rep"], rows[-1]["fold"], rows[-1]["rho"], flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
