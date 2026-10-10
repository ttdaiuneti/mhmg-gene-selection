"""E3 : label-noise robustness, rho in {0.1, 0.2}, training labels only; resumable."""
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
OUT = HERE / "raw_noise.csv"


def flip(y, rho, seed):
    rng = np.random.default_rng(seed); y = y.copy(); cls = np.unique(y)
    idx = rng.choice(len(y), size=int(round(rho * len(y))), replace=False)
    for i in idx:
        y[i] = rng.choice(cls[cls != y[i]])
    return y, idx


def mechanism(X, yn, flipped, B):
    """Among non-flipped training samples: fraction whose nearest enemy (under noisy labels, genes B) is a flipped sample."""
    Z = X[:, B]; sq = (Z * Z).sum(1); D = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    en = yn[:, None] != yn[None, :]; Dm = np.where(en, D, np.inf); ne = Dm.argmin(1)
    isf = np.zeros(len(yn), bool); isf[flipped] = True
    clean = ~isf
    frac_ne_flipped = float(isf[ne[clean]].mean())
    share_flipped_in_enemies = float((en & isf[None, :])[clean].sum() / en[clean].sum())
    return frac_ne_flipped, share_flipped_in_enemies


def run(name, rep, fold, tr, te, rho):
    warnings.filterwarnings("ignore")
    X, y = DA.load(name)
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    yn, fl = flip(y[tr], rho, 1000 * rep + 10 * fold + int(100 * rho))
    m = X.shape[1]; sel = {}
    t0 = time.perf_counter(); sel["MHMG"] = (M.mhmg(Xtr, yn), time.perf_counter() - t0)
    k = max(1, len(sel["MHMG"][0]))
    for meth, fn in (("F", M.top_f), ("FCQ", M.fcq), ("ReliefF", M.relieff), ("L1", M.l1)):
        t0 = time.perf_counter(); sel[meth] = (fn(Xtr, yn, k), time.perf_counter() - t0)
    sel["ALL"] = (list(range(m)), 0.0)
    mech = mechanism(Xtr, yn, fl, sel["MHMG"][0] or list(range(m)))
    rows = []
    for meth, (B, t) in sel.items():
        Be = list(B) or list(range(m))
        r = dict(dataset=name, rep=rep, fold=fold, rho=rho, method=meth, size=len(B), time_s=t, n_attr=m,
                 n_flipped=len(fl), reduct="|".join(map(str, B)) if meth != "ALL" else "",
                 ne_flipped=mech[0] if meth == "MHMG" else np.nan, share_flipped=mech[1] if meth == "MHMG" else np.nan)
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
