"""E5 : competitors tuned within q <= 10 (inner 3-fold CV on the outer training part), outer folds of E1.
Resumable, append-only; one row per (dataset, rep, fold, method). MHMG and ALL are taken from e1/raw_results.csv.
Usage: python run_e4.py [workers] | python run_e4.py --smoke (one fold, timing only, accuracies not printed)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e1"))
import data as DA  # noqa: E402
import methods as M  # noqa: E402
REPS = (0, 1, 2)
Q = (2, 3, 4, 5, 6, 7, 8, 9, 10)
OUT = HERE / "raw_e5.csv"


def knn_acc(Xa, ya, Xb, yb, B):
    if len(B) == 0:
        return 0.0
    return float(np.mean(KNeighborsClassifier(n_neighbors=3).fit(Xa[:, B], ya).predict(Xb[:, B]) == yb))


# ---- one function per selector family: returns {config: subset} on (X, y) ----
def rank_configs(rank_fn):
    def f(X, y):
        r = list(rank_fn(X, y))
        return {q: r[:q] for q in Q}
    return f


def rfe_rank(X, y):
    rk = RFE(LinearSVC(max_iter=5000, random_state=0), n_features_to_select=2, step=0.1).fit(X, y).ranking_
    return np.lexsort((np.arange(len(rk)), rk))


def l1_rank(X, y):
    c = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000, random_state=0).fit(X, y)
    return np.argsort(-np.abs(c.coef_).max(0), kind="stable")


FAMILIES = {
    "F-cv10": (rank_configs(lambda X, y: M.top_f(X, y, max(Q))), lambda X, y, q: M.top_f(X, y, q)),
    "FCQ-cv10": (rank_configs(lambda X, y: M.fcq(X, y, max(Q))), lambda X, y, q: M.fcq(X, y, q)),
    "ReliefF-cv10": (rank_configs(lambda X, y: M.relieff(X, y, max(Q))), lambda X, y, q: M.relieff(X, y, q)),
    "L1-cv10": (rank_configs(l1_rank), lambda X, y, q: list(l1_rank(X, y)[:q])),
    "RFE-cv10": (rank_configs(rfe_rank),
                 lambda X, y, q: list(np.flatnonzero(RFE(LinearSVC(max_iter=5000, random_state=0), n_features_to_select=q,
                                                         step=0.1).fit(X, y).support_))),
}


def tune(meth, Xtr, ytr, inner):
    configs_fn, final_fn = FAMILIES[meth]
    t0 = time.perf_counter()
    score, size = {}, {}
    for itr, iva in inner:
        for cfg, B in configs_fn(Xtr[itr], ytr[itr]).items():
            score.setdefault(cfg, []).append(knn_acc(Xtr[itr], ytr[itr], Xtr[iva], ytr[iva], list(B)))
            size.setdefault(cfg, []).append(len(B))
    grid = list(score)
    best = max(grid, key=lambda c: (np.mean(score[c]), -np.mean(size[c]), -grid.index(c)))
    B, cfg = list(final_fn(Xtr, ytr, best)), best
    return B, cfg, time.perf_counter() - t0


def run(name, rep, fold, tr, te, methods=tuple(FAMILIES)):
    warnings.filterwarnings("ignore")
    X, y = DA.load(name)
    sc = MinMaxScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    ytr, yte, m = y[tr], y[te], X.shape[1]
    inner = list(StratifiedKFold(3, shuffle=True, random_state=100 * rep + fold).split(Xtr, ytr))
    rows = []
    for meth in methods:
        B, cfg, t = tune(meth, Xtr, ytr, inner)
        Be = [int(b) for b in B] or list(range(m))
        r = dict(dataset=name, rep=rep, fold=fold, method=meth, config=cfg, size=len(B), time_s=t, n=len(y), n_attr=m,
                 reduct="|".join(map(str, Be)))
        for nm, clf in (("knn", KNeighborsClassifier(n_neighbors=3)), ("svm", SVC(kernel="rbf"))):
            r[f"{nm}_acc"] = float(np.mean(clf.fit(Xtr[:, Be], ytr).predict(Xte[:, Be]) == yte))
        rows.append(r)
    return rows


def main():
    if "--smoke" in sys.argv:                            # timing only; accuracies deliberately not printed
        name = max(DA.NAMES, key=lambda n: np.prod(DA.load(n)[0].shape))
        X, y = DA.load(name)
        tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
        for r in run(name, 0, 1, tr, te):
            print(name, r["method"], f"time {r['time_s']:.1f}s", flush=True)
        return
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["rep"], r["fold"]) for r in rows}
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 5) as ex:
        jobs = []
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
