"""Check that the reference implementations in this folder reproduce the reducts recorded in the fold-level results.

    python3 src/reproduce_reducts.py                         # three small datasets, repetitions 0 and 1
    python3 src/reproduce_reducts.py GLIOMA leukemia --reps 0 1 2

For every (dataset, repetition, fold) the training part is min-max scaled, MHMG (src/mhmg.py), NRS with
radius 0.1 and 0.3, and FNRS (src/nrs_fnrs.py) are run, and the selected attributes are compared with
paper/e1/raw_results.csv. The 17 datasets are loaded by paper/e1/data.py (OpenML download, cached in
paper/feas/data, and the scikit-feature .mat files in paper/data_asu; see data/README.md).
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "paper" / "e1")]
import mhmg, nrs_fnrs, data as DA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="*", default=["colon-cancer", "duke-breast-cancer", "GLIOMA"])
ap.add_argument("--reps", nargs="*", type=int, default=[0, 1])
a = ap.parse_args()
raw = pd.read_csv(ROOT / "paper" / "e1" / "raw_results.csv")
METHODS = {"MHMG": lambda X, y: mhmg.select_features(X, y),
           "NRS_0.1": lambda X, y: nrs_fnrs.nrs(X, y, 0.1), "NRS_0.3": lambda X, y: nrs_fnrs.nrs(X, y, 0.3),
           "FNRS": lambda X, y: nrs_fnrs.fnrs(X, y)}
tally = {m: [0, 0] for m in METHODS}
for ds in a.datasets:
    X, y = DA.load(ds)
    for rep in a.reps:
        for fold, (tr, _) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y), 1):
            Xs = MinMaxScaler().fit(X[tr]).transform(X[tr])
            for m, fn in METHODS.items():
                out = fn(Xs, y[tr]); got = sorted(int(v) for v in (out[0] if isinstance(out, tuple) else out))
                rec = raw[(raw.dataset == ds) & (raw.rep == rep) & (raw.fold == fold) & (raw.method == m)].reduct.iloc[0]
                want = sorted(int(v) for v in str(rec).split("|"))
                tally[m][1] += 1; tally[m][0] += got == want
for m, (ok, n) in tally.items():
    print(f"{m:8s} identical reducts: {ok}/{n}")
sys.exit(0 if all(ok == n for ok, n in tally.values()) else 1)
