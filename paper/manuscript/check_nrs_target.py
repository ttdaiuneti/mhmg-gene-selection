"""Why NRS subsets are small on microarray data: on the first training fold of each dataset, the delta-positive region
of the full gene set contains every sample for delta in {0.1, 0.2, 0.3}; reports the smallest between-class distance."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "e1"))
import data as DA  # noqa: E402
rows = []
for nm in DA.NAMES:
    X, y = DA.load(nm); tr, _ = next(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    Z = MinMaxScaler().fit_transform(X[tr]); yt = y[tr]
    sq = (Z * Z).sum(1); D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)); en = yt[:, None] != yt[None, :]
    r = dict(dataset=nm, min_between_class=float(np.where(en, D, np.inf).min()))
    for d in (0.1, 0.2, 0.3):
        r[f"gammaC_{d}"] = 1 - float(((D <= d) & en).any(1).mean())
    rows.append(r)
pd.DataFrame(rows).to_csv(Path(__file__).with_name("check_nrs_target.csv"), index=False); print(pd.DataFrame(rows).round(3).to_string())
