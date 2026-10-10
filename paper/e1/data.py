"""The 17 datasets (cached as npz)."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "feas" / "data"
OPENML = {"lymphoma_2classes": 1101, "leukemia": 1104, "BurkittLymphoma": 1084, "GCM": 1106, "tumors_C": 1107,
          "rsctc2010_1": 1077, "rsctc2010_3": 1079, "AP_Omentum_Uterus": 1124,
          "colon-cancer": 1432, "duke-breast-cancer": 1434}
ASU = ["CLL_SUB_111", "GLIOMA", "GLI_85", "Prostate_GE", "SMK_CAN_187", "TOX_171", "lung"]
NAMES = list(OPENML) + ASU
NEW = ["colon-cancer", "duke-breast-cancer"] + ASU


def load(name):
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f); return z["X"], z["y"]
    if name in ASU:
        import scipy.io as sio
        d = sio.loadmat(ROOT / "data_asu" / f"{name}.mat")
        X = np.asarray(d["X"], dtype=float); y = pd.factorize(d["Y"].ravel())[0]
    else:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=OPENML[name], as_frame=False, parser="auto")
        X = d.data.toarray() if hasattr(d.data, "toarray") else np.asarray(d.data, dtype=float)
        y = pd.factorize(pd.Series(d.target).astype(str))[0]
    X = np.nan_to_num(X.astype(float))
    X = X[:, np.ptp(X, axis=0) > 0]
    CACHE.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, X=X, y=y)
    return X, y
