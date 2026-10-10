"""Label noise: MHMG vs FNRS at rho = 0 (E1) and 0.2."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = Path(__file__).resolve().parent
fnall = pd.read_csv(HERE / "raw_noise_fnrs.csv")
assert fnall.groupby(["dataset", "rho"]).size().eq(15).all() and fnall.dataset.nunique() == 17
fn = fnall[fnall.rho == 0.2]
e1 = pd.read_csv(HERE.parent / "e1" / "raw_results.csv")
c_m = e1[e1.method == "MHMG"].groupby("dataset").knn_acc.mean(); c_f = e1[e1.method == "FNRS"].groupby("dataset").knn_acc.mean()
mh = pd.read_csv(HERE / "raw_noise.csv"); n_m = mh[(mh.method == "MHMG") & (mh.rho == 0.2)].groupby("dataset").knn_acc.mean()
n_f = fn.groupby("dataset").knn_acc.mean()
dm, df = n_m - c_m, n_f - c_f
out = [f"3-NN rho=0: MHMG {c_m.mean():.3f}, FNRS {c_f.mean():.3f}", f"3-NN rho=0.2: MHMG {n_m.mean():.3f}, FNRS {n_f.mean():.3f}",
       f"mean drop: MHMG {dm.mean():+.3f}, FNRS {df.mean():+.3f}; median (drop_MHMG - drop_FNRS) {np.median(dm - df):+.4f}; "
       f"two-sided Wilcoxon p {wilcoxon(dm, df, zero_method='zsplit').pvalue:.4f}; MHMG drops more on {(dm < df).sum()}/17",
       f"at rho=0.2 MHMG more accurate on {(n_m > n_f).sum()}/17; FNRS size min/max {fn['size'].min()}/{fn['size'].max()}"]
if (fnall.rho == 0.1).any():
    f1 = fnall[fnall.rho == 0.1].groupby("dataset").knn_acc.mean()
    m1 = mh[(mh.method == "MHMG") & (mh.rho == 0.1)].groupby("dataset").knn_acc.mean()
    d1m, d1f = m1 - c_m, f1 - c_f
    out += [f"3-NN rho=0.1: MHMG {m1.mean():.3f}, FNRS {f1.mean():.3f}; mean drop MHMG {d1m.mean():+.3f}, FNRS {d1f.mean():+.3f}; "
            f"two-sided Wilcoxon p {wilcoxon(d1m, d1f, zero_method='zsplit').pvalue:.4f}; MHMG more accurate on {(m1 > f1).sum()}/17"]
print("\n".join(out)); (HERE / "analysis_e3_fnrs.txt").write_text("\n".join(out) + "\n")
