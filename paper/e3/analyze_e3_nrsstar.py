"""E3 amendment 2 analysis (written before its results): MHMG vs NRS* (clean-data radius) at rho = 0 and 0.2."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = Path(__file__).resolve().parent
nsall = pd.read_csv(HERE / "raw_noise_nrsstar.csv")
assert nsall.groupby(["dataset", "rho"]).size().eq(15).all() and nsall.dataset.nunique() == 17
ns = nsall[nsall.rho == 0.2]
e1 = pd.read_csv(HERE.parent / "e1" / "raw_results.csv")
dstar = ns.groupby("dataset").delta.first()
clean_n = pd.Series({ds: e1[(e1.dataset == ds) & (e1.method == f"NRS_{dstar[ds]}")].knn_acc.mean() for ds in dstar.index})
clean_ns = pd.Series({ds: e1[(e1.dataset == ds) & (e1.method == f"NRS_{dstar[ds]}")]["size"].mean() for ds in dstar.index})
mh = pd.read_csv(HERE / "raw_noise.csv"); mh = mh[(mh.method == "MHMG") & (mh.rho == 0.2)]
clean_m = e1[e1.method == "MHMG"].groupby("dataset").knn_acc.mean()
noisy_m = mh.groupby("dataset").knn_acc.mean(); noisy_n = ns.groupby("dataset").knn_acc.mean()
dm, dn = noisy_m - clean_m, noisy_n - clean_n
out = [f"3-NN rho=0: MHMG {clean_m.mean():.3f}, NRS* {clean_n.mean():.3f}",
       f"3-NN rho=0.2: MHMG {noisy_m.mean():.3f}, NRS* {noisy_n.mean():.3f}",
       f"mean drop: MHMG {dm.mean():+.3f}, NRS* {dn.mean():+.3f}; median (drop_MHMG - drop_NRS*) {np.median(dm - dn):+.4f}, "
       f"two-sided Wilcoxon p {wilcoxon(dm, dn, zero_method='zsplit').pvalue:.4f}; MHMG drops more on {(dm < dn).sum()}/17",
       f"at rho=0.2 MHMG more accurate on {(noisy_m > noisy_n).sum()}/17",
       f"median size NRS*: rho=0 {clean_ns.median():.1f}, rho=0.2 {ns.groupby('dataset')['size'].mean().median():.1f}; "
       f"MHMG rho=0.2 {mh.groupby('dataset')['size'].mean().median():.1f}"]
if (nsall.rho == 0.1).any():
    n1 = nsall[nsall.rho == 0.1].groupby("dataset").knn_acc.mean()
    m1 = pd.read_csv(HERE / "raw_noise.csv"); m1 = m1[(m1.method == "MHMG") & (m1.rho == 0.1)].groupby("dataset").knn_acc.mean()
    d1m, d1n = m1 - clean_m, n1 - clean_n
    out += [f"3-NN rho=0.1: MHMG {m1.mean():.3f}, NRS* {n1.mean():.3f}; mean drop MHMG {d1m.mean():+.3f}, NRS* {d1n.mean():+.3f}; "
            f"two-sided Wilcoxon p {wilcoxon(d1m, d1n, zero_method='zsplit').pvalue:.4f}; MHMG more accurate on {(m1 > n1).sum()}/17"]
print("\n".join(out)); (HERE / "analysis_e3_nrsstar.txt").write_text("\n".join(out) + "\n")
