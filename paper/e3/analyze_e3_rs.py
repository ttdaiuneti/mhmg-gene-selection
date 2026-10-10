"""E3 amendment 1 analysis (written before its results): MHMG vs NRS (each radius, NRS* with the clean-data radius)
and FNRS under label noise. rho = 0 rows from e1/raw_results.csv; MHMG under noise from raw_noise.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = Path(__file__).resolve().parent
RS = ["NRS_0.1", "NRS_0.2", "NRS_0.3", "FNRS"]
rs = pd.read_csv(HERE / "raw_noise_rs.csv"); assert rs.groupby(["dataset", "rho"]).size().eq(15 * len(RS)).all()
mh = pd.read_csv(HERE / "raw_noise.csv"); mh = mh[mh.method == "MHMG"]
e1 = pd.read_csv(HERE.parent / "e1" / "raw_results.csv"); e1 = e1[e1.method.isin(RS + ["MHMG"])].assign(rho=0.0)
d = pd.concat([e1, mh, rs], ignore_index=True)
acc = d.groupby(["dataset", "rho", "method"]).knn_acc.mean().unstack()
size = d.groupby(["dataset", "rho", "method"])["size"].mean().unstack()
clean = acc.xs(0.0, level="rho")
star = clean[["NRS_0.1", "NRS_0.2", "NRS_0.3"]].idxmax(axis=1)                 # radius chosen on clean data (E1)
for t in (acc, size):
    t["NRS*"] = [t.loc[(ds, r), star[ds]] for ds, r in t.index]
COLS = ["MHMG", "NRS_0.1", "NRS_0.2", "NRS_0.3", "NRS*", "FNRS"]
out = []
P = lambda s: out.append(s) or print(s)
for rho in (0.0, 0.1, 0.2):
    a, s = acc.xs(rho, level="rho"), size.xs(rho, level="rho")
    P(f"rho={rho}: 3-NN " + ", ".join(f"{c} {a[c].mean():.3f}" for c in COLS) + " | median size " +
      ", ".join(f"{c} {s[c].median():.1f}" for c in COLS))
drop = acc.xs(0.2, level="rho") - acc.xs(0.0, level="rho")
P("mean drop 0->0.2: " + ", ".join(f"{c} {drop[c].mean():+.3f}" for c in COLS))
for c in COLS[1:]:
    p = wilcoxon(drop.MHMG, drop[c], zero_method="zsplit").pvalue
    a2 = acc.xs(0.2, level="rho")
    P(f"vs {c}: median (drop_MHMG - drop_{c}) {np.median(drop.MHMG - drop[c]):+.4f} (two-sided p {p:.4f}); "
      f"at rho=0.2 MHMG more accurate on {(a2.MHMG > a2[c]).sum()}/17")
acc.round(4).to_csv(HERE / "acc_by_rho_rs.csv"); (HERE / "analysis_e3_rs.txt").write_text("\n".join(out) + "\n")
