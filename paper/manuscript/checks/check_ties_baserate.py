"""Round-2 review checks (exploratory, fixed before computing):
(1) tie prevalence per dataset: share of attributes with at least one pair of samples of different classes with equal
    value; median (over attributes) share of samples with an empty single-attribute granule (delta = 0); min-max on all data;
(2) base rate of mean-blind attributes (same definition as the E8b analysis: ANOVA p > 0.05 and Bonferroni-corrected
    one-vs-rest KS p < 0.05) among ALL attributes of each dataset, against the share among the MHMG selections;
(3) on the tie-free datasets (< 5 % tied attributes): Spearman correlation over 300 random attributes between E({a}) and
    each runs bound of Proposition prop:runs.
Writes ties_numbers.tex, tab_ties.tex, ties.csv."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.feature_selection import f_classif
H = Path(__file__).resolve().parent; G = H.parents[1]; G2 = G.parent / "MHMG-Gene2"
sys.path.insert(0, str(G / "e1")); import data as DA
warnings.filterwarnings("ignore")
# optional input from a separate study (macros baseAbove/baseBelow); not used by the paper and not part of this repository
sf = G2 / "e8" / "e8b_share.csv"
share = pd.read_csv(sf).pivot(index="dataset", columns="method", values="mb_share") if sf.exists() else None


def energy_and_bounds(x, y):
    o = np.argsort(x, kind="stable"); xs, ys = x[o], y[o]; n = len(y)
    D = np.abs(xs[:, None] - xs[None, :]); en = ys[:, None] != ys[None, :]; r = np.where(en, D, np.inf).min(1)
    E = np.log2((D < r[:, None]).sum(1) + 1).sum()
    run = np.concatenate([[0], np.cumsum(ys[1:] != ys[:-1])]); lens = np.bincount(run)
    up = float((lens * np.log2(lens + 1)).sum()); lo = 0.0
    first = np.searchsorted(run, run, "left"); last = np.searchsorted(run, run, "right") - 1
    for i in range(n):
        sides = ([i - first[i]] if first[i] > 0 else []) + ([last[i] - i] if last[i] < n - 1 else [])
        lo += np.log2(2 + min(sides))
    return E, lo, up


rows = []; rng = np.random.default_rng(0)
for name in DA.NAMES:
    X, y = DA.load(name); X = (X - X.min(0)) / np.where(np.ptp(X, 0) > 0, np.ptp(X, 0), 1); n, m = X.shape
    tied = np.zeros(m, bool); empty = np.zeros(m)
    for j in range(m):
        v = X[:, j]; D = v[:, None] == v[None, :]; en = y[:, None] != y[None, :]; z = (D & en).any(1)
        tied[j] = z.any(); empty[j] = z.mean()
    pF = np.nan_to_num(f_classif(X, y)[1], nan=1.0); cls = np.unique(y)
    pks = np.array([min(1.0, len(cls) * min(ks_2samp(X[y == c, j], X[y != c, j]).pvalue for c in cls)) for j in range(m)])
    base = float(((pF > 0.05) & (pks < 0.05)).mean())
    r = dict(dataset=name, n=n, m=m, tied=tied.mean(), empty_med=float(np.median(empty)), base=base, mhmg=(share.loc[name, "MHMG"] if share is not None else float("nan")))
    if tied.mean() < 0.05:
        J = rng.choice(np.flatnonzero(~tied), min(300, int((~tied).sum())), replace=False)
        V = np.array([energy_and_bounds(X[:, j], y) for j in J])
        r["rho_lo"] = spearmanr(V[:, 0], V[:, 1]).correlation; r["rho_up"] = spearmanr(V[:, 0], V[:, 2]).correlation
    rows.append(r); print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "dataset"}, flush=True)
d = pd.DataFrame(rows); d.to_csv(H / "ties.csv", index=False)
free = d[d.tied < 0.05]
M = {"tieHeavy": str(int((d.tied > 2 / 3).sum())), "tieFree": str(len(free)), "tieFreeNames": ", ".join(free.dataset).replace("_", "-"),
     "tieEmptyMax": f"{100 * d.empty_med.max():.0f}", "tieEmptyMaxDs": d.loc[d.empty_med.idxmax(), "dataset"].replace("_", "-"),
     "baseMed": f"{100 * d.base.median():.1f}",
     "rhoLoMin": f"{free.rho_lo.min():.2f}", "rhoLoMax": f"{free.rho_lo.max():.2f}", "rhoUpMin": f"{free.rho_up.min():.2f}", "rhoUpMax": f"{free.rho_up.max():.2f}"}
(H / "ties_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(M)
