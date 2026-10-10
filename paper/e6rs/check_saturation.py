"""Saturation check (the study protocol, descriptive): rep 0 of E1. For NRS at eps in {0.1,0.2,0.3}, on the training fold and the
NRS reduct stored in e1/raw_results.csv: is every safety radius > eps at the stopping point (gamma_N = 1)? For the MHMG
reduct of the same fold: share of samples with radius > eps, and its size. Writes saturation.csv and sat_numbers.tex."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; G = HERE.parent; sys.path.insert(0, str(G / "e1"))
import data as DA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
E1 = pd.read_csv(G / "e1" / "raw_results.csv"); rows = []
for name in DA.NAMES:
    X, y = DA.load(name)
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y), 1):
        Xtr = MinMaxScaler().fit_transform(X[tr]); ytr = y[tr]; en = ytr[:, None] != ytr[None, :]
        def radii(R):
            Z = Xtr[:, R]; s = (Z * Z).sum(1); D = np.sqrt(np.maximum(s[:, None] + s[None, :] - 2 * Z @ Z.T, 0)); return np.where(en, D, np.inf).min(1)
        g = E1[(E1.dataset == name) & (E1.rep == 0) & (E1.fold == fold)].set_index("method")
        Rm = [int(a) for a in str(g.loc["MHMG", "reduct"]).split("|")]; rm = radii(Rm)
        for e in (0.1, 0.2, 0.3):
            Rn = [int(a) for a in str(g.loc[f"NRS_{e}", "reduct"]).split("|")]; rn = radii(Rn)
            rows.append(dict(dataset=name, fold=fold, eps=e, nrs_size=len(Rn), nrs_saturated=bool(rn.min() > e), nrs_share_above=float((rn > e).mean()),
                             mhmg_size=len(Rm), mhmg_share_above=float((rm > e).mean())))
    print(name, flush=True)
d = pd.DataFrame(rows); d.to_csv(HERE / "saturation.csv", index=False)
M = {}
for e, w in ((0.1, "A"), (0.2, "B"), (0.3, "C")):
    x = d[d.eps == e]; M[f"satShare{w}"] = f"{100 * x.nrs_saturated.mean():.0f}"; M[f"satSize{w}"] = f"{x.nrs_size.median():.0f}"
    M[f"satMhmgAbove{w}"] = f"{100 * x.mhmg_share_above.median():.0f}"
M["satFolds"] = str(d[d.eps == 0.1].shape[0])
(HERE / "sat_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(M)
