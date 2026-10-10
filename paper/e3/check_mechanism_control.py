"""Control for the label-noise mechanism: fraction of correctly labelled training samples whose nearest
enemy is a flipped sample, under (a) the MHMG genes selected on the noisy fold, (b) the F genes of the same size
selected on the noisy fold, (c) random genes of the same size. Same flips as run_e3.py. Exploratory."""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0, "../e1")
import data as DA  # noqa: E402
from run_e3 import flip, mechanism  # noqa: E402

warnings.filterwarnings("ignore")
raw = pd.read_csv("raw_noise.csv")
rows = []
for name in DA.NAMES:
    X, y = DA.load(name)
    for rep in (0, 1, 2):
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y), 1):
            Xtr = MinMaxScaler().fit(X[tr]).transform(X[tr])
            for rho in (0.1, 0.2):
                yn, fl = flip(y[tr], rho, 1000 * rep + 10 * fold + int(100 * rho))
                g = raw[(raw.dataset == name) & (raw.rep == rep) & (raw.fold == fold) & (raw.rho == rho)].set_index("method")
                Bm = [int(a) for a in str(g.loc["MHMG", "reduct"]).split("|")]
                Bf = [int(a) for a in str(g.loc["F", "reduct"]).split("|")]
                Br = list(np.random.default_rng(1000 * rep + 10 * fold + int(100 * rho)).choice(X.shape[1], len(Bm), replace=False))
                r = dict(dataset=name, rep=rep, fold=fold, rho=rho)
                for lab, B in (("MHMG", Bm), ("F", Bf), ("random", Br)):
                    r[f"ne_{lab}"], r["share"] = mechanism(Xtr, yn, fl, B)
                rows.append(r)
d = pd.DataFrame(rows); d.to_csv("check_mechanism_control.csv", index=False)
med = d.groupby(["dataset", "rho"]).mean(numeric_only=True).groupby(level="rho").median()
print(med[["ne_MHMG", "ne_F", "ne_random", "share"]].round(3).to_string())
