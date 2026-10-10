"""Exploratory check for Proposition prop:saturation: on the 85 training folds of rep 0, the number of attributes at which
RANDOM attribute subsets (20 random orders per fold) saturate NRS (every safety radius > eps), eps in {0.1, 0.3}.
Writes randsat.csv and randsat_numbers.tex."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
G = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(G / "e1")); import data as DA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); rng = np.random.default_rng(0); rows = []
for name in DA.NAMES:
    X, y = DA.load(name)
    for fold, (tr, _) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y), 1):
        Z = MinMaxScaler().fit_transform(X[tr]); yt = y[tr]; n, m = Z.shape; en = yt[:, None] != yt[None, :]
        for r in range(20):
            order = rng.permutation(m); acc = np.zeros((n, n)); q = 0; hit = {}
            while len(hit) < 2 and q < m:
                d = Z[:, order[q]][:, None] - Z[:, order[q]][None, :]; acc += d * d; q += 1
                mn = np.where(en, acc, np.inf).min()
                for e in (0.1, 0.3):
                    if e not in hit and mn > e * e: hit[e] = q
            rows.append(dict(dataset=name, m=m, n=n, fold=fold, r=r, q01=hit.get(0.1, m), q03=hit.get(0.3, m)))
    print(name, flush=True)
d = pd.DataFrame(rows); d.to_csv(Path(__file__).with_name("randsat.csv"), index=False)
g = d.groupby("dataset").agg(m=("m", "first"), q01=("q01", "median"), q03=("q03", "median"))
M = {"rsatMedA": f"{d.q01.median():.0f}", "rsatMedC": f"{d.q03.median():.0f}", "rsatMaxDsA": f"{g.q01.max():.0f}",
     "rsatMaxDsC": f"{g.q03.max():.0f}", "rsatRhoM": f"{spearmanr(g.m, g.q01).correlation:.2f}", "rsatRhoMp": f"{spearmanr(g.m, g.q01).pvalue:.2f}"}
Path(__file__).with_name("randsat_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(g.round(1).to_string()); print(M)
