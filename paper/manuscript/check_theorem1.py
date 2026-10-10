"""Numerical check of Theorem 1 (granule size >= k+1 => leave-one-out k-NN correct; energy lower bound) on the MHMG
reducts of E1 (training folds, rep 0), k = 1 and 3. Exact distances; self excluded by index; any tie-breaking."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
E1 = Path(__file__).resolve().parent.parent / "e1"
sys.path.insert(0, str(E1))
import data as DA  # noqa: E402
res = pd.read_csv(E1 / "raw_results.csv")
rows = []
for name in DA.NAMES:
    X, y = DA.load(name)
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y), 1):
        r = res[(res.dataset == name) & (res.rep == 0) & (res.fold == fold) & (res.method == "MHMG")].iloc[0]
        B = [int(v) for v in str(r.reduct).split("|")]
        Z = MinMaxScaler().fit(X[tr]).transform(X[tr])[:, B]; yt = y[tr]; n = len(yt)
        D = np.zeros((n, n))
        for j in range(Z.shape[1]):
            dd = Z[:, j][:, None] - Z[:, j][None, :]; D += dd * dd
        enemy = yt[:, None] != yt[None, :]
        delta = np.where(enemy, D, np.inf).min(1)
        G = (D < delta[:, None]).sum(1)
        same_other = (~enemy) & ~np.eye(n, dtype=bool)
        radius_ties = int((same_other & (D == delta[:, None])).sum())
        cls, cnt = np.unique(yt, return_counts=True); nmax = cnt.max()
        E = np.log2(G + 1).sum()
        for k in (1, 3):
            Dx = D.copy(); np.fill_diagonal(Dx, np.inf)
            order = np.argsort(Dx, axis=1, kind="stable")[:, :k]
            worst_ok = True; correct = 0
            for i in range(n):
                votes = yt[order[i]]
                vals, c = np.unique(votes, return_counts=True)
                correct += int(vals[np.argmax(c)] == yt[i] and c.max() > k / 2)
                if G[i] >= k + 1 and not (votes == yt[i]).all():
                    worst_ok = False
            A = (G >= k + 1).mean()
            L, Mx = np.log2(k + 1), np.log2(nmax + 1)
            bound = (E / n - L) / (Mx - L)
            rows.append(dict(dataset=name, fold=fold, k=k, part_i=worst_ok, loo_acc=correct / n, A=A, bound=bound,
                             ii=correct / n >= A - 1e-12, iii=A >= bound - 1e-12, delta0=int((delta == 0).sum()), radius_ties=radius_ties))
d = pd.DataFrame(rows); d.to_csv(Path(__file__).with_name("check_theorem1.csv"), index=False)
print("rows", len(d), "| (i) holds:", d.part_i.all(), "| (ii) holds:", d.ii.all(), "| (iii) holds:", d.iii.all(),
      "| folds with delta=0:", int((d.delta0 > 0).sum()), "| folds with ties at the radius:", int((d.radius_ties > 0).sum()))
print(d.groupby("k")[["loo_acc", "A", "bound"]].median().round(3))
