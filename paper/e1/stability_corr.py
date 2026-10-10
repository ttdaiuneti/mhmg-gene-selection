"""EXPLORATORY (decided after seeing E1 results, user request 2026-10-05): correlation-aware stability
Phi_C = 1 - tr(C S) / tr(C Sigma0)  (Sechidis, Papangelou, Nogueira, Weatherall & Brown, ECML-PKDD 2019),
C_fg = |rho_fg| if |rho_fg| >= theta else 0 (C_ff = 1), rho on the full table (label-free). C = I gives Nogueira 2018.
Also: mean within-set |rho| (redundancy) of each selected set. Reads raw_results.csv only."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DA  # noqa: E402
THETAS = (0.5, 0.7, 0.9)


def zmat(sets, m):
    Z = np.zeros((len(sets), m))
    for i, s in enumerate(sets):
        Z[i, s] = 1
    return Z


def phi_c(sets, m, Cu, U, offsum):
    """Cu: C restricted to U (union of selected features); offsum: sum of off-diagonal C over all m x m pairs."""
    Z = zmat(sets, m)[:, U]
    M = len(sets)
    kbar = Z.sum(1).mean()
    S = np.cov(Z, rowvar=False, ddof=1) if len(U) > 1 else np.array([[Z.var(ddof=1)]])
    a = kbar / m * (1 - kbar / m)
    b = kbar * (kbar - 1) / (m * (m - 1)) - (kbar / m) ** 2
    return 1 - float((Cu * S).sum()) / (a * m + b * offsum)


def nogueira(sets, m):
    Z = zmat(sets, m); kbar = Z.sum(1).mean()
    return 1 - Z.var(0, ddof=1).mean() / (kbar / m * (1 - kbar / m))


def main():
    d = pd.read_csv(HERE / "raw_results.csv")
    d = d[d.method != "ALL"]
    rows, red = [], []
    for name in DA.NAMES:
        X, _ = DA.load(name)
        Xs = (X - X.mean(0)) / X.std(0)
        n, m = Xs.shape
        g = d[d.dataset == name]
        sets_by = {meth: [list(map(int, r.split("|"))) if isinstance(r, str) and r else [] for r in gg.reduct]
                   for meth, gg in g.groupby("method")}
        U = sorted(set(f for ss in sets_by.values() for s in ss for f in s))
        RU = np.abs(Xs[:, U].T @ Xs[:, U] / n)
        offs = {th: 0.0 for th in THETAS}                  # off-diagonal sum of C over all feature pairs, blockwise
        for i in range(0, m, 1000):
            B = np.abs(Xs[:, i:i + 1000].T @ Xs / n)
            for th in THETAS:
                offs[th] += float(np.where(B >= th, B, 0).sum())
        offs = {th: v - m for th, v in offs.items()}        # remove the diagonal (|rho_ff| = 1)
        pos = {f: j for j, f in enumerate(U)}
        for meth, sets in sets_by.items():
            Um = sorted(set(f for s in sets for f in s))
            idx = [pos[f] for f in Um]
            r = dict(dataset=name, method=meth, nogueira=nogueira(sets, m))
            r["check_C_eq_I"] = phi_c(sets, m, np.eye(len(Um)), Um, 0.0)
            for th in THETAS:
                Cu = RU[np.ix_(idx, idx)]; Cu = np.where(Cu >= th, Cu, 0); np.fill_diagonal(Cu, 1)
                r[f"phiC_{th}"] = phi_c(sets, m, Cu, Um, offs[th])
            rows.append(r)
            for s in sets:
                if len(s) > 1:
                    jj = [pos[f] for f in s]; R = RU[np.ix_(jj, jj)]
                    red.append(dict(dataset=name, method=meth, mean_abs_rho=(R.sum() - len(s)) / (len(s) * (len(s) - 1))))
        print(name, flush=True)
    out = pd.DataFrame(rows); out.to_csv(HERE / "stability_corr.csv", index=False)
    rd = pd.DataFrame(red).groupby(["dataset", "method"]).mean_abs_rho.mean().unstack()
    rd.to_csv(HERE / "redundancy.csv")
    print("max |check - nogueira| =", float((out.check_C_eq_I - out.nogueira).abs().max()))
    med = out.groupby("method")[["nogueira"] + [f"phiC_{t}" for t in THETAS]].median()
    print(med.round(3).sort_values("phiC_0.7", ascending=False).to_string())
    print("\nmean within-set |rho| (median over datasets):")
    print(rd.median().round(3).sort_values().to_string())


if __name__ == "__main__":
    main()
