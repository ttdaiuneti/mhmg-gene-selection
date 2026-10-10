"""Macros for the tuned q<=10 comparison (E5), the rounding sensitivity (e1/check_rounding.csv) and the expected-energy
variant (MHMG-Stable/kill). Writes extra_numbers.tex. Never type these numbers by hand."""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; G = H.parent; ROOT = G.parent
M = {}
def mac(k, v, fmt="{:.3f}"):
    t = v if isinstance(v, str) else fmt.format(v)
    M[k] = t.replace("-0.000", "0.000").replace("+0.000", "0.000")
s = pd.read_csv(G / "e5" / "e5_summary.csv").set_index("method")
for meth, key in (("L1-cv10", "Lone"), ("F-cv10", "F"), ("FCQ-cv10", "Fcq"), ("ReliefF-cv10", "Relief"), ("RFE-cv10", "Rfe")):
    r = s.loc[meth]
    mac(f"tenAcc{key}", r.mean_acc); mac(f"tenSize{key}", r.med_size, "{:.1f}"); mac(f"tenStab{key}", r.stab, "{:.2f}")
    mac(f"tenDiff{key}", r.mean_diff, "{:+.3f}"); mac(f"tenCiLo{key}", r.ci_lo, "{:+.3f}"); mac(f"tenCiHi{key}", r.ci_hi, "{:+.3f}")
    mac(f"tenW{key}", str(int(r.wins))); mac(f"tenL{key}", str(int(r.losses))); mac(f"tenHolm{key}", r.p_holm, "{:.3f}")
    mac(f"tenTime{key}", r.time, "{:.2f}")
    mac(f"tenNonInf{key}", "yes" if r.noninferior else "no")
mac("tenNonInfCount", str(int(s.noninferior.sum())))
r = pd.read_csv(G / "e1" / "check_rounding.csv"); g = r.groupby("dataset").mean(numeric_only=True)
mac("rndSizeTwo", g["size"].median(), "{:.1f}"); mac("rndSizeThree", g["size3"].median(), "{:.1f}")
mac("rndAccTwo", g["knn_acc"].mean()); mac("rndAccThree", g["acc3"].mean())
mac("rndFolds", str(len(r))); mac("rndHigher", str(int((g["acc3"] > g["knn_acc"]).sum())))
e1 = pd.read_csv(G / "e1" / "raw_results.csv")
def nog(sets, m):
    Z = np.zeros((len(sets), m))
    for i, st in enumerate(sets): Z[i, st] = 1
    kb = Z.sum(1).mean(); return 1 - Z.var(0, ddof=1).mean() / (kb / m * (1 - kb / m))
def phi(d):
    return pd.Series({ds: nog([list(map(int, str(x).split("|"))) for x in gg.reduct], int(gg.n_attr.iloc[0])) for ds, gg in d.groupby("dataset")})
pm = phi(e1[e1.method == "MHMG"])
# the expected-energy variant (macros rm*) needs a separate raw file that is not part of this repository; the paper does not use it
kf = ROOT / "MHMG-Stable" / "kill" / "raw_kill.csv"
if kf.exists():
    k = pd.read_csv(kf); pr = phi(k)
    mac("rmPhi", pr.median(), "{:.2f}"); mac("rmPhiMhmg", pm.median(), "{:.2f}"); mac("rmPhiHigher", str(int((pr > pm).sum())))
    mac("rmAcc", k.groupby("dataset").knn_acc.mean().mean()); mac("rmTimeRatio", k.groupby("dataset").time_s.mean().median() / e1[e1.method == "MHMG"].groupby("dataset").time_s.mean().median(), "{:.0f}")

# ---- table for the tuned-within-ten comparison (E5) with SVM and CI
e1m = e1[e1.method == "MHMG"]
macc = e1m.groupby("dataset").knn_acc.mean().mean(); msvm = e1m.groupby("dataset").svm_acc.mean().mean()
msize = e1m.groupby("dataset")["size"].mean().median(); mtime = e1m.groupby("dataset").time_s.mean().median()
rows = [f"MHMG & {msize:.1f} & {macc:.3f} & {msvm:.3f} & {pm.median():.2f} & {mtime:.2f} & -- & -- & -- \\\\"]
for meth in ("L1-cv10", "FCQ-cv10", "ReliefF-cv10", "F-cv10", "RFE-cv10"):
    r = s5 = s.loc[meth]
    dtxt = f"{r.mean_diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]".replace("-", "$-$")
    rows.append(f"{meth.replace('-cv10', '')}-cv10 & {r.med_size:.1f} & {r.mean_acc:.3f} & {r.mean_svm:.3f} & {r.stab:.2f} & {r.time:.2f} & "
                + dtxt + f" & {int(r.wins)}/{int(r.ties)}/{int(r.losses)} & {r.p_holm:.3f} \\\\")
(H / "tables" / "tab_ten.tex").write_text(r"""\begin{table*}[!t]
\centering
\caption{MHMG against selectors that tune their size within ten attributes (17 datasets). Size: median number of
attributes; $\Phi$: median Nogueira index; time: median seconds per fold, tuning included.}
\label{tab:ten}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrlrr@{}}
\toprule
Method & Size & 3-NN & SVM & $\Phi$ & Time & $\Delta$ [95\% CI] & W/T/L & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")
e4 = pd.read_csv(G / "e4" / "raw_e4.csv")
mac("tunAccFcv", e4[e4.method == "F-cv"].groupby("dataset").knn_acc.mean().mean())

import re as _re
for _k, _v in _re.findall(r"\\newcommand\{\\(noiseDrop[A-Za-z]+)\}\{\$-\$([0-9.]+)\}", (H / "numbers.tex").read_text()):
    mac(_k.replace("noiseDrop", "noiseLoss"), _v)
(H / "extra_numbers.tex").write_text("".join(f"\\newcommand{{\\{a}}}{{{b}}}\n" for a, b in M.items()))
print(open(H / "extra_numbers.tex").read())
