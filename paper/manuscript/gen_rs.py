"""Table tab:rs and macros rs_numbers.tex for the comparison within the neighbourhood rough-set family (E-RS prereg,
e6rs/the study protocol), from e6rs/raw_ers.csv, e1/raw_results.csv and e4/raw_e4.csv. Never type these numbers by hand."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
H = Path(__file__).resolve().parent; G = H.parent
e1 = pd.read_csv(G / "e1" / "raw_results.csv"); e4 = pd.read_csv(G / "e4" / "raw_e4.csv"); rs = pd.read_csv(G / "e6rs" / "raw_ers.csv")
_fix = G / "e6rs" / "raw_ers_gbfix.csv"
if _fix.exists() and len(pd.read_csv(_fix)) == 255:   # GBNRS rerun with the reference loop (e6rs/AMENDMENT_gbnrs.md)
    rs = pd.concat([rs[rs.method != "GBNRS"], pd.read_csv(_fix)], ignore_index=True)
def per_ds(d, meth):
    x = d[d.method == meth]; g = x.groupby("dataset")
    full = g.size() == 15
    return pd.DataFrame({"acc": g.knn_acc.mean(), "svm": g.svm_acc.mean(), "size": g["size"].mean(), "t": g.time_s.mean(), "full": full})
T = {"MHMG": per_ds(e1, "MHMG")}
for e in ("0.1", "0.2", "0.3"):
    T[f"NRS ($\\varepsilon={e}$)"] = per_ds(e1, f"NRS_{e}")
nr = pd.concat({e: T[f"NRS ($\\varepsilon={e}$)"] for e in ("0.1", "0.2", "0.3")}, axis=1)
best = nr.xs("acc", axis=1, level=1).idxmax(1)
T["NRS$^*$ (best $\\varepsilon$ on test)"] = pd.DataFrame({c: [nr.loc[ds, (best[ds], c)] for ds in nr.index] for c in ("acc", "svm", "size", "t", "full")}, index=nr.index)
T["NRS-cv"] = per_ds(e4, "NRS-cv"); T["FNRS (50 attributes)"] = per_ds(e1, "FNRS")
for k, lab in (("FNRS-q", "FNRS at the MHMG size"), ("KNN-NRS", "KNN-NRS"), ("WKNRS", "WKNRS"), ("MHMG-pf", "MHMG-80"), ("ANRS", "Adaptive NRS-80"), ("GBNRS", "GBNRS-80")):
    if (rs.method == k).any(): T[lab] = per_ds(rs, k)
GA = [k for k in T if k not in ("MHMG", "MHMG-80", "Adaptive NRS-80", "GBNRS-80")]; GB = ["Adaptive NRS-80", "GBNRS-80"]
rng = np.random.default_rng(0); M = {}; rows = []
def compare(ref, oth):
    a, b = T[ref], T[oth]; ok = a.index[a.full & b.reindex(a.index).full.fillna(False).astype(bool)]
    d = (a.loc[ok, "acc"] - b.loc[ok, "acc"]).values; idx = rng.integers(0, len(d), (10000, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    p = wilcoxon(a.loc[ok, "acc"], b.loc[ok, "acc"], zero_method="zsplit").pvalue
    return dict(ok=len(ok), mean=d.mean(), lo=lo, hi=hi, w=int((d > 0).sum()), t=int((d == 0).sum()), l=int((d < 0).sum()), p=p)
res = {o: compare("MHMG", o) for o in GA}; res.update({o: compare("MHMG-80", o) for o in GB if o in T and "MHMG-80" in T})
def holm(keys):
    o = sorted(keys, key=lambda k: res[k]["p"]); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (len(o) - i) * res[k]["p"])); res[k]["ph"] = run
holm(GA); holm([g for g in GB if g in res])
def fmt(v): return f"{v:+.3f}".replace("-", "$-$")
lines = []
for lab in ["MHMG"] + GA + [g for g in ["MHMG-80"] + GB if g in T]:
    x = T[lab]; base = f"{lab} & {x['size'].median():.1f} & {x.acc.mean():.3f} & {x.svm.mean():.3f} & {x.t.median():.1f}"
    if lab in res:
        r = res[lab]; base += f" & {fmt(r['mean'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] & {r['w']}/{r['t']}/{r['l']} & " + ("$<$0.001" if r['ph'] < 0.001 else f"{r['ph']:.3f}") + f" & {r['ok']}"
    else:
        base += " & -- & -- & -- & 17"
    lines.append(base + r" \\"); 
    if lab == GA[-1]: lines.append(r"\midrule")
(H / "tables" / "tab_rs.tex").write_text(r"""\begin{table*}[!t]
\centering
\caption{Summary of the comparison within the neighbourhood rough-set family (17 datasets). Size: median over the datasets of the mean size; time: median seconds per fold; $D$: datasets with all 15 folds. NRS$^*$: best of the three radii on the test folds (Section~\ref{sec:exp:eps} gives the eight-radius version). In the lower block, $\Delta$ is relative to MHMG-80.}
\label{tab:rs}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrlrrr@{}}
\toprule
Model & Size & 3-NN & SVM & Time & $\Delta$ [95\% CI] & W/T/L & $p$ & $D$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")
key = {"NRS ($\\varepsilon=0.1$)": "NA", "NRS ($\\varepsilon=0.2$)": "NB", "NRS ($\\varepsilon=0.3$)": "NC", "FNRS at the MHMG size": "Fq", "KNN-NRS": "Knn", "WKNRS": "Wk", "Adaptive NRS-80": "An", "GBNRS-80": "Gb", "NRS-cv": "Ncv", "NRS$^*$ (best $\\varepsilon$ on test)": "Nstar", "FNRS (50 attributes)": "Ffull"}
for lab, k in key.items():
    if lab in res:
        r = res[lab]; M.update({f"rs{k}Acc": f"{T[lab].acc.mean():.3f}", f"rs{k}Diff": fmt(r["mean"]), f"rs{k}Lo": fmt(r["lo"]), f"rs{k}Hi": fmt(r["hi"]),
                                f"rs{k}W": str(r["w"]), f"rs{k}L": str(r["l"]), f"rs{k}P": ("$<$0.001" if r["ph"] < 0.001 else f"{r['ph']:.3f}"), f"rs{k}D": str(r["ok"]), f"rs{k}Size": f"{T[lab]['size'].median():.1f}",
                                f"rs{k}NonInf": "yes" if r["lo"] > -0.02 else "no"})
_g = rs[rs.method == "GBNRS"]; M["rsGbFoldMed"] = f"{_g['size'].median():.0f}"; M["rsGbShareSeventy"] = f"{100 * (_g['size'] >= 70).mean():.0f}"
M["rsMhmgPfAcc"] = f"{T['MHMG-80'].acc.mean():.3f}" if "MHMG-80" in T else "--"
M["rsNonInfA"] = str(sum(res[k]["lo"] > -0.02 for k in ("FNRS at the MHMG size", "KNN-NRS", "WKNRS") if k in res))
M["rsNonInfB"] = str(sum(res[k]["lo"] > -0.02 for k in GB if k in res))
(H / "rs_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(open(H / "tables" / "tab_rs.tex").read()); print(M)
