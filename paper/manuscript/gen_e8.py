"""Macros e8_numbers.tex and tables tab_rad.tex (radius grid) and tab_ctrl.tex (control comparisons) from the E8-rev
runs (e8rev/raw_e8.csv, analysis as registered in e8rev/analyze_e8.py), plus the SVM comparison FNRS-50 vs MHMG
from e1. Never type these numbers by hand."""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; G = H.parent
sys.path.insert(0, str(G / "e8rev"))
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_e8 as A  # noqa: E402  (registered analysis; rewrites e8rev/analysis.txt identically)
M = {}
f3 = lambda v: f"{v:+.3f}".replace("-", "$-$")
for k, key in (("H13", "Nstar"), ("H14", "Ncv"), ("H15_0.1", "QA"), ("H15_0.2", "QB"), ("H15_0.3", "QC"), ("H16", "Rnd"),
               ("C2", "RndEighty"), ("C3", "KnnE"), ("D", "Three"), ("H15s_0.1", "QAs"), ("H15s_0.2", "QBs"), ("H15s_0.3", "QCs"),
               ("H16s", "Rnds"), ("Ds", "Threes")):
    r = A.R[k]
    M.update({f"ev{key}D": f3(r["mean"]), f"ev{key}Lo": f3(r["lo"]), f"ev{key}Hi": f3(r["hi"]), f"ev{key}W": str(r["w"]),
              f"ev{key}L": str(r["l"]), f"ev{key}P": ("$<$0.001" if r.get("ph", r["p"]) < 0.001 else f"{r.get('ph', r['p']):.3f}")})
e8 = A.e8; Aa = e8[e8.part == "A"]
for e, key in (("0.4", "Four"), ("0.5", "Five"), ("0.75", "SevenFive"), ("1.0", "One"), ("med", "Med")):
    x = Aa[Aa.method == f"NRS_{e}"]
    M[f"evEmpty{key}"] = str(int((x.status == "empty").sum()))
M["evFolds"] = str(len(Aa[Aa.method == "NRS_0.4"]))
M["evNstarAcc"] = f"{A.nstar8.mean():.3f}"; M["evNcvAcc"] = f"{A.ncv8.mean():.3f}"
M["evNstarLarge"] = str(int(A.best.isin(["NRS_0.4", "NRS_0.5", "NRS_0.75"]).sum()))
ke = e8[(e8.part == "C") & (e8.method == "KNN-NRS-E")]
M["evKnnESmall"] = str(int((ke["size"] <= 2).sum())); M["evKnnEFolds"] = str(len(ke))
D = e8[e8.part == "D"]
M["evThreeSize"] = f"{D.groupby('dataset')['size'].mean().median():.1f}"
M["evThreeAcc"] = f"{D.groupby('dataset').knn_acc.mean().mean():.3f}"
Ed = e8[e8.part == "E"]; e1 = A.e1; old = e1[e1.dataset == "lymphoma_2classes"]
dm_old = old[old.method == "MHMG"].knn_acc.mean() - max(old[old.method == f"NRS_{e}"].knn_acc.mean() for e in ("0.1", "0.2", "0.3"))
dm_new = Ed[Ed.method == "MHMG-inF"].knn_acc.mean() - max(Ed[Ed.method == f"NRS_{e}-inF"].knn_acc.mean() for e in ("0.1", "0.2", "0.3"))
M["evLymChange"] = f3(dm_new - dm_old)
an = A.cmp(A.dsm(pd.read_csv(G / "e6rs" / "raw_ers.csv"), "ANRS"), A.dsm(e8[e8.part == "C"], "Random80@MHMG-80"))
M.update(anRndD=f3(an["mean"]), anRndLo=f3(an["lo"]), anRndHi=f3(an["hi"]), anRndW=str(an["w"]))
gf = pd.read_csv(G / "e8rev" / "check_gamma_full.csv")
M.update(gcOneFolds=str(int((gf["gC_1.0"] == 1).sum())), gcSevenFolds=str(int((gf["gC_0.75"] == 1).sum())),
         gcMedMin=f"{gf['gC_med'].min():.2f}")
# FNRS-50 vs MHMG under the SVM (e1), uncorrected, as reported by the reviewer
s = A.cmp(A.dsm(e1, "FNRS", "svm_acc"), A.dsm(e1, "MHMG", "svm_acc"))
M.update(svmFfiftyD=f3(s["mean"]), svmFfiftyLo=f3(s["lo"]), svmFfiftyHi=f3(s["hi"]), svmFfiftyW=str(s["w"]), svmFfiftyP=f"{s['p']:.3f}")
(H / "e8_numbers.tex").write_text("".join(f"\\newcommand{{\\{a}}}{{{b}}}\n" for a, b in M.items()))

# radius grid table
grid = A.grid; rows = []
for e in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "med"):
    x = grid[f"NRS_{e}"]; ne = x[x["size"] > 0]
    emp = int((x["size"] == 0).sum()); ok = A.E[f"NRS_{e}"]
    accv = x.groupby("dataset").knn_acc.mean().reindex(A.NAMES)[ok]
    lab = "data-scaled" if e == "med" else e
    art = e == "1.0"   # non-empty reducts at eps = 1.0 exist only through rounding of the scaled values (e8rev/check_gamma_full.py)
    dag = "$^\\dagger$" if art else ""
    rows.append(f"{lab}{dag} & {emp} & {ne['size'].mean():.1f} & {int(ok.sum())} & " + (f"{accv.mean():.3f}" if ok.sum() and not art else "--") + r" \\"
                if len(ne) else f"{lab} & {emp} & -- & 0 & -- \\\\")
L = [r"\begin{table}[!t]", r"\centering",
     r"\caption{NRS over a wide grid of radii (17 datasets, 255 folds). Empty: folds with an empty reduct; size: mean number of "
     r"attributes of the non-empty reducts; $D$: datasets with a non-empty reduct in all 15 folds; 3-NN: mean over these datasets. "
     r"$^\dagger$Non-empty reducts at $\varepsilon=1.0$ arise only from rounding errors of the scaled values.}", r"\label{tab:rad}", r"\scriptsize",
     r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}", r"\toprule",
     r"$\varepsilon$ & Empty & Size & $D$ & 3-NN \\", r"\midrule"] + rows + [
     r"\midrule", f"NRS$^*_8$ (best on test) & -- & -- & 17 & {A.nstar8.mean():.3f} \\\\",
     f"NRS-cv$_8$ & -- & -- & 17 & {A.ncv8.mean():.3f} \\\\", f"MHMG & 0 & {A.e1[A.e1.method == 'MHMG']['size'].mean():.1f} & 17 & {A.mh.mean():.3f} \\\\",
     r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(H / "tables" / "tab_rad.tex").write_text("\n".join(L) + "\n")

# control comparisons table
def line(lab, k):
    r = A.R[k]; p = r.get("ph", r["p"])
    return f"{lab} & {f3(r['mean'])} [{f3(r['lo'])}, {f3(r['hi'])}] & {r['w']}/{r['t']}/{r['l']} & " + ("$<$0.001" if p < 0.001 else f"{p:.3f}") + r" \\"
L = [r"\begin{table}[!t]", r"\centering",
     r"\caption{Control comparisons (3-NN, 17 datasets). $\Delta$: MHMG minus the control; equal size: MHMG cut to the size of the NRS reduct; $p$: Holm-corrected over the first six rows, uncorrected in the last two.}",
     r"\label{tab:ctrl}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
     r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrr@{}}", r"\toprule", r"Control & $\Delta$ [95\% CI] & W/T/L & $p$ \\", r"\midrule",
     line("NRS$^*_8$ (best of 8 radii)", "H13"), line("NRS-cv$_8$", "H14"),
     line("NRS$_{0.1}$, equal size", "H15_0.1"), line("NRS$_{0.2}$, equal size", "H15_0.2"),
     line("NRS$_{0.3}$, equal size", "H15_0.3"), line("Random, MHMG size", "H16"), r"\midrule",
     line("Random, 80 attr.\\ (MHMG-80)", "C2"), line("KNN-NRS, Euclidean", "C3"),
     r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(H / "tables" / "tab_ctrl.tex").write_text("\n".join(L) + "\n")
print(M)
ers = pd.read_csv(G / "e6rs" / "raw_ers.csv"); kn = ers[ers.method == "KNN-NRS"]
M2 = {"knnOneFolds": str(int((kn["size"] == 1).sum())), "knnFolds": str(len(kn))}
with open(H / "e8_numbers.tex", "a") as fh:
    fh.write("".join(f"\\newcommand{{\\{a}}}{{{b}}}\n" for a, b in M2.items()))
print(M2)

# equal-size table (MHMG vs FNRS at 10/20/50 attributes; numbers from size_numbers.tex via gen_size.py)
L = [r"\begin{table}[!t]", r"\centering",
     r"\caption{MHMG and FNRS continued to the same number of attributes (mean accuracy over 17 datasets; $\Delta$ and W/L for 3-NN).}",
     r"\label{tab:eqsize}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
     r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrlrrr@{}}", r"\toprule",
     r" & \multicolumn{4}{c}{3-NN} & \multicolumn{2}{c}{SVM} \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-7}",
     r"Size & MHMG & FNRS & $\Delta$ [95\% CI] & W/L & MHMG & FNRS \\", r"\midrule"]
for n, k in (("10", "Ten"), ("20", "Twenty"), ("50", "Fifty")):
    L.append(f"{n} & \\szK{k}M & \\szK{k}F & \\szK{k}D{{}} [\\szK{k}Lo, \\szK{k}Hi] & \\szK{k}W/\\szK{k}L & \\szS{k}M & \\szS{k}F \\\\")
L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(H / "tables" / "tab_eqsize.tex").write_text("\n".join(L) + "\n")
