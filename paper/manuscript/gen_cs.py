"""Per-dataset tables of the neighbourhood rough-set family (3-NN, SVM, size and time), the Friedman/Nemenyi CD
diagram of the family and the radius-sensitivity figure, from the raw fold-level files. Writes tables/tab_fam_*.tex,
figures/fig_fam_cd.pdf, figures/fig_eps.pdf and cs_numbers.tex."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, friedmanchisquare
H = Path(__file__).resolve().parent; G = H.parent
sys.path.insert(0, str(G / "e1"))
import data as DA  # noqa: E402
LABEL = {"lymphoma_2classes": "Lymphoma", "leukemia": "Leukemia", "BurkittLymphoma": "Burkitt", "GCM": "GCM",
         "tumors_C": "Tumors-C", "rsctc2010_1": "RSCTC-1", "rsctc2010_3": "RSCTC-3", "AP_Omentum_Uterus": "AP-OmU",
         "colon-cancer": "Colon", "duke-breast-cancer": "Duke", "CLL_SUB_111": "CLL-SUB", "GLIOMA": "Glioma",
         "GLI_85": "GLI-85", "Prostate_GE": "Prostate", "SMK_CAN_187": "SMK-CAN", "TOX_171": "TOX-171", "lung": "Lung"}

_CM = os.environ.get("FIG_STYLE") == "cm"   # Computer Modern for the single-column layout; Times for the two-column layout
_TSP = os.environ.get("FIG_STYLE") == "tsp"   # Arial, labels of at least 8 pt, capital panel letters
_SANS = os.environ.get("FIG_STYLE") in ("sans", "tsp")   # Arial/Helvetica for journals that ask for sans-serif artwork
plt.rcParams.update({"font.family": "sans-serif" if _SANS else "serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.serif": (["cmr10"] if _CM else ["Times New Roman", "Times", "STIXGeneral"]),
                     "mathtext.fontset": ("cm" if _CM else "stixsans" if _SANS else "stix"), "axes.formatter.use_mathtext": True,
                     "font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
if _TSP:   # every text at least 8 pt
    import matplotlib.figure as _mf, matplotlib.text as _mt
    _orig_savefig = _mf.Figure.savefig
    def _savefig(self, *a, **k):
        for t in self.findobj(_mt.Text):
            if t.get_text() and t.get_fontsize() < 8: t.set_fontsize(8)
        return _orig_savefig(self, *a, **k)
    _mf.Figure.savefig = _savefig
ORDER = DA.NAMES
FIGD = Path(os.environ.get("FIG_DIR", H / "figures"))
e1 = pd.read_csv(G / "e1" / "raw_results.csv"); e4 = pd.read_csv(G / "e4" / "raw_e4.csv")
rs = pd.read_csv(G / "e6rs" / "raw_ers.csv"); fix = pd.read_csv(G / "e6rs" / "raw_ers_gbfix.csv")
assert len(fix) == 255
rs = pd.concat([rs[rs.method != "GBNRS"], fix], ignore_index=True)

# NRS*: radius with the highest mean 3-NN accuracy per dataset (chosen on the test folds, favours NRS)
nrs = e1[e1.method.str.startswith("NRS_")]
best = nrs.groupby(["dataset", "method"]).knn_acc.mean().unstack().idxmax(1)
nstar = pd.concat([nrs[(nrs.dataset == ds) & (nrs.method == m)] for ds, m in best.items()]).assign(method="NRS*")
F = {"MHMG": e1[e1.method == "MHMG"], "NRS$^*$": nstar, "NRS-cv": e4[e4.method == "NRS-cv"],
     "FNRS-50": e1[e1.method == "FNRS"], "FNRS-$q$": rs[rs.method == "FNRS-q"], "KNN-NRS": rs[rs.method == "KNN-NRS"],
     "WKNRS": rs[rs.method == "WKNRS"], "MHMG-80": rs[rs.method == "MHMG-pf"], "ANRS-80": rs[rs.method == "ANRS"],
     "GBNRS-80": rs[rs.method == "GBNRS"]}
for k, x in F.items():
    c = x.groupby("dataset").size()
    assert c.reindex(ORDER).eq(15).all(), (k, c)
A = ["MHMG", "NRS$^*$", "NRS-cv", "FNRS-50", "FNRS-$q$", "KNN-NRS", "WKNRS"]; B = ["MHMG-80", "ANRS-80", "GBNRS-80"]
COLS = A + B
mean = {v: pd.DataFrame({k: F[k].groupby("dataset")[v].mean() for k in COLS}).loc[ORDER] for v in ("knn_acc", "svm_acc", "size", "time_s")}
std = {v: pd.DataFrame({k: F[k].groupby("dataset")[v].std() for k in COLS}).loc[ORDER] for v in ("knn_acc", "svm_acc")}
M = {}


def ranks(df, cols):
    return pd.DataFrame(np.vstack([rankdata(-df.loc[i, cols].round(3)) for i in df.index]), index=df.index, columns=cols)


HEAD = r"\multicolumn{%d}{c}{All attributes} & \multicolumn{%d}{c}{80 attributes of largest variance} \\" % (len(A), len(B))
RULE = r"\cmidrule(lr){2-%d}\cmidrule(l){%d-%d}" % (len(A) + 1, len(A) + 2, len(COLS) + 1)


def acc_table(v, fname, label, caption):
    m, s = mean[v], std[v]
    L = [r"\begin{table*}[!t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\scriptsize",
         r"\setlength{\tabcolsep}{1.6pt}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(COLS) + "@{}}", r"\toprule",
         " & " + HEAD, RULE, "Dataset & " + " & ".join(COLS) + r" \\", r"\midrule"]
    for ds in ORDER:
        cells = []
        for blk in (A, B):
            top = m.loc[ds, blk].round(3).max()
            for c in blk:
                t = f"{m.loc[ds, c]:.3f}$\\pm${s.loc[ds, c]:.3f}"
                cells.append(f"\\textbf{{{t}}}" if round(m.loc[ds, c], 3) == top else t)
        L.append(LABEL[ds] + " & " + " & ".join(cells) + r" \\")
    rk = pd.concat([ranks(m, A).mean(), ranks(m, B).mean()])
    L += [r"\midrule", "Average & " + " & ".join(f"{m[c].mean():.3f}" for c in COLS) + r" \\",
          "Average rank & " + " & ".join(f"{rk[c]:.2f}" for c in COLS) + r" \\",
          r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    (H / "tables" / fname).write_text("\n".join(L) + "\n")
    return rk


CAPN = "Bold: best value in each block; ranks are computed within each block."
rk3 = acc_table("knn_acc", "tab_fam_knn.tex", "tab:fam_knn",
                "3-NN accuracy (mean$\\pm$std over 15 folds) of the neighbourhood rough-set models. " + CAPN)
rks = acc_table("svm_acc", "tab_fam_svm.tex", "tab:fam_svm",
                "SVM accuracy (mean$\\pm$std over 15 folds) of the neighbourhood rough-set models. " + CAPN)

def tfmt(t):
    return "$<$0.1" if t < 0.05 else f"{t:.1f}"


# size and time per dataset
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Mean number of selected attributes and, in parentheses, mean running time in seconds per fold. Settings as "
     r"in Table~\ref{tab:fam_knn}.}", r"\label{tab:fam_size}", r"\scriptsize", r"\setlength{\tabcolsep}{1.6pt}",
     r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(COLS) + "@{}}", r"\toprule",
     " & " + HEAD, RULE, "Dataset & " + " & ".join(COLS) + r" \\", r"\midrule"]
for ds in ORDER:
    L.append(LABEL[ds] + " & " + " & ".join(f"{mean['size'].loc[ds, c]:.1f} ({tfmt(mean['time_s'].loc[ds, c])})" for c in COLS) + r" \\")
L += [r"\midrule", "Average & " + " & ".join(f"{mean['size'][c].mean():.1f} ({tfmt(mean['time_s'][c].mean())})" for c in COLS) + r" \\",
      r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
(H / "tables" / "tab_fam_size.tex").write_text("\n".join(L) + "\n")

# Friedman test and Nemenyi CD over the six models that select from all attributes and return small reducts.
# FNRS-50 (50 attributes in every fold) is left out of the rank comparison; it is compared at equal sizes in tab_eqsize.
A_CD = [c for c in A if c != "FNRS-50"]
m3 = mean["knn_acc"][A_CD]
chi, p = friedmanchisquare(*[m3[c].values for c in A_CD])
N, k = len(m3), len(A_CD)
q = {6: 2.850, 7: 2.949}[k]                                         # Demsar (2006), Table 5(a), alpha = 0.05
cd = q * np.sqrt(k * (k + 1) / (6 * N))
r = ranks(m3, A_CD).mean().sort_values()
ms = mean["svm_acc"][A_CD]; chis, ps = friedmanchisquare(*[ms[c].values for c in A_CD]); rs_ = ranks(ms, A_CD).mean()
sig = lambda a, b: abs(r[a] - r[b]) >= cd
M.update(famFriedChi=f"{chi:.1f}", famFriedP=("$<$0.001" if p < 0.001 else f"{p:.3f}"), famCD=f"{cd:.2f}", famK=str(k),
         famRankMhmg=f"{r['MHMG']:.2f}", famRankWk=f"{r['WKNRS']:.2f}",
         famRankFq=f"{r['FNRS-$q$']:.2f}", famRankNstar=f"{r['NRS$^*$']:.2f}", famRankKnn=f"{r['KNN-NRS']:.2f}",
         famSvmRankMhmg=f"{rs_['MHMG']:.2f}", famSvmRankFq=f"{rs_['FNRS-$q$']:.2f}",
         famSvmFriedP=("$<$0.001" if ps < 0.001 else f"{ps:.3f}"),
         famMedSize=f"{max(F[c]['size'].median() for c in A_CD):.0f}")
assert not sig("MHMG", "FNRS-$q$") and not sig("MHMG", "WKNRS") and all(sig("MHMG", c) for c in ("NRS$^*$", "NRS-cv", "KNN-NRS")), r
fig, ax = plt.subplots(figsize=((5.0 if _TSP else 3.5), 1.6))
lo, hi = 1, k
ax.set_xlim(lo - 1.6, hi + 1.6); ax.set_ylim(0.2, 1.1); ax.axis("off")
AX = 0.82
ax.plot([lo, hi], [AX, AX], color="k", lw=0.8)
for t in range(lo, hi + 1):
    ax.plot([t, t], [AX, AX + 0.03], color="k", lw=0.8); ax.text(t, AX + 0.05, str(t), ha="center", fontsize=7)
ax.plot([lo, lo + cd], [1.0, 1.0], color="k", lw=1.2); ax.text(lo + cd / 2, 1.02, f"CD = {cd:.2f}", ha="center", fontsize=7, va="bottom")
for t in (lo, lo + cd):
    ax.plot([t, t], [0.98, 1.02], color="k", lw=0.8)
groups, rv = [], r.values
for i in range(k):
    j = i
    while j + 1 < k and rv[j + 1] - rv[i] < cd:
        j += 1
    if j > i and not any(a <= i and j <= b for a, b in groups):
        groups.append((i, j))
for gi, (a, b) in enumerate(groups):
    yy = AX - 0.06 - 0.045 * gi
    ax.plot([rv[a] - 0.06, rv[b] + 0.06], [yy, yy], color="k", lw=2.2, solid_capstyle="round")
base = AX - 0.06 - 0.045 * len(groups) - 0.03
half = (k + 1) // 2
for i, (nm, rk_) in enumerate(r.items()):
    left = i < half
    yy = base - 0.1 * (i if left else k - 1 - i)
    xx = lo - 0.2 if left else hi + 0.2
    ax.plot([rk_, rk_, xx], [AX, yy, yy], color="0.35", lw=0.6)
    ax.text(xx + (-0.06 if left else 0.06), yy, f"{nm} ({rk_:.2f})", ha="right" if left else "left", va="center",
            fontsize=6.5, fontweight="bold" if nm == "MHMG" else None)
fig.tight_layout(); fig.savefig(FIGD / "fig_fam_cd.pdf"); plt.close(fig)

# radius sensitivity: NRS at each radius against MHMG, which has no radius
# radii 0.1-0.3 from e1, 0.4-0.5 from the E8 grid (e8rev); all 17 datasets have non-empty reducts here
e8 = pd.read_csv(G / "e8rev" / "raw_e8.csv"); grid = pd.concat([e1[e1.method.str.startswith("NRS_")], e8[e8.part == "A"]])
eps = ["0.1", "0.2", "0.3", "0.4", "0.5"]
for e in eps:
    assert (grid[grid.method == f"NRS_{e}"]["size"] > 0).all() and len(grid[grid.method == f"NRS_{e}"]) == 255
na = [grid[grid.method == f"NRS_{e}"].groupby("dataset").knn_acc.mean().mean() for e in eps]
ns = [grid[grid.method == f"NRS_{e}"].groupby("dataset")["size"].mean().mean() for e in eps]
ma, ms = mean["knn_acc"]["MHMG"].mean(), mean["size"]["MHMG"].mean()
nwin = [int((grid[grid.method == f"NRS_{e}"].groupby("dataset").knn_acc.mean().loc[ORDER] > mean["knn_acc"]["MHMG"]).sum()) for e in eps]
fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.4) if _TSP else (3.5, 1.6))
x = [float(e) for e in eps]
axs[0].plot(x, na, "s-", color="tab:brown", ms=4, label="NRS"); axs[0].axhline(ma, color="k", lw=1, ls="--", label="MHMG")
axs[0].set_xlabel(r"radius $\varepsilon$"); axs[0].set_ylabel("mean 3-NN accuracy"); axs[0].set_xticks(x)
axs[0].set_ylim(min(na) - 0.02, ma + 0.02)
axs[1].plot(x, ns, "s-", color="tab:brown", ms=4); axs[1].axhline(ms, color="k", lw=1, ls="--")
axs[1].set_xlabel(r"radius $\varepsilon$"); axs[1].set_ylabel("mean reduct size"); axs[1].set_xticks(x)
axs[1].set_ylim(0, max(ns + [ms]) * 1.25)
axs[0].legend(fontsize=6.5, frameon=True, loc="center left", bbox_to_anchor=(0.02, 0.62))
for a in axs:
    a.grid(alpha=0.3)
if _TSP:
    for a, L in zip(axs, "AB"):
        a.text(0.02, 0.98, L, transform=a.transAxes, fontsize=10, fontweight="bold", va="top")
fig.tight_layout(); fig.savefig(FIGD / "fig_eps.pdf"); plt.close(fig)
for e, a_, s_, w in zip(("A", "B", "C", "D", "E"), na, ns, nwin):
    M[f"eps{e}Acc"] = f"{a_:.3f}"; M[f"eps{e}Size"] = f"{s_:.1f}"; M[f"eps{e}Above"] = str(w)
M["epsMaxAbove"] = str(max(nwin)); M["epsBestAcc"] = f"{max(na):.3f}"
(H / "cs_numbers.tex").write_text("".join(f"\\newcommand{{\\{a}}}{{{b}}}\n" for a, b in M.items()))
print(M)
