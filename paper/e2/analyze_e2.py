"""E2 analysis (written before the E2 results were inspected; descriptive, the study protocol). Reads raw_curves.csv only.
Outputs: curves.csv, e2_numbers.tex (macros), manuscript/figures/fig_curves.pdf."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "manuscript" / "figures"
d = pd.read_csv(HERE / "raw_curves.csv")
assert d.groupby(["dataset", "rep", "fold"]).ngroups == 255
METH = ["MHMG", "L1", "FCQ", "ReliefF", "F"]
ds_k = d.groupby(["dataset", "method", "k"])[["knn_acc", "svm_acc"]].mean().reset_index()
curve = ds_k.groupby(["method", "k"])[["knn_acc", "svm_acc"]].mean().unstack(0)
curve.to_csv(HERE / "curves.csv")
mac = []
K = sorted(ds_k.k.unique())
wins = {}
for other in ("L1", "FCQ"):
    better = 0
    for k in K:
        a = ds_k[(ds_k.method == "MHMG") & (ds_k.k == k)].set_index("dataset").knn_acc
        b = ds_k[(ds_k.method == other) & (ds_k.k == k)].set_index("dataset").knn_acc
        a, b = a.align(b, join="inner")
        better += int(np.median(a - b) > 0)
    wins[other] = better
    mac.append(f"\\newcommand{{\\curveWins{other.replace('1', 'one')}}}{{{better}}}")
mac.append(f"\\newcommand{{\\curveK}}{{{len(K)}}}")
# where is the stopping point? best k on MHMG's own path vs its stopping size, per dataset (3-NN)
stop = d[d.method == "MHMG"].groupby("dataset").mhmg_stop.mean()
mh = ds_k[ds_k.method == "MHMG"].pivot(index="dataset", columns="k", values="knn_acc")
kbest = mh.idxmax(axis=1)
at_stop = pd.Series({ds: mh.loc[ds, int(min(max(1, round(stop[ds])), mh.columns.max()))] for ds in mh.index})
gap = mh.max(axis=1) - at_stop
mac.append(f"\\newcommand{{\\stopGapMed}}{{{gap.median():.3f}}}")
mac.append(f"\\newcommand{{\\stopGapMax}}{{{gap.max():.3f}}}")
mac.append(f"\\newcommand{{\\kBestMed}}{{{kbest.median():.0f}}}")
mac.append(f"\\newcommand{{\\stopMed}}{{{stop.median():.1f}}}")
for k in (5, 10, 20):
    for mth in ("MHMG", "L1", "FCQ"):
        v = curve[("knn_acc", mth)].loc[k]
        mac.append(f"\\newcommand{{\\cur{mth.replace('1', 'one')}k{k}}}{{{v:.3f}}}".replace("k5", "kfive")
                   .replace("k10", "kten").replace("k20", "ktwenty"))
(HERE / "e2_numbers.tex").write_text("\n".join(mac) + "\n")
print("\n".join(mac))
print(pd.DataFrame({"stop": stop, "kbest": kbest, "gap": gap}).round(3).to_string())
# figure
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharex=True)
style = {"MHMG": ("k", "o"), "L1": ("tab:blue", "s"), "FCQ": ("tab:green", "^"), "ReliefF": ("tab:orange", "v"),
         "F": ("tab:red", "d")}
for ax, key, title in zip(axes, ("knn_acc", "svm_acc"), ("3-NN", "SVM-RBF")):
    for m_ in METH:
        c, mk = style[m_]
        ax.plot(K, curve[(key, m_)].loc[K], color=c, marker=mk, ms=3, lw=1.2 if m_ == "MHMG" else 0.9, label=m_)
    ax.axvline(stop.median(), color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("Number of selected genes $q$"); ax.set_title(title); ax.grid(alpha=0.3)
    for s in ax.spines.values():
        s.set_visible(True)
axes[0].set_ylabel("Mean accuracy (17 datasets)")
axes[1].legend(loc="lower right", fontsize=7, frameon=True)
fig.tight_layout(); FIG.mkdir(exist_ok=True); fig.savefig(FIG / "fig_curves.pdf"); print("figure written")
