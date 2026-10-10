"""Figures 2-6 of the manuscript, generated from the fold-level result files (style as e2/analyze_e2.py: serif, cm math)."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
E1, E3, FIG = HERE.parent / "e1", HERE.parent / "e3", Path(os.environ.get("FIG_DIR", HERE / "figures"))
FIG.mkdir(exist_ok=True)
_CM = os.environ.get("FIG_STYLE") == "cm"   # Computer Modern for the Springer (sn-jnl) version; Times for IEEE Access
plt.rcParams.update({"font.family": "serif", "font.serif": (["cmr10"] if _CM else ["Times New Roman", "Times", "STIXGeneral"]),
                     "mathtext.fontset": ("cm" if _CM else "stix"), "axes.formatter.use_mathtext": True,
                     "font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
STY = {"MHMG": ("k", "o"), "L1": ("tab:blue", "s"), "FCQ": ("tab:green", "^"), "ReliefF": ("tab:orange", "v"),
       "F": ("tab:red", "d"), "RFE": ("tab:purple", "P"), "NRS*": ("tab:brown", "X"), "FNRS": ("tab:gray", "*"),
       "All attributes": ("tab:cyan", "h")}


def box(ax):
    for s in ax.spines.values():
        s.set_visible(True)


raw = pd.read_csv(E1 / "raw_results.csv")
acc = raw.pivot_table(index="dataset", columns="method", values="knn_acc")
NRS = ["NRS_0.1", "NRS_0.2", "NRS_0.3"]
best = acc[NRS].idxmax(axis=1)
acc["NRS*"] = [acc.loc[i, c] for i, c in best.items()]
acc = acc.rename(columns={"ALL": "All attributes"})
COLS = ["MHMG", "F", "FCQ", "ReliefF", "L1", "RFE", "NRS*", "FNRS", "All attributes"]

# ---- Fig. A: conceptual illustration of granules and of a flipped label ----
rng = np.random.default_rng(3)
A = rng.normal([0.0, 0.0], 0.55, (14, 2)); Bc = rng.normal([2.3, 0.4], 0.55, (12, 2))
X = np.vstack([A, Bc]); y = np.r_[np.zeros(len(A), int), np.ones(len(Bc), int)]


def granules(X, y):
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(2)); en = y[:, None] != y[None, :]
    r = np.where(en, D, np.inf).min(1); ne = np.where(en, D, np.inf).argmin(1)
    return r, ne, (D < r[:, None]).sum(1)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True, sharey=True)
focus = int(np.argmin(((A - [0.6, 0.1]) ** 2).sum(1)))           # a class-0 sample near the boundary
yf = y.copy(); flip = int(np.argmin(((X[:len(A)] - (X[focus] + [0.25, 0.35])) ** 2).sum(1) + (np.arange(len(A)) == focus) * 9))
yf[flip] = 1
_, _, g_clean = granules(X, y)
for ax, lab, title in ((axes[0], y, "(a) clean labels"), (axes[1], yf, "(b) one label flipped")):
    r, ne, g = granules(X, lab)
    for c, col, mk in ((0, "tab:blue", "o"), (1, "tab:red", "s")):
        idx = np.where(lab == c)[0]
        ax.scatter(X[idx, 0], X[idx, 1], c=col, marker=mk, s=18, edgecolors="k", linewidths=0.4, zorder=3,
                   label=f"class {c + 1}")
    ax.add_patch(Circle(X[focus], r[focus], fill=False, ls="--", lw=1.0, color="k", zorder=2))
    members = np.where(np.sqrt(((X - X[focus]) ** 2).sum(1)) < r[focus])[0]
    ax.scatter(X[members, 0], X[members, 1], s=70, facecolors="none", edgecolors="tab:green", linewidths=1.0, zorder=4)
    ax.plot(*zip(X[focus], X[ne[focus]]), color="k", lw=0.8, zorder=2)
    ax.annotate(r"$x$", X[focus] + [-0.22, -0.28], fontsize=9)
    ax.annotate("nearest enemy", X[ne[focus]] + [0.08, 0.12], fontsize=7)
    if lab is yf:
        ax.annotate("flipped", X[flip] + [0.08, -0.3], fontsize=7, color="tab:red")
    extra = "" if lab is y else f"; {int(((g < g_clean) & (np.arange(len(y)) != flip)).sum())} granules shrink"
    ax.set_title(f"{title}: $|G_B(x)|={g[focus]}${extra}", fontsize=8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); box(ax)
axes[0].legend(loc="lower right", fontsize=6.5, frameon=True)
fig.tight_layout(); fig.savefig(FIG / "fig_concept.pdf"); plt.close(fig)

# ---- Fig. B: critical difference diagram (Friedman mean ranks, Nemenyi CD at alpha = 0.05) ----
N, k = len(acc), len(COLS)
ranks = pd.DataFrame(np.vstack([rankdata(-acc.loc[i, COLS]) for i in acc.index]), columns=COLS).mean().sort_values()
q = {9: 3.102}[k]                                                  # Demsar (2006), Table 5(a), alpha = 0.05
cd = q * np.sqrt(k * (k + 1) / (6 * N))
fig, ax = plt.subplots(figsize=(3.5, 2.1))
lo, hi = 1, k
ax.set_xlim(lo - 1.6, hi + 1.6); ax.set_ylim(0, 1.08); ax.axis("off")
AX = 0.82
ax.plot([lo, hi], [AX, AX], color="k", lw=0.8)
for t in range(lo, hi + 1):
    ax.plot([t, t], [AX, AX + 0.03], color="k", lw=0.8); ax.text(t, AX + 0.05, str(t), ha="center", fontsize=7)
ax.plot([lo, lo + cd], [1.0, 1.0], color="k", lw=1.2); ax.text(lo + cd / 2, 1.02, f"CD = {cd:.2f}", ha="center", fontsize=7, va="bottom")
for t in (lo, lo + cd):
    ax.plot([t, t], [0.98, 1.02], color="k", lw=0.8)
groups, r = [], ranks.values
for i in range(k):
    j = i
    while j + 1 < k and r[j + 1] - r[i] < cd:
        j += 1
    if j > i and not any(a <= i and j <= b for a, b in groups):
        groups.append((i, j))
for gi, (a, b) in enumerate(groups):
    yy = AX - 0.06 - 0.045 * gi
    ax.plot([r[a] - 0.06, r[b] + 0.06], [yy, yy], color="k", lw=2.2, solid_capstyle="round")
base = AX - 0.06 - 0.045 * len(groups) - 0.03
half = (k + 1) // 2
for i, (nm, rk) in enumerate(ranks.items()):
    left = i < half
    yy = base - 0.1 * (i if left else k - 1 - i)
    xx = lo - 0.2 if left else hi + 0.2
    ax.plot([rk, rk, xx], [AX, yy, yy], color="0.35", lw=0.6)
    ax.text(xx + (-0.06 if left else 0.06), yy, f"{nm} ({rk:.2f})", ha="right" if left else "left", va="center",
            fontsize=6.5, fontweight="bold" if nm == "MHMG" else None)
fig.tight_layout(); fig.savefig(FIG / "fig_cd.pdf"); plt.close(fig)
pd.Series({"CD": cd, "N": N, "k": k}).to_csv(FIG / "fig_cd_values.csv")

# ---- Fig. C: accuracy versus stability ----
stc = pd.read_csv(E1 / "stability_corr.csv")
stc["m2"] = stc.method
for ds, c in best.items():
    stc.loc[(stc.dataset == ds) & (stc.method == c), "m2"] = "NRS*"
stab = stc[~stc.m2.isin(NRS)].groupby("m2").nogueira.median()
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for m in ["MHMG", "F", "FCQ", "ReliefF", "L1", "RFE", "NRS*", "FNRS"]:
    c, mk = STY[m]
    ax.scatter(stab[m], acc[m].mean(), color=c, marker=mk, s=40, edgecolors="k", linewidths=0.4, zorder=3)
    ax.annotate(m + (" (50 attributes)" if m == "FNRS" else ""), (stab[m], acc[m].mean()), xytext=(4, 3),
                textcoords="offset points", fontsize=7, fontweight="bold" if m == "MHMG" else None)
ax.axhline(acc["All attributes"].mean(), color="0.5", ls="--", lw=0.8); ax.text(0.45, acc["All attributes"].mean() + 0.003, "all attributes", fontsize=7, color="0.4")
ax.set_xlabel(r"Selection stability $\Phi$ (median over datasets)"); ax.set_ylabel("Mean 3-NN accuracy")
ax.set_xlim(0, 0.75); ax.grid(alpha=0.3); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig_stab_acc.pdf"); plt.close(fig)

# ---- Fig. D: fraction of training radii that change when the test fold is removed ----
p2 = pd.read_csv(HERE / "check_prop2.csv").groupby("dataset").frac_changed.mean()
p2r = pd.read_csv(HERE / "check_prop2_random.csv").groupby("dataset")[["frac_changed", "s_over"]].mean()
order = p2.sort_values().index
fig, ax = plt.subplots(figsize=(3.5, 3.0))
yy = np.arange(len(order))
ax.scatter(p2.loc[order], yy, color="k", marker="o", s=22, label="MHMG attributes", zorder=3)
ax.scatter(p2r.loc[order, "frac_changed"], yy, color="tab:orange", marker="s", s=18, label="random attributes, same size", zorder=3)
ax.scatter(p2r.loc[order, "s_over"], yy, color="tab:blue", marker="|", s=60, label=r"$s/(n-1)$ (Corollary 1)", zorder=3)
LBL = {"lymphoma_2classes": "Lymphoma", "leukemia": "Leukemia", "BurkittLymphoma": "Burkitt", "GCM": "GCM",
       "tumors_C": "Tumors-C", "rsctc2010_1": "RSCTC-1", "rsctc2010_3": "RSCTC-3", "AP_Omentum_Uterus": "AP-OmU",
       "colon-cancer": "Colon", "duke-breast-cancer": "Duke", "CLL_SUB_111": "CLL-SUB", "GLIOMA": "Glioma",
       "GLI_85": "GLI-85", "Prostate_GE": "Prostate", "SMK_CAN_187": "SMK-CAN", "TOX_171": "TOX-171", "lung": "Lung"}
ax.set_yticks(yy); ax.set_yticklabels([LBL[d] for d in order], fontsize=6.5)
ax.set_xlabel("Fraction of training radii that change"); ax.set_xlim(0, 0.8); ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=2, fontsize=6.5, frameon=True); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig_radii.pdf"); plt.close(fig)

# ---- Fig. E: accuracy versus label-noise rate ----
n3 = pd.read_csv(E3 / "raw_noise.csv"); ns = pd.read_csv(E3 / "raw_noise_nrsstar.csv").assign(method="NRS*")
nf = pd.read_csv(E3 / "raw_noise_fnrs.csv")
clean = raw[raw.method.isin(["MHMG", "F", "FCQ", "ReliefF", "L1", "FNRS", "ALL"])].assign(rho=0.0)
clean_n = pd.concat([raw[(raw.dataset == ds) & (raw.method == c)] for ds, c in best.items()]).assign(method="NRS*", rho=0.0)
allr = pd.concat([clean, clean_n, n3, ns, nf], ignore_index=True).replace({"method": {"ALL": "All attributes"}})
curve = allr.groupby(["dataset", "rho", "method"]).knn_acc.mean().groupby(level=["rho", "method"]).mean().unstack()
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for m in ["MHMG", "NRS*", "FNRS", "L1", "FCQ", "ReliefF", "F", "All attributes"]:
    c, mk = STY[m]
    nearest = m in ("MHMG", "NRS*", "FNRS")
    ax.plot(curve.index, curve[m], color=c, marker=mk, ms=4, lw=1.4 if nearest else 0.9, ls="-" if nearest else "--", label=m)
ax.set_xticks([0, 0.1, 0.2]); ax.set_xlabel(r"Fraction $\rho$ of flipped training labels"); ax.set_ylabel("Mean 3-NN accuracy")
ax.grid(alpha=0.3); ax.legend(fontsize=6.5, ncol=4, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.2)); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig_noise.pdf"); plt.close(fig)
# ---- Fig. F: accuracy against panel size (same-size selectors, rough-set selectors, tuned selectors) ----
t4 = pd.read_csv(HERE.parent / "e4" / "raw_e4.csv")
base = raw[raw.method.isin(["MHMG", "F", "FCQ", "ReliefF", "L1", "RFE", "FNRS", "ALL"])]
nrs_s = pd.concat([raw[(raw.dataset == ds) & (raw.method == c)] for ds, c in best.items()]).assign(method="NRS*")
pts = pd.concat([base, nrs_s, t4], ignore_index=True).assign(size=lambda x: np.where(x.method == "ALL", x.n_attr, x["size"]))
ag = pts.groupby(["dataset", "method"])[["knn_acc", "size"]].mean().groupby(level="method").agg({"knn_acc": "mean", "size": "median"})
fig, ax = plt.subplots(figsize=(3.5, 2.9))
same = ["F", "FCQ", "ReliefF", "L1", "RFE"]
ax.scatter(ag.loc[same, "size"], ag.loc[same, "knn_acc"], s=16, color="0.6", zorder=2, label="same size as MHMG")
tuned = ["L1-cv", "FCQ-cv", "ReliefF-cv", "RFE-cv", "F-cv", "NRS-cv"]
ax.scatter(ag.loc[tuned, "size"], ag.loc[tuned, "knn_acc"], s=22, marker="s", color="tab:blue", zorder=3, label="size tuned by inner CV")
ax.scatter(ag.loc[["NRS*", "FNRS"], "size"], ag.loc[["NRS*", "FNRS"], "knn_acc"], s=22, marker="^", color="tab:brown",
           zorder=3, label="rough-set, own size")
ax.scatter(ag.loc["ALL", "size"], ag.loc["ALL", "knn_acc"], s=26, marker="h", color="tab:cyan", zorder=3, label="all attributes")
ax.scatter(ag.loc["MHMG", "size"], ag.loc["MHMG", "knn_acc"], s=46, marker="o", color="k", zorder=4, label="MHMG")
# Pareto front: no other point has fewer genes and higher accuracy
order = ag.assign(neg=-ag.knn_acc).sort_values(["size", "neg"]); front, best_acc = [], -1
for m_, r_ in order.iterrows():
    if r_.knn_acc > best_acc:
        front.append(m_); best_acc = r_.knn_acc
ax.plot(ag.loc[front, "size"], ag.loc[front, "knn_acc"], color="k", lw=0.7, ls=":", zorder=1)
off = {"MHMG": (5, -10), "L1-cv": (-30, 0), "FCQ-cv": (-34, 2), "ReliefF-cv": (-46, 1), "RFE-cv": (5, -7), "F-cv": (5, -3),
       "NRS-cv": (5, -4), "NRS*": (5, -2), "FNRS": (5, 2), "ALL": (-14, 5)}
for m_, (dx, dy) in off.items():
    ax.annotate("All attributes" if m_ == "ALL" else m_, (ag.loc[m_, "size"], ag.loc[m_, "knn_acc"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=6)
ax.set_xscale("log"); ax.set_ylim(0.69, 0.845); ax.set_xlabel("Number of selected attributes (median, log scale)"); ax.set_ylabel("Mean 3-NN accuracy")
ax.grid(alpha=0.3); ax.legend(fontsize=6, frameon=True, loc="lower right"); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig_pareto.pdf"); plt.close(fig)
print("Pareto front:", front)
print("figures written; CD =", round(cd, 3), "; mean ranks:", ranks.round(2).to_dict(), "; cliques:", groups)
