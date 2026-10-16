"""Generate every table and number macro of the manuscript from the fold-level result files (no hand-typed numbers)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "e1"
sys.path.insert(0, str(E1))
import data as DA  # noqa: E402

d = pd.read_csv(E1 / "raw_results.csv")
assert d.groupby("dataset").size().eq(165).all()
SAMEK = ["F", "FCQ", "ReliefF", "L1", "RFE"]
NRS = ["NRS_0.1", "NRS_0.2", "NRS_0.3"]
acc = d.pivot_table(index="dataset", columns="method", values="knn_acc")
best_delta = acc[NRS].idxmax(axis=1)
d["method2"] = d.method
for ds, c in best_delta.items():
    d.loc[(d.dataset == ds) & (d.method == c), "method2"] = "NRS*"
d = d[~d.method2.isin(NRS)]
COLS = ["MHMG"] + SAMEK + ["NRS*", "FNRS", "ALL"]
ORDER = DA.NAMES
LABEL = {"lymphoma_2classes": "Lymphoma", "leukemia": "Leukemia", "BurkittLymphoma": "Burkitt", "GCM": "GCM",
         "tumors_C": "Tumors-C", "rsctc2010_1": "RSCTC-1", "rsctc2010_3": "RSCTC-3", "AP_Omentum_Uterus": "AP-OmU",
         "colon-cancer": "Colon", "duke-breast-cancer": "Duke", "CLL_SUB_111": "CLL-SUB", "GLIOMA": "Glioma",
         "GLI_85": "GLI-85", "Prostate_GE": "Prostate", "SMK_CAN_187": "SMK-CAN", "TOX_171": "TOX-171", "lung": "Lung"}
g = d.groupby(["dataset", "method2"])
M = {k: g[k].mean().unstack().loc[ORDER, COLS] for k in ("knn_acc", "svm_acc", "size", "time_s")}
S = {k: g[k].std().unstack().loc[ORDER, COLS] for k in ("knn_acc", "svm_acc")}
macros = []


def sgn(v):
    """Signed number for text mode: typographic minus/plus."""
    return f"${v[0]}${v[1:]}" if isinstance(v, str) and v[:1] in "+-" and v[1:2].isdigit() else v


def boot_ci(diff, n_boot=10000, seed=0):
    """95% percentile bootstrap CI of the mean paired difference over datasets."""
    diff = np.asarray(diff, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n_boot, len(diff)))
    return np.percentile(diff[idx].mean(1), [2.5, 97.5])


def mac(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{sgn(val)}}}")


def acc_table(key, fname, caption, label):
    lines = [r"\begin{table*}[!t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(COLS) + "@{}}", r"\toprule",
             "Dataset & " + " & ".join(c.replace("ALL", "All attributes") for c in COLS) + r" \\", r"\midrule"]
    for ds in ORDER:
        row = M[key].loc[ds]; best = row[COLS[:-1]].max()
        cells = []
        for c in COLS:
            s = f"{row[c]:.3f}$\\pm${S[key].loc[ds, c]:.3f}"
            cells.append(f"\\textbf{{{s}}}" if c != "ALL" and abs(row[c] - best) < 5e-4 else s)
        lines.append(LABEL[ds] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", "Mean & " + " & ".join(f"{M[key][c].mean():.3f}" for c in COLS) + r" \\",
              r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    (HERE / "tables" / fname).write_text("\n".join(lines) + "\n")


acc_table("knn_acc", "tab_acc_knn.tex", "3-NN accuracy (mean$\\pm$std over 15 folds) of MHMG, the five same-size selectors, "
          "NRS$^*$, FNRS-50 and all attributes. Bold: best selector on each dataset.", "tab:acc_knn")
acc_table("svm_acc", "tab_acc_svm.tex", "SVM-RBF accuracy (mean$\\pm$std over $3\\times5$ folds); settings as in "
          "Table~\\ref{tab:acc_knn}.", "tab:acc_svm")

# datasets table
info = []
for ds in ORDER:
    X, y = DA.load(ds)
    info.append((LABEL[ds], X.shape[0], X.shape[1], len(np.unique(y)), "yes" if ds not in DA.NEW else "--",
                 "OpenML" if ds in DA.OPENML else "scikit-feature", M["size"].loc[ds, "MHMG"]))
lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Gene-expression datasets ($n$: samples, $m$: attributes, $K$: classes).}",
         r"\label{tab:data}", r"\scriptsize", r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrl@{}}", r"\toprule",
         r"Dataset & $n$ & $m$ & $K$ & Source \\", r"\midrule"]
lines += [f"{a} & {b} & {c} & {k} & {src} \\\\" for a, b, c, k, s, src, r in info]
lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(HERE / "tables" / "tab_data.tex").write_text("\n".join(lines) + "\n")

# Wilcoxon-Holm vs same-size selectors (3-NN dataset means), as in e1/analyze_e1.py
mk = M["knn_acc"]
ps = {c: wilcoxon(mk.MHMG, mk[c], zero_method="zsplit").pvalue for c in SAMEK}
order = sorted(ps, key=ps.get)
holm = {c: min(1.0, max(ps[o] * (len(order) - j) for j, o in enumerate(order[:i + 1]))) for i, c in enumerate(order)}
lines = [r"\begin{table}[!t]", r"\centering", r"\caption{MHMG against the five same-size selectors (3-NN, 17 datasets). $p$: uncorrected; "
         r"$p_{\mathrm{Holm}}$: Holm-corrected.}",
         r"\label{tab:wilcoxon}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
         r"vs. & Median $\Delta$ & Mean $\Delta$ [95\% CI] & W/T/L & $p$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
for c in SAMEK:
    diff = mk.MHMG - mk[c]
    w, t_, l_ = (diff > 5e-4).sum(), (diff.abs() <= 5e-4).sum(), (diff < -5e-4).sum()
    lo_, hi_ = boot_ci(diff)
    lines.append(f"{c} & {sgn(f'{np.median(diff):+.3f}')} & {sgn(f'{diff.mean():+.3f}')} [{sgn(f'{lo_:+.3f}')}, {sgn(f'{hi_:+.3f}')}] & "
                 f"{w}/{t_}/{l_} & {ps[c]:.3f} & {holm[c]:.3f} \\\\")
    mac(f"ciLo{c.replace('1', 'one')}", f"{lo_:+.3f}"); mac(f"ciHi{c.replace('1', 'one')}", f"{hi_:+.3f}")
    mac(f"meanDiff{c.replace('1', 'one')}", f"{diff.mean():+.3f}")
    mac(f"holm{c.replace('1', 'one')}", f"{holm[c]:.3f}")
    mac(f"meddiff{c.replace('1', 'one')}", f"{np.median(diff):+.3f}")
    mac(f"wins{c.replace('1', 'one')}", int(w)); mac(f"losses{c.replace('1', 'one')}", int(l_))
lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(HERE / "tables" / "tab_wilcoxon.tex").write_text("\n".join(lines) + "\n")

# stability / redundancy / time
st = pd.read_csv(E1 / "stability_corr.csv")
st["method2"] = st.method
for ds, c in best_delta.items():
    st.loc[(st.dataset == ds) & (st.method == c), "method2"] = "NRS*"
st = st[~st.method2.isin(NRS)]
rd = pd.read_csv(E1 / "redundancy.csv", index_col=0)
for ds, c in best_delta.items():
    rd.loc[ds, "NRS*"] = rd.loc[ds, c]
SC = ["MHMG"] + SAMEK + ["NRS*", "FNRS"]
med = st.groupby("method2")[["nogueira", "phiC_0.5", "phiC_0.7", "phiC_0.9"]].median().loc[SC]
lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Selection stability over the 15 training folds (Nogueira index $\Phi$, median over "
         r"datasets) and median selection time per fold.}", r"\label{tab:stab}",
         r"\scriptsize", r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr@{}}", r"\toprule",
         r"Method & $\Phi$ & Time (s) \\", r"\midrule"]
for c in SC:
    r = med.loc[c]
    lines.append(f"{c} & {r.nogueira:.3f} & {M['time_s'][c].median():.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(HERE / "tables" / "tab_stab.tex").write_text("\n".join(lines) + "\n")

# prose macros
mac("nData", 17); mac("nNew", len(DA.NEW))
for c, nm in (("MHMG", "Mhmg"), ("L1", "Lone"), ("FCQ", "Fcq"), ("ALL", "All"), ("FNRS", "Fnrs"), ("F", "F"),
              ("ReliefF", "Relief"), ("RFE", "Rfe"), ("NRS*", "Nrs")):
    mac(f"meanAcc{nm}", f"{mk[c].mean():.3f}")
    mac(f"meanSvm{nm}", f"{M['svm_acc'][c].mean():.3f}")
mac("medSizeMhmg", f"{M['size']['MHMG'].median():.1f}")
mac("pctGenesMhmg", f"{100 * (M['size']['MHMG'] / pd.Series({ds: DA.load(ds)[0].shape[1] for ds in ORDER})).median():.3f}")
mac("stabMhmg", f"{med.loc['MHMG', 'nogueira']:.3f}")
mac("medTimeMhmg", f"{M['time_s']['MHMG'].median():.1f}")
dA = mk.MHMG - mk.ALL
mac("hOneAll", int((dA >= -0.02).sum())); mac("hOneNew", int((dA[DA.NEW] >= -0.02).sum()))
from scipy.stats import friedmanchisquare, rankdata, spearmanr
FC = ["MHMG"] + SAMEK + ["NRS*", "FNRS", "ALL"]
mac("friedmanP", f"{friedmanchisquare(*[mk[c] for c in FC]).pvalue:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
rk = pd.DataFrame(np.vstack([rankdata(-mk.loc[i, FC]) for i in mk.index]), columns=FC).mean()
for c, nm in (("MHMG", "Mhmg"), ("L1", "Lone"), ("FNRS", "Fnrs"), ("FCQ", "Fcq"), ("ALL", "All"), ("RFE", "Rfe"),
              ("NRS*", "Nrs"), ("F", "F"), ("ReliefF", "Relief")):
    mac(f"rank{nm}", f"{rk[c]:.2f}")
sz = M["size"]
for c, nm in (("NRS*", "Nrs"), ("FNRS", "Fnrs")):
    pv = wilcoxon(sz.MHMG, sz[c], alternative='less', zero_method='zsplit').pvalue
    mac(f"hThreeP{nm}", f"{pv:.3f}" if pv >= 0.001 else "<0.001")
    mac(f"hThreeDiff{nm}", f"{np.median(mk.MHMG - mk[c]):+.3f}")
mac("medSizeNrs", f"{sz['NRS*'].median():.1f}")
mac("hTwoMain", f"{np.median((mk.MHMG - mk.FCQ)[DA.NEW]):+.3f}")
# when does MHMG win (exploratory): Spearman of MHMG - L1 / - best same-size selector vs data traits
traits = pd.DataFrame({ds: dict(n=DA.load(ds)[0].shape[0], m=DA.load(ds)[0].shape[1], K=len(np.unique(DA.load(ds)[1])))
                       for ds in ORDER}).T
traits["npc"] = traits.n / traits.K
tgt = {"L1": mk.MHMG - mk.L1, "best": mk.MHMG - mk[SAMEK].max(1)}
pmin, lines_w = 1.0, []
for t in ("n", "m", "K", "npc"):
    for nm, v in tgt.items():
        r, pv = spearmanr(traits[t].astype(float), v.loc[traits.index]); pmin = min(pmin, pv)
        lines_w.append(f"{t},{nm},{r:.3f},{pv:.3f}")
(HERE / "tables" / "when_wins.csv").write_text("trait,target,rho,p\n" + "\n".join(lines_w) + "\n")
mac("whenWinsMinP", f"{pmin:.3f}"); mac("whenWinsTests", len(lines_w))

# scalability of the running time: MHMG time vs dataset size (prose in "Costs of the criterion")
tmed = d[d.method == "MHMG"].groupby("dataset").time_s.mean().loc[ORDER]
rho_m, p_m = spearmanr(traits["m"].astype(float), tmed); rho_n, p_n = spearmanr(traits["n"].astype(float), tmed)
assert max(p_m, p_n) < 0.01   # the prose says "both p < 0.01"
mac("rhoTimeM", f"{rho_m:.2f}"); mac("rhoTimeN", f"{rho_n:.2f}")
_pf = lambda v: "<0.001" if v < 0.001 else f"={v:.3f}"   # P < 0.001 convention; the sign is part of the macro
mac("pTimeM", _pf(p_m)); mac("pTimeN", _pf(p_n))
mac("timeMax", f"{tmed.max():.1f}"); mac("timeMaxDs", LABEL[tmed.idxmax()])
mac("timeMin", f"{tmed.min():.1f}"); mac("timeMinDs", LABEL[tmed.idxmin()])

NM = {"MHMG": "Mhmg", "F": "F", "FCQ": "Fcq", "ReliefF": "Relief", "L1": "Lone", "RFE": "Rfe", "NRS*": "Nrs", "FNRS": "Fnrs"}
for c in SAMEK:
    mac(f"hOneCount{NM[c]}", int(((mk[c] - mk.ALL) >= -0.02).sum()))
mac("hOnePct", f"{100 * (dA >= -0.02).mean():.0f}")
for c in SC:
    mac(f"stab{NM[c]}x", f"{med.loc[c, 'nogueira']:.3f}")
    mac(f"red{NM[c]}", f"{rd[c].median():.3f}")
pc = med.loc["MHMG", ["phiC_0.5", "phiC_0.7", "phiC_0.9"]]
mac("phiCMhmgLo", f"{pc.min():.2f}"); mac("phiCMhmgHi", f"{pc.max():.2f}")
w = st.pivot(index="dataset", columns="method2", values="phiC_0.7")[["MHMG"] + SAMEK]
mac("phiCLastCount", int((w.rank(axis=1, method="min").MHMG == 1).sum()))
_m = pd.read_csv(E1 / "raw_results.csv"); _m = _m[_m.method == "MHMG"]      # reducts are stored in selection order
_first = _m.groupby("dataset").reduct.agg(lambda r: r.astype(str).str.split("|").str[0].nunique())
mac("firstGeneMed", f"{_first.median():.0f}"); mac("firstGeneVar", int((_first > 1).sum()))
for th, key in (("phiC_0.5", "Five"), ("phiC_0.9", "Nine"), ("nogueira", "Nog")):
    w2 = st.pivot(index="dataset", columns="method2", values=th)[["MHMG"] + SAMEK]
    mac(f"phiCLastCount{key}", int((w2.rank(axis=1, method="min").MHMG == 1).sum()))
nrs1 = d[d.method.isin(NRS)].groupby("dataset").time_s.mean()       # mean over the three deltas = one-delta run
mac("timeRatioNrs", f"{(M['time_s']['MHMG'] / nrs1.loc[ORDER]).median():.1f}")
mac("timeRatioLone", f"{(M['time_s']['MHMG'] / M['time_s']['L1']).median():.0f}")
t1 = pd.read_csv(HERE / "check_theorem1.csv"); t3 = t1[t1.k == 3]
assert t1.part_i.all() and t1.ii.all() and t1.iii.all()
mac("thmAccThree", f"{t3.loo_acc.median():.3f}")
t1k1 = t1[t1.k == 1]
mac("ivHypFolds", int((t1k1.radius_ties == 0).sum())); mac("ivTotalFolds", len(t1k1))
mac("ivEqualFolds", int(((t1k1.loo_acc - t1k1.A).abs() < 1e-12).sum())); mac("thmAThree", f"{t3.A.median():.3f}"); mac("thmBoundThree", f"{t3.bound.median():.3f}")
p2 = pd.read_csv(HERE / "check_prop2.csv"); p2r = pd.read_csv(HERE / "check_prop2_random.csv")
assert p2.agree.all()
mac("propMhmgFrac", f"{p2.frac_changed.median():.3f}"); mac("propRandFrac", f"{p2r.frac_changed.median():.3f}")
mac("propSOver", f"{p2r.s_over.median():.3f}")
# reduct-size table: MHMG vs NRS (each delta), NRS* and FNRS; sizes from the raw rows (method, not method2)
raw = pd.read_csv(E1 / "raw_results.csv")
SZ = ["MHMG", "NRS_0.1", "NRS_0.2", "NRS_0.3", "NRS*", "FNRS"]
gm = raw.groupby(["dataset", "method"])["size"]
smean, sstd = gm.mean().unstack(), gm.std().unstack()
amean = raw.groupby(["dataset", "method"])["knn_acc"].mean().unstack()
smean["NRS*"] = [smean.loc[ds, c] for ds, c in best_delta.items()]
sstd["NRS*"] = [sstd.loc[ds, c] for ds, c in best_delta.items()]
amean["NRS*"] = [amean.loc[ds, c] for ds, c in best_delta.items()]
smean, sstd, amean = smean.loc[ORDER, SZ], sstd.loc[ORDER, SZ], amean.loc[ORDER, SZ]
head = {"MHMG": "MHMG", "NRS_0.1": "NRS$_{0.1}$", "NRS_0.2": "NRS$_{0.2}$", "NRS_0.3": "NRS$_{0.3}$",
        "NRS*": "NRS$^*$", "FNRS": "FNRS"}
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Number of selected attributes (mean$\pm$std over $3\times5$ folds) for the selectors that choose their "
         r"own size. NRS$_\varepsilon$: NRS with radius $\varepsilon$; NRS$^*$: the $\varepsilon$ with the best mean 3-NN accuracy "
         r"per dataset; FNRS is capped at 50 attributes. W/L: datasets on which MHMG selects fewer/more attributes; $p$: "
         r"two-sided Wilcoxon signed-rank test on dataset means.}", r"\label{tab:size}", r"\scriptsize",
         r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l" + "r" * len(SZ) + "@{}}", r"\toprule",
         "Dataset & " + " & ".join(head[c] for c in SZ) + r" \\", r"\midrule"]
for ds in ORDER:
    cells = [f"{smean.loc[ds, c]:.1f}" + ("" if c == "FNRS" else f"$\\pm${sstd.loc[ds, c]:.1f}") for c in SZ]
    lines.append(LABEL[ds] + " & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
lines.append("Median & " + " & ".join(f"{smean[c].median():.1f}" for c in SZ) + r" \\")
wl, pv = [], []
for c in SZ[1:]:
    w_, l_ = int((smean.MHMG < smean[c]).sum()), int((smean.MHMG > smean[c]).sum())
    wl.append(f"{w_}/{l_}")
    if c == "FNRS":
        pv.append("--")
    else:
        p_ = wilcoxon(smean.MHMG, smean[c]).pvalue
        pv.append(f"{p_:.3f}" if p_ >= 0.001 else "$<$0.001")
    key = {"NRS_0.1": "NrsA", "NRS_0.2": "NrsB", "NRS_0.3": "NrsC", "NRS*": "Nrsstar", "FNRS": "Fnrs"}[c]
    mac(f"sizeWin{key}", w_); mac(f"sizeLoss{key}", l_); mac(f"sizeMed{key}", f"{smean[c].median():.1f}")
    if c != "FNRS":
        mac(f"sizeP{key}", f"{p_:.3f}" if p_ >= 0.001 else "<0.001")
    mac(f"accSize{key}", f"{amean[c].mean():.3f}")
lines.append(r"W/L vs MHMG & -- & " + " & ".join(wl) + r" \\")
lines.append(r"$p$ (Wilcoxon) & -- & " + " & ".join(pv) + r" \\")
lines.append("Mean 3-NN acc. & " + " & ".join(f"{amean[c].mean():.3f}" for c in SZ) + r" \\")
lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(HERE / "tables" / "tab_size.tex").write_text("\n".join(lines) + "\n")
nt = pd.read_csv(HERE / "check_nrs_target.csv")
assert (nt[["gammaC_0.1", "gammaC_0.2", "gammaC_0.3"]] == 1).all().all()
mac("minBetweenClass", f"{nt.min_between_class.min():.1f}")
# ---- E3: label noise (e3/raw_noise.csv, raw_noise_nrsstar.csv, raw_noise_fnrs.csv; rho = 0 from E1) ----
E3 = HERE.parent / "e3"
n3 = pd.read_csv(E3 / "raw_noise.csv"); ns = pd.read_csv(E3 / "raw_noise_nrsstar.csv"); nf = pd.read_csv(E3 / "raw_noise_fnrs.csv")
for t, k in ((n3, 6), (ns, 1), (nf, 1)):
    assert t.groupby(["dataset", "rho"]).size().eq(15 * k).all() and t.dataset.nunique() == 17 and set(t.rho) == {0.1, 0.2}
ns = ns.assign(method="NRS*")
clean = raw[raw.method.isin(["MHMG", "F", "FCQ", "ReliefF", "L1", "FNRS", "ALL"])].assign(rho=0.0)
clean_nrs = pd.concat([raw[(raw.dataset == ds) & (raw.method == c)] for ds, c in best_delta.items()]).assign(method="NRS*", rho=0.0)
allr = pd.concat([clean, clean_nrs, n3, ns, nf], ignore_index=True)
NC = ["MHMG", "F", "FCQ", "ReliefF", "L1", "NRS*", "FNRS", "ALL"]
acc3 = allr.groupby(["dataset", "rho", "method"]).knn_acc.mean().unstack()[NC]
A0, A1, A2 = (acc3.xs(r, level="rho").loc[ORDER] for r in (0.0, 0.1, 0.2))
D1, D2 = A1 - A0, A2 - A0
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Mean 3-NN accuracy when a fraction $\rho$ of the training labels is flipped, and drop from $\rho=0$. "
         r"$p$: Wilcoxon test of the drop at $\rho=0.2$ against MHMG (uncorrected); last column: datasets on which MHMG is "
         r"more accurate at $\rho=0.2$.}", r"\label{tab:noise}", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}", r"\toprule",
         r"Method & $\rho=0$ & $0.1$ & $0.2$ & $\Delta_{0.1}$ & $\Delta_{0.2}$ & $p$ & MHMG better \\", r"\midrule"]
for c in NC:
    if c == "MHMG":
        pc, wc = "--", "--"
    else:
        pv_ = wilcoxon(D2.MHMG, D2[c], zero_method="zsplit").pvalue
        pc, wc = (f"{pv_:.3f}" if pv_ >= 0.001 else "$<$0.001"), f"{int((A2.MHMG > A2[c]).sum())}/17"
    lines.append(f"{c.replace('ALL', 'All attributes')} & {A0[c].mean():.3f} & {A1[c].mean():.3f} & {A2[c].mean():.3f} & "
                 f"{sgn(f'{D1[c].mean():+.3f}')} & {sgn(f'{D2[c].mean():+.3f}')} & {pc} & {wc} \\\\")
lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(HERE / "tables" / "tab_noise.tex").write_text("\n".join(lines) + "\n")
NN = {"MHMG": "Mhmg", "F": "F", "FCQ": "Fcq", "ReliefF": "Relief", "L1": "Lone", "NRS*": "Nrs", "FNRS": "Fnrs", "ALL": "All"}
for c in NC:
    mac(f"noiseAcc{NN[c]}", f"{A2[c].mean():.3f}"); mac(f"noiseDrop{NN[c]}", f"{D2[c].mean():+.3f}")
    mac(f"noiseDropOne{NN[c]}", f"{D1[c].mean():+.3f}")
    if c != "MHMG":
        mac(f"noiseWin{NN[c]}", int((A2.MHMG > A2[c]).sum())); mac(f"noiseWinOne{NN[c]}", int((A1.MHMG > A1[c]).sum()))
# H4: one-sided, Holm over L1 and FCQ
ph = {c: wilcoxon(D2.MHMG, D2[c], alternative="less", zero_method="zsplit").pvalue for c in ("L1", "FCQ")}
oh = sorted(ph, key=ph.get)
hh = {c: min(1.0, max(ph[o] * (len(oh) - j) for j, o in enumerate(oh[:i + 1]))) for i, c in enumerate(oh)}
mac("hFourHolmFcq", f"{hh['FCQ']:.3f}"); mac("hFourHolmLone", f"{hh['L1']:.3f}")
mac("hFourMoreFcq", int((D2.MHMG < D2.FCQ).sum())); mac("hFourMoreLone", int((D2.MHMG < D2.L1).sum()))
for c in ("NRS*", "FNRS"):
    mac(f"noiseP{NN[c]}", f"{wilcoxon(D2.MHMG, D2[c], zero_method='zsplit').pvalue:.2f}")
mech = n3[n3.method == "MHMG"].groupby(["dataset", "rho"])[["ne_flipped", "share_flipped"]].mean().groupby(level="rho").median()
mac("mechNeOne", f"{100 * mech.loc[0.1, 'ne_flipped']:.1f}"); mac("mechShareOne", f"{100 * mech.loc[0.1, 'share_flipped']:.1f}")
mac("mechNeTwo", f"{100 * mech.loc[0.2, 'ne_flipped']:.1f}"); mac("mechShareTwo", f"{100 * mech.loc[0.2, 'share_flipped']:.1f}")
sz2 = n3[(n3.method == "MHMG")].groupby(["dataset", "rho"])["size"].mean().groupby(level="rho").median()
mac("noiseSizeOne", f"{sz2.loc[0.1]:.1f}"); mac("noiseSizeTwo", f"{sz2.loc[0.2]:.1f}")
# ---- tuned comparison (e4/raw_e4.csv; MHMG and ALL from E1 on the same folds; analysis as in e4/analyze_e4.py) ----
E4 = HERE.parent / "e4"
t4 = pd.read_csv(E4 / "raw_e4.csv")
TU = ["L1-cv", "FCQ-cv", "ReliefF-cv", "RFE-cv", "F-cv", "NRS-cv"]
assert len(t4) == 17 * 15 * len(TU)
_e1 = pd.read_csv(E1 / "raw_results.csv")
b4 = pd.concat([_e1[_e1.method.isin(["MHMG", "ALL"])], t4], ignore_index=True)
a4 = b4.pivot_table(index="dataset", columns="method", values="knn_acc")
s4 = b4.pivot_table(index="dataset", columns="method", values="size")
v4 = b4.pivot_table(index="dataset", columns="method", values="svm_acc")
tm4 = b4.groupby(["dataset", "method"]).time_s.mean().unstack().median()   # as in tab_stab


def _nog(sets, m):
    Z = np.zeros((len(sets), m))
    for i, s_ in enumerate(sets):
        Z[i, s_] = 1
    kb = Z.sum(1).mean()
    return 1 - Z.var(0, ddof=1).mean() / (kb / m * (1 - kb / m))


st4 = pd.Series({(ds, mt): _nog([list(map(int, str(r).split("|"))) for r in g.reduct], int(g.n_attr.iloc[0]))
                 for (ds, mt), g in b4[b4.method != "ALL"].groupby(["dataset", "method"])}).unstack().median()
p4 = {c: wilcoxon(a4.MHMG, a4[c], zero_method="zsplit").pvalue for c in TU}
o4 = sorted(p4, key=p4.get); h4, run_ = {}, 0.0
for i, c in enumerate(o4):
    run_ = max(run_, min(1.0, (len(o4) - i) * p4[c])); h4[c] = run_
NT = {"L1-cv": "Lone", "FCQ-cv": "Fcq", "ReliefF-cv": "Relief", "RFE-cv": "Rfe", "F-cv": "F", "NRS-cv": "Nrs"}
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Selectors that choose their own size by inner 3-fold cross-validation on the training part, against "
         r"MHMG on the same folds (17 datasets). Size: median number of attributes; $\Phi$: median Nogueira index; time: seconds "
         r"per fold including tuning (median over datasets of the mean); $\Delta$: median difference MHMG minus selector in 3-NN accuracy; "
         r"W/T/L for MHMG; $p_{\mathrm{Holm}}$: two-sided Wilcoxon test with Holm correction.}",
         r"\label{tab:tuned}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}", r"\toprule",
         r"Method & 3-NN & Size & $\Phi$ & Time & $\Delta$ & W/T/L & $p_{\mathrm{Holm}}$ \\", r"\midrule",
         f"MHMG & {a4.MHMG.mean():.3f} & {s4.MHMG.median():.1f} & {st4.MHMG:.3f} & {tm4.MHMG:.2f} & -- & -- & -- \\\\"]
for c in TU:
    dd = a4.MHMG - a4[c]
    ph_ = f"{h4[c]:.3f}" if h4[c] >= 0.001 else "$<$0.001"
    lines.append(f"{c} & {a4[c].mean():.3f} & {s4[c].median():.1f} & {st4[c]:.3f} & {tm4[c]:.2f} & "
                 f"{sgn(f'{dd.median():+.3f}')} & {int((dd > 0).sum())}/{int((dd == 0).sum())}/{int((dd < 0).sum())} & {ph_} \\\\")
    mac(f"tunAcc{NT[c]}", f"{a4[c].mean():.3f}"); mac(f"tunSize{NT[c]}", f"{s4[c].median():.0f}")
    mac(f"tunDiff{NT[c]}", f"{dd.median():+.3f}"); mac(f"tunLoss{NT[c]}", int((dd < 0).sum())); mac(f"tunWin{NT[c]}", int((dd > 0).sum()))
    mac(f"tunHolm{NT[c]}", f"{h4[c]:.3f}" if h4[c] >= 0.001 else "<0.001"); mac(f"tunStab{NT[c]}", f"{st4[c]:.3f}")
    mac(f"tunTime{NT[c]}", f"{tm4[c]:.2f}"); mac(f"tunSvm{NT[c]}", f"{v4[c].mean():.3f}")
lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
(HERE / "tables" / "tab_tuned.tex").write_text("\n".join(lines) + "\n")
mac("tunTimeMhmg", f"{tm4.MHMG:.2f}")
mac("hSevenP", f"{wilcoxon(s4.MHMG, s4['L1-cv'], alternative='less', zero_method='zsplit').pvalue:.4f}"
    if wilcoxon(s4.MHMG, s4['L1-cv'], alternative='less', zero_method='zsplit').pvalue >= 1e-4 else "<0.0001")
mac("tunSizeRatioLo", f"{min(s4[c].median() for c in TU if c != 'NRS-cv') / s4.MHMG.median():.0f}")
mac("tunSizeRatioHi", f"{max(s4[c].median() for c in TU if c != 'NRS-cv') / s4.MHMG.median():.0f}")
cfg = t4.groupby("method").config.apply(lambda s_: (s_ == 50).mean())
mac("tunAtFiftyLo", f"{100 * min(cfg[c] for c in ['F-cv', 'FCQ-cv', 'ReliefF-cv', 'RFE-cv']):.0f}")
mac("tunAtFiftyHi", f"{100 * max(cfg[c] for c in ['F-cv', 'FCQ-cv', 'ReliefF-cv', 'RFE-cv']):.0f}")
# mechanism control (e3/check_mechanism_control.py): nearest enemy is a flipped sample, under F and random genes
_mc = pd.read_csv(HERE.parent / "e3" / "check_mechanism_control.csv")
_mm = _mc.groupby(["dataset", "rho"]).mean(numeric_only=True).groupby(level="rho").median()
assert abs(100 * _mm.loc[0.1, "ne_MHMG"] - float(f"{100 * mech.loc[0.1, 'ne_flipped']:.1f}")) < 0.06   # reproduces E3
for rho_, key in ((0.1, "One"), (0.2, "Two")):
    mac(f"mechF{key}", f"{100 * _mm.loc[rho_, 'ne_F']:.1f}"); mac(f"mechRand{key}", f"{100 * _mm.loc[rho_, 'ne_random']:.1f}")
# fixed-size curves: MHMG vs L1 where MHMG leads (q = 3..5), two-sided Wilcoxon on dataset means
_cv = pd.read_csv(HERE.parent / "e2" / "raw_curves.csv")
_cm = _cv.groupby(["dataset", "method", "k"]).knn_acc.mean().unstack(["method", "k"])
_pq = {q: wilcoxon(_cm[("MHMG", q)], _cm[("L1", q)], zero_method="zsplit").pvalue for q in (3, 4, 5)}
mac("curvePMin", f"{min(_pq.values()):.2f}"); mac("curvePMax", f"{max(_pq.values()):.2f}")
_dq = {q: (_cm[("MHMG", q)] - _cm[("L1", q)]).mean() for q in (3, 4, 5)}
mac("curveDiffMin", f"{min(_dq.values()):.3f}"); mac("curveDiffMax", f"{max(_dq.values()):.3f}")
(HERE / "numbers.tex").write_text("\n".join(macros) + "\n")
print("\n".join(macros))
