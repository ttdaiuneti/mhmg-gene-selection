"""Analysis of E8-rev (paired differences, bootstrap CI, Holm correction, descriptive parts). Writes analysis.txt;
e8_numbers.tex is written by manuscript/gen_e8.py."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
H = Path(__file__).resolve().parent; G = H.parent
e8 = pd.read_csv(H / "raw_e8.csv"); e1 = pd.read_csv(G / "e1" / "raw_results.csv")
NAMES = sorted(e1.dataset.unique()); out, M = [], {}


def dsm(d, meth, col="knn_acc"):
    return d[d.method == meth].groupby("dataset")[col].mean().reindex(NAMES)


def cmp(a, b):
    ok = a.notna() & b.notna(); d = (a[ok] - b[ok]).values
    idx = np.random.default_rng(0).integers(0, len(d), (10000, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    p = wilcoxon(a[ok], b[ok], zero_method="zsplit").pvalue
    return dict(n=int(ok.sum()), mean=d.mean(), lo=lo, hi=hi, w=int((d > 0).sum()), t=int((d == 0).sum()), l=int((d < 0).sum()), p=p)


mh = dsm(e1, "MHMG"); mhs = dsm(e1, "MHMG", "svm_acc")
# ---- A: radius grid
A = e8[e8.part == "A"]
for e in ("0.4", "0.5", "0.75", "1.0", "med"):
    x = A[A.method == f"NRS_{e}"]
    out.append(f"NRS_{e}: folds={len(x)} empty={int((x.status == 'empty').sum())} timeout={int((x.status == 'timeout').sum())} "
               f"mean size (non-empty)={x[x['size'] > 0]['size'].mean():.2f}")
grid = {f"NRS_{e}": e1[e1.method == f"NRS_{e}"] for e in ("0.1", "0.2", "0.3")}
grid.update({f"NRS_{e}": A[A.method == f"NRS_{e}"] for e in ("0.4", "0.5", "0.75", "1.0", "med")})
elig = {k: v.groupby("dataset").apply(lambda g: len(g) == 15 and (g["size"] > 0).all() and (g.get("status", pd.Series(["ok"] * len(g))) == "ok").all()).reindex(NAMES).fillna(False)
        for k, v in grid.items()}
acc = pd.DataFrame({k: v.groupby("dataset").knn_acc.mean().reindex(NAMES) for k, v in grid.items()})
svm = pd.DataFrame({k: v.groupby("dataset").svm_acc.mean().reindex(NAMES) for k, v in grid.items()})
E = pd.DataFrame(elig)
best = acc.where(E).idxmax(1)
nstar8 = pd.Series({ds: acc.loc[ds, best[ds]] for ds in NAMES}); nstar8s = pd.Series({ds: svm.loc[ds, best[ds]] for ds in NAMES})
out.append("NRS*8 chosen radius counts: " + str(best.value_counts().to_dict()))
cv = A[A.method == "NRS-cv8"]; out.append(f"NRS-cv8 timeouts={int((cv.status == 'timeout').sum())}; choices={cv.eps_choice.value_counts().to_dict()}")
ncv8 = cv[cv.status == "ok"].groupby("dataset").knn_acc.mean().reindex(NAMES)
R = {"H13": cmp(mh, nstar8), "H14": cmp(mh, ncv8)}
# ---- B: equal size
B = e8[e8.part == "B"]
for e in ("0.1", "0.2", "0.3"):
    R[f"H15_{e}"] = cmp(dsm(B, f"MHMG@NRS_{e}"), dsm(e1, f"NRS_{e}"))
    R[f"H15s_{e}"] = cmp(dsm(B, f"MHMG@NRS_{e}", "svm_acc"), dsm(e1, f"NRS_{e}", "svm_acc"))
# ---- C
C = e8[e8.part == "C"]
R["H16"] = cmp(mh, dsm(C, "Random@MHMG")); R["H16s"] = cmp(mhs, dsm(C, "Random@MHMG", "svm_acc"))
ers = pd.read_csv(G / "e6rs" / "raw_ers.csv")
R["C2"] = cmp(dsm(ers, "MHMG-pf"), dsm(C, "Random80@MHMG-80"))
ke = C[C.method == "KNN-NRS-E"]
out.append(f"KNN-NRS-E: size<=2 in {int((ke['size'] <= 2).sum())}/{len(ke)} folds; median size {ke['size'].median()}; timeouts {int((ke.status == 'timeout').sum())}")
R["C3"] = cmp(mh, dsm(C, "KNN-NRS-E"))
# ---- D
D = e8[e8.part == "D"]
R["D"] = cmp(dsm(D, "MHMG-3dp"), mh); R["Ds"] = cmp(dsm(D, "MHMG-3dp", "svm_acc"), mhs)
out.append(f"MHMG-3dp size: median of dataset means {D.groupby('dataset')['size'].mean().median():.1f} (2dp: {e1[e1.method == 'MHMG'].groupby('dataset')['size'].mean().median():.1f})")
# ---- E
Ed = e8[e8.part == "E"]
old = e1[e1.dataset == "lymphoma_2classes"]
dm_old = old[old.method == "MHMG"].knn_acc.mean() - max(old[old.method == f"NRS_{e}"].knn_acc.mean() for e in ("0.1", "0.2", "0.3"))
dm_new = Ed[Ed.method == "MHMG-inF"].knn_acc.mean() - max(Ed[Ed.method == f"NRS_{e}-inF"].knn_acc.mean() for e in ("0.1", "0.2", "0.3"))
out.append(f"Lymphoma MHMG-NRS*: old {dm_old:+.3f} in-fold {dm_new:+.3f} change {dm_new - dm_old:+.3f}")
# Holm over H13-H16 family (H13, H14, H15 x3, H16)
fam = ["H13", "H14", "H15_0.1", "H15_0.2", "H15_0.3", "H16"]
o = sorted(fam, key=lambda k: R[k]["p"]); run = 0
for i, k in enumerate(o):
    run = max(run, min(1, (len(o) - i) * R[k]["p"])); R[k]["ph"] = run
for k, r in R.items():
    out.append(f"{k}: n={r['n']} mean={r['mean']:+.3f} CI[{r['lo']:+.3f},{r['hi']:+.3f}] W/T/L={r['w']}/{r['t']}/{r['l']} p={r['p']:.4f}"
               + (f" pHolm={r['ph']:.4f}" if "ph" in r else "") + (" HELD" if k in fam and r["lo"] > 0 else (" NOT-HELD" if k in fam else "")))
out.append(f"NRS*8 mean 3-NN {nstar8.mean():.3f} SVM {nstar8s.mean():.3f}; NRS-cv8 {ncv8.mean():.3f}; MHMG {mh.mean():.3f}")
(H / "analysis.txt").write_text("\n".join(out) + "\n"); print("\n".join(out))
