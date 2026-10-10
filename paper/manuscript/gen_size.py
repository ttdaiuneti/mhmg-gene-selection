"""Macros for the size-matched MHMG vs FNRS comparison (e7size/the study protocol) from e7size/raw_size.csv."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
H = Path(__file__).resolve().parent; d = pd.read_csv(H.parent / "e7size" / "raw_size.csv"); rng = np.random.default_rng(0); M = {}
W = {10: "Ten", 20: "Twenty", 50: "Fifty"}
def f(v): return f"{v:+.3f}".replace("-", "$-$")
nonin = 0
for q in (10, 20, 50):
    for c, k in (("knn_acc", "K"), ("svm_acc", "S")):
        w = d[d.q == q].groupby(["dataset", "method"])[c].mean().unstack(); diff = (w.MHMG - w.FNRS).values
        lo, hi = np.percentile(diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1), [2.5, 97.5])
        M.update({f"sz{k}{W[q]}M": f"{w.MHMG.mean():.3f}", f"sz{k}{W[q]}F": f"{w.FNRS.mean():.3f}", f"sz{k}{W[q]}D": f(diff.mean()),
                  f"sz{k}{W[q]}Lo": f(lo), f"sz{k}{W[q]}Hi": f(hi), f"sz{k}{W[q]}W": str(int((diff > 0).sum())), f"sz{k}{W[q]}L": str(int((diff < 0).sum()))})
        if k == "K": nonin += lo > -0.02
M["szNonInf"] = str(nonin)
(H / "size_numbers.tex").write_text("".join(f"\\newcommand{{\\{a}}}{{{b}}}\n" for a, b in M.items())); print(M)
