"""Check of the runs proposition (single attribute, distinct values):
   sum_x log2(2 + m_x) <= E({a}) <= sum_runs L log2(L + 1),
where runs are maximal blocks of equal labels in the sorted order, and m_x = min over the sides of x that contain an
enemy of the number of run-mates of x on that side. Also checks G(x) is contained in the run of x."""
import numpy as np
rng = np.random.default_rng(0); bad = tot = 0; tight_lo = tight_up = 0
for t in range(20000):
    n = int(rng.integers(3, 40)); K = int(rng.integers(2, 4)); x = rng.standard_normal(n) * rng.choice([0.1, 1, 10])
    y = rng.integers(0, K, n)
    if len(np.unique(y)) < 2: continue
    o = np.argsort(x); xs, ys = x[o], y[o]
    D = np.abs(xs[:, None] - xs[None, :]); en = ys[:, None] != ys[None, :]; r = np.where(en, D, np.inf).min(1)
    G = D < r[:, None]; E = np.log2(G.sum(1) + 1).sum()
    run = np.zeros(n, int); start = [0]
    for i in range(1, n):
        run[i] = run[i - 1] + (ys[i] != ys[i - 1])
    lens = np.bincount(run); up = sum(L * np.log2(L + 1) for L in lens)
    lo = 0.0; inside = True
    for i in range(n):
        idx = np.flatnonzero(run == run[i]); left = i - idx[0]; right = idx[-1] - i
        sides = []
        if idx[0] > 0: sides.append(left)
        if idx[-1] < n - 1: sides.append(right)
        lo += np.log2(2 + min(sides)); inside &= set(np.flatnonzero(G[i])) <= set(idx)
    tot += 1; bad += not (lo - 1e-9 <= E <= up + 1e-9 and inside)
    tight_lo += abs(E - lo) < 1e-9; tight_up += abs(E - up) < 1e-9
print(f"cases {tot}, violations {bad}, lower bound attained {tight_lo}, upper bound attained {tight_up}")

# within-class ties allowed (no tie between classes): weakened hypothesis
rng2 = np.random.default_rng(5); bad2 = tot2 = 0
for t in range(20000):
    n = int(rng2.integers(3, 40)); K = int(rng2.integers(2, 4)); y = rng2.integers(0, K, n)
    if len(np.unique(y)) < 2: continue
    x = np.round(rng2.standard_normal(n), 1)
    if any((x[i] == x[j]) and y[i] != y[j] for i in range(n) for j in range(i + 1, n)): continue
    o = np.lexsort((rng2.random(n), x)); xs, ys = x[o], y[o]
    D = np.abs(xs[:, None] - xs[None, :]); en = ys[:, None] != ys[None, :]; r = np.where(en, D, np.inf).min(1)
    E = np.log2((D < r[:, None]).sum(1) + 1).sum()
    run = np.concatenate([[0], np.cumsum(ys[1:] != ys[:-1])]); lens = np.bincount(run); up = (lens * np.log2(lens + 1)).sum(); lo = 0
    for i in range(n):
        idx = np.flatnonzero(run == run[i]); sd = ([i - idx[0]] if idx[0] > 0 else []) + ([idx[-1] - i] if idx[-1] < n - 1 else [])
        lo += np.log2(2 + min(sd))
    tot2 += 1; bad2 += not (lo - 1e-9 <= E <= up + 1e-9)
print("within-class ties:", tot2, "violations", bad2)
from pathlib import Path
Path(__file__).with_name("runs_numbers.tex").write_text(
    "\\newcommand{\\runsCases}{%s}\n\\newcommand{\\runsLowTight}{%s}\n\\newcommand{\\runsUpTight}{%s}\n\\newcommand{\\runsViol}{%d}\n\\newcommand{\\runsTieCases}{%s}\n\\newcommand{\\runsTieViol}{%d}\n"
    % (f"{tot:,}".replace(",", "\\,"), f"{tight_lo:,}".replace(",", "\\,"), f"{tight_up:,}".replace(",", "\\,"), bad, f"{tot2:,}".replace(",", "\\,"), bad2))
