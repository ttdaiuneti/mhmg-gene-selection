"""Search and verify the worked example of Section III: a 6-sample table on which NRS (eps) saturates after one
attribute while the MHMG energy still increases when a second attribute is added."""
import itertools, numpy as np
EPS = 0.15
y = np.array([0, 0, 0, 1, 1, 1])
def radii(X, B):
    D = np.sqrt(((X[:, None, B] - X[None, :, B]) ** 2).sum(-1)).round(12)
    return np.array([D[i, y != y[i]].min() for i in range(len(y))]), D
def gamma(X, B):
    if not B: return 0.0, None, None
    d, D = radii(X, B)
    G = np.array([(D[i] < d[i]).sum() for i in range(len(y))])
    K = sum(np.log2((y == y[i]).sum() + 1) for i in range(len(y)))
    return sum(np.log2(G + 1)) / K, d, G
def gn(X, B):
    d, _ = radii(X, B); return (d > EPS).mean()
if __name__ == "__main__":
    vals = [0.0, 0.3, 0.4, 0.6, 0.7, 1.0]
    rng = np.random.default_rng(1); best = None
    for _ in range(200000):
        a1 = np.r_[0.0, 1.0, rng.choice(np.arange(1, 10) / 10, 4)]; rng.shuffle(a1); a2 = np.r_[0.0, 1.0, rng.choice(np.arange(1, 10) / 10, 4)]; rng.shuffle(a2)
        X = np.c_[a1, a2]
        if not (gn(X, [0]) == 1.0 and gn(X, [1]) < 1.0): continue
        g1, d1, G1 = gamma(X, [0]); g2, d2, G2 = gamma(X, [1]); g12, d12, G12 = gamma(X, [0, 1])
        if not (round(g1, 2) > round(g2, 2) and round(g12, 2) > round(g1, 2)): continue
        if (a1[:3] < a1[3:].min()).all() or (a1[:3] > a1[3:].max()).all(): continue  # a1 alone must not sort classes
        best = X; break
    X = best; print("X=\n", X)
    for B in ([0], [1], [0, 1]):
        g, d, G = gamma(X, B); print(B, "gammaN", gn(X, B), "radii", d.round(3), "|G|", G, "gamma", round(g, 4))


def verify():
    """Assert the values printed in Example 1 of the manuscript."""
    X = np.array([[0.3, 0.1], [0.7, 0.3], [0.2, 0.3], [0.9, 0.4], [1.0, 0.0], [0.0, 1.0]])
    g1, d1, G1 = gamma(X, [0]); g2, _, _ = gamma(X, [1]); g12, _, G12 = gamma(X, [0, 1])
    assert np.allclose(d1, [0.3, 0.2, 0.2, 0.2, 0.3, 0.2]) and gn(X, [0]) == 1.0 and gn(X, [0, 1]) == 1.0
    assert list(G1) == [2, 1, 2, 2, 2, 1] and list(G12) == [3, 1, 3, 1, 2, 1]
    assert (round(g1, 2), round(g2, 2), round(g12, 2)) == (0.69, 0.65, 0.72)
    print("Example 1 verified")


verify()
