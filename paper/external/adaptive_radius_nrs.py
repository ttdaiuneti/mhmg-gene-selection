import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances


class AdaptiveNRSReducer:
    """
    Adaptive-radius neighborhood rough set reducer (Def. 1-9, Qiu et al. 2025).

    Notes:
    - Implements adaptive radius mechanism and ADE-based forward greedy selection.
    - For very high-dimensional datasets, an optional prefilter can be enabled
      to keep runtime practical while preserving transparency.
    """

    def __init__(
        self,
        min_clip=0.1,
        max_clip=0.9,
        use_prefilter=True,
        prefilter_max_features=80,
        max_reduct_size=20,
    ):
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.use_prefilter = use_prefilter
        self.prefilter_max_features = prefilter_max_features
        self.max_reduct_size = max_reduct_size
        self.reduct_ = []
        self.ade_history_ = []
        self.full_ade_ = 0.0

    @staticmethod
    def _pairwise_dist(X_sub):
        X_sub = np.asarray(X_sub, dtype=np.float64)
        return pairwise_distances(X_sub, metric="euclidean")

    @staticmethod
    def compute_h_gap(sorted_dist_row):
        # sorted_dist_row excludes self, ascending
        n1 = sorted_dist_row.shape[0]
        if n1 <= 1:
            return 0.0
        dmin = sorted_dist_row[0]
        dmax = sorted_dist_row[-1]
        return 1.5 * (dmax - dmin) / max(n1, 1)

    def _distribution_radius(self, d_row_sorted, labels, yi):
        """Def.2 + Def.3 (returns min(delta_d, delta_d_max))."""
        # Def.2: first h-gap
        h_gap = self.compute_h_gap(d_row_sorted)
        intervals = np.diff(d_row_sorted)
        gap_idx = np.where(intervals > h_gap)[0]
        if gap_idx.size == 0:
            delta_d = d_row_sorted[-1] if d_row_sorted.size > 0 else 0.0
        else:
            # Eq.(12): distance to first gap location
            m_idx = gap_idx[0] + 1
            delta_d = d_row_sorted[m_idx]

        # Def.3: class-center based upper bound (weighted by inverse distance)
        # For numerical stability, use epsilon where needed.
        eps = 1e-12
        same_mask = labels == yi
        diff_mask = ~same_mask

        # Remove self point not present in d_row_sorted; centers are on full U coordinates.
        # Weighted centers are computed externally in _delta_d_max_for_i because require X_sub.
        return float(max(delta_d, 0.0)), float(max(h_gap, 0.0))

    def _delta_d_max_for_i(self, X_sub, y, i):
        """Def.3 Eq.(13-15)."""
        eps = 1e-12
        yi = y[i]
        xi = X_sub[i]

        same_idx = np.where(y == yi)[0]
        diff_idx = np.where(y != yi)[0]

        # fallbacks if one class missing
        if diff_idx.size == 0:
            return np.inf
        if same_idx.size <= 1:
            return np.inf

        # weighted center of same label (excluding itself)
        same_idx_wo = same_idx[same_idx != i]
        if same_idx_wo.size == 0:
            return np.inf

        ds = np.linalg.norm(X_sub[same_idx_wo] - xi, axis=1)
        ws = 1.0 / np.maximum(ds, eps)
        c_same = (ws[:, None] * X_sub[same_idx_wo]).sum(axis=0) / ws.sum()

        dd = np.linalg.norm(X_sub[diff_idx] - xi, axis=1)
        wd = 1.0 / np.maximum(dd, eps)
        c_diff = (wd[:, None] * X_sub[diff_idx]).sum(axis=0) / wd.sum()

        return 0.5 * np.linalg.norm(c_same - c_diff)

    @staticmethod
    def _entropy_from_neighbor_sizes(nei_sizes, n):
        # NE(B) = 1 - (1/|U|) * sum_i |N_i|/|U|
        return 1.0 - (1.0 / n) * np.sum(nei_sizes / n)

    def _compute_adaptive_radius(self, X_sub, y):
        """Def.1-5 computed for current feature subset B."""
        n = X_sub.shape[0]
        if X_sub.shape[1] == 0:
            return np.zeros(n), np.zeros(n), 0.0

        D = self._pairwise_dist(X_sub)

        delta_d = np.zeros(n, dtype=float)
        # For Def.2, sort distances excluding self
        for i in range(n):
            row = np.concatenate([D[i, :i], D[i, i + 1 :]])
            row.sort()
            d2, _ = self._distribution_radius(row, y, y[i])
            dmax = self._delta_d_max_for_i(X_sub, y, i)
            delta_d[i] = min(d2, dmax)

        # NE(B): neighborhoods under delta_d
        nei_mask_B = D <= delta_d[:, None]
        nei_sizes_B = nei_mask_B.sum(axis=1).astype(float)
        ne_B = self._entropy_from_neighbor_sizes(nei_sizes_B, n)

        # NE(D union B): append label as one extra numeric dimension
        y_num, _ = self._encode_labels(y)
        X_bd = np.column_stack([X_sub, y_num])
        D_bd = self._pairwise_dist(X_bd)
        nei_mask_BD = D_bd <= delta_d[:, None]
        nei_sizes_BD = nei_mask_BD.sum(axis=1).astype(float)
        ne_BD = self._entropy_from_neighbor_sizes(nei_sizes_BD, n)

        ne_cond = max(ne_BD - ne_B, 0.0)
        xi = 1.0 / (1.0 + ne_cond)
        xi = min(max(xi, self.min_clip), self.max_clip)

        delta_adap = xi * delta_d
        return delta_adap, delta_d, xi

    @staticmethod
    def _encode_labels(y):
        cls, inv = np.unique(y, return_inverse=True)
        return inv.astype(float), cls

    def _ade(self, X_sub, y):
        """Def.6-9: ADE(D,U) = |POS|/|U|."""
        n = X_sub.shape[0]
        if X_sub.shape[1] == 0:
            return 0.0

        delta_adap, _, _ = self._compute_adaptive_radius(X_sub, y)
        D = self._pairwise_dist(X_sub)
        classes = np.unique(y)

        # lower approximation per class: N(i) subset class
        pos = np.zeros(n, dtype=bool)
        for c in classes:
            class_mask = y == c
            # neighborhood mask for each i
            nei_mask = D <= delta_adap[:, None]
            # subset test: all neighbors belong to class c
            # equivalent: no neighbor outside class
            bad = np.any(nei_mask & (~class_mask)[None, :], axis=1)
            lower_c = ~bad
            pos |= lower_c

        return float(pos.mean())

    def fit_reduction(self, X_raw, y, stop_tol=1e-12):
        """Forward greedy on ADE (Def.9)."""
        X = MinMaxScaler().fit_transform(X_raw)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float64)
        n, p = X.shape

        feature_pool = np.arange(p)
        # practical prefilter for high-dimensional data
        if self.use_prefilter and p > self.prefilter_max_features:
            # keep top-variance attributes to bound runtime
            var = X.var(axis=0)
            keep = np.argsort(var)[::-1][: self.prefilter_max_features]
            feature_pool = np.sort(keep)

        # target dependency on candidate full set
        self.full_ade_ = self._ade(X[:, feature_pool], y)

        reduct = []
        remaining = list(feature_pool)
        best_ade = 0.0
        self.ade_history_ = []

        while remaining:
            if len(reduct) >= self.max_reduct_size:
                break
            best_f = None
            best_local = best_ade

            for f in remaining:
                cand = reduct + [f]
                ade = self._ade(X[:, cand], y)
                if ade > best_local + stop_tol:
                    best_local = ade
                    best_f = f

            if best_f is None:
                break

            reduct.append(best_f)
            remaining.remove(best_f)
            best_ade = best_local
            self.ade_history_.append(best_ade)

            if best_ade >= self.full_ade_ - stop_tol:
                break

        self.reduct_ = reduct
        return reduct, best_ade

    def fit(self, X, y):
        self.fit_reduction(X, y)
        return self

    def transform(self, X):
        if not self.reduct_:
            return np.empty((X.shape[0], 0))
        return X[:, self.reduct_]
