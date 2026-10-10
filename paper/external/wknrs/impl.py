"""Tai lap WKNRS (Weighted k-nearest Neighborhood Rough Set).

Nguon: Ning Wang, Enhui Zhao, "A new method for feature selection based on
weighted k-nearest neighborhood rough set", Expert Systems With Applications 238
(2024) 122324. Tai lap tu Dinh nghia 1-5 + cong thuc (7)-(18) + Thuat toan 1.
File PDF: kMHMG-Dynamic/ref-pdf/1-s2.0-S0957417423028269-main.pdf

CO CHE (da doc methods):
- Neighborhood = k lang gieng gan nhat TRON LOP (ct. 7).
- Sim(x, y_i) = 1/(1 + d(x,y_i))                                   (ct. 13)
- omega(y_i) = 1/(1 + std)                                          (ct. 13)
- SIM_k(x) = sum_{cung lop} omega*Sim - sum_{khac lop} omega*Sim    (Def 1, ct. 13)
- x thuoc positive region (lower approx) neu SIM_k(x) > alpha       (Def 2, ct. 14)
- gamma_B(D) = |KPOS_B(D)| / |U|                                    (ct. 17)
- Sig(a,B,D) = gamma_{B+a} - gamma_B; forward-greedy, dung khi Sig<=0 (Thuat toan 1)

QUYET DINH TAI LAP (nguong ambiguity trong bai, ghi ro de trung thuc):
1. std(x, y_i): bai viet mo ho ("standard deviation of y_i within its class").
   Doc chon: voi moi lop c co mat trong k-neighborhood, std_c = do lech chuan cua
   CAC KHOANG CACH tu x toi cac lang gieng lop c; moi lang gieng lop c nhan
   omega = 1/(1+std_c). Lop chi co 1 lang gieng -> std=0 -> omega=1.
2. alpha: bai noi "real number close to 0"; dung alpha = 0.01 (mac dinh, tham so hoa).
3. k: bai chi ra ket qua on dinh voi k=1..15; runner sweep k rieng.
Cac lua chon nay deu la dien giai; neu doi chieu nguoc so tu-bao-cao lech nhieu se
xem lai (1)/(2).
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler


def _pairwise(X):
    return cdist(X, X, metric="euclidean")


def _wknrs_pos_fraction(Xsub, y, k, alpha):
    """gamma_B(D) = ty le mau trong positive region tren tap thuoc tinh Xsub."""
    n = Xsub.shape[0]
    if Xsub.shape[1] == 0:
        return 0.0
    D = _pairwise(Xsub)
    np.fill_diagonal(D, np.inf)                 # loai chinh no khoi k-neighborhood
    kk = min(k, n - 1)
    in_pos = 0
    for i in range(n):
        nn = np.argpartition(D[i], kk - 1)[:kk]  # k lang gieng gan nhat (tron lop)
        d_nn = D[i][nn]
        y_nn = y[nn]
        sim = 1.0 / (1.0 + d_nn)
        # omega theo lop: std cua khoang cach trong tung lop co mat
        omega = np.empty(kk)
        for c in np.unique(y_nn):
            m = y_nn == c
            std_c = np.std(d_nn[m]) if m.sum() > 1 else 0.0
            omega[m] = 1.0 / (1.0 + std_c)
        same = y_nn == y[i]
        sim_k = np.sum(omega[same] * sim[same]) - np.sum(omega[~same] * sim[~same])
        if sim_k > alpha:
            in_pos += 1
    return in_pos / n


def make_wknrs_reduct_fn(k, alpha=0.01):
    """Factory: reduct_fn WKNRS (forward-greedy tren Sig, Thuat toan 1)."""
    def reduct_fn(X_train, y_train):
        scaler = MinMaxScaler().fit(X_train)
        Xs = scaler.transform(X_train)
        y = np.asarray(y_train)
        n_features = Xs.shape[1]

        reduct = []
        gamma_best = 0.0
        remaining = list(range(n_features))
        while remaining:
            best_sig, best_f, best_g = 0.0, None, gamma_best
            for f in remaining:
                g = _wknrs_pos_fraction(Xs[:, reduct + [f]], y, k, alpha)
                sig = g - gamma_best
                if sig > best_sig:
                    best_sig, best_f, best_g = sig, f, g
            if best_f is not None and best_sig > 0:   # Sig > 0 moi them (Thuat toan 1)
                reduct.append(best_f)
                remaining.remove(best_f)
                gamma_best = best_g
            else:
                break

        if not reduct:
            reduct = [0]
        selected = list(reduct)

        def transform_fn(X_test):
            return scaler.transform(X_test)[:, selected]

        return Xs[:, selected], transform_fn, selected

    return reduct_fn
