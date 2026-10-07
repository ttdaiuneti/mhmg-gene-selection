# Datasets

The experiments use 17 gene-expression datasets drawn from two public sources.

## OpenML (downloaded automatically by `src/data.py`)

| Dataset | OpenML ID |
|---|---|
| lymphoma_2classes | 1101 |
| leukemia | 1104 |
| BurkittLymphoma | 1084 |
| GCM | 1106 |
| tumors_C | 1107 |
| rsctc2010_1 | 1077 |
| rsctc2010_3 | 1079 |
| AP_Omentum_Uterus | 1124 |
| colon-cancer | 1432 |
| duke-breast-cancer | 1434 |

`src/data.py` calls `sklearn.datasets.fetch_openml` with the IDs above and caches
the result as `.npz` files under `feas/data/` on first run.

## ASU Feature Selection Repository

The remaining 7 datasets (`CLL_SUB_111`, `GLIOMA`, `GLI_85`, `Prostate_GE`,
`SMK_CAN_187`, `TOX_171`, `lung`) are available in MATLAB format from the
[ASU Feature Selection Repository](https://jundongl.github.io/scikit-feature/datasets.html).

Download the `.mat` files and place them in a folder named `data_asu/` at the
repository root (same level as `src/`). `src/data.py` reads them with `scipy.io.loadmat`.
