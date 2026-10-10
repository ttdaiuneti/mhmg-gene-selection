# Datasets

Seventeen public gene-expression datasets, loaded by `paper/e1/data.py`.

## OpenML (downloaded and cached automatically)

`lymphoma_2classes` 1101, `leukemia` 1104, `BurkittLymphoma` 1084, `GCM` 1106, `tumors_C` 1107,
`rsctc2010_1` 1077, `rsctc2010_3` 1079, `AP_Omentum_Uterus` 1124, `colon-cancer` 1432, `duke-breast-cancer` 1434
(OpenML data ids). They are fetched with `sklearn.datasets.fetch_openml` on first use and cached in `paper/feas/data/`.

## scikit-feature repository (manual download)

`CLL_SUB_111`, `GLIOMA`, `GLI_85`, `Prostate_GE`, `SMK_CAN_187`, `TOX_171`, `lung`, from
<https://jundongl.github.io/scikit-feature/datasets.html> in MATLAB format. Put the `.mat` files in `paper/data_asu/`.

Constant attributes are removed after loading; missing values (the lymphoma set) are replaced by attribute means.
