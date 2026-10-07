# Overcoming Radius Saturation in Neighbourhood Rough-Set Attribute Reduction — Code and Results

Supplementary code and raw results for the paper:

> **Overcoming Radius Saturation in Neighbourhood Rough-Set Attribute Reduction
> for High-Dimensional, Small-Sample Data**  
> Thi Thuy Tram Nguyen, Long Giang Nguyen, Manh Hung Nguyen, Thanh Dai Tran  
> Manuscript submitted to *IEEE Access*

---

## Repository layout

```
src/
  mhmg.py            MHMG greedy reduct algorithm (P0)
  nrs_fnrs.py        NRS and FNRS vectorised implementations
  methods.py         All selectors used in the experiments
  data.py            Dataset loader (OpenML + ASU .mat files)

experiments/
  run_main.py        Main experiment (Table I datasets, all selectors)
  analyze_main.py    Analysis script for the main experiment
  run_supplement.py  Supplementary experiment (radius grid, equal-size, baselines)
  analyze_supplement.py  Analysis for the supplementary experiment

results/
  raw_main.csv       Per-fold results of the main experiment
  raw_supplement.csv Per-fold results of the supplementary experiment
  raw_rs.csv         Per-fold results of the radius-sensitivity experiment
  raw_size.csv       Per-fold results of the equal-size experiment

data/
  README.md          Dataset download instructions
```

---

## Reproducing the results

### 1. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn
```

### 2. Download datasets

See `data/README.md`.  
Place the ASU `.mat` files in `data_asu/` at the repo root.  
OpenML datasets are fetched and cached automatically on first run.

### 3. Run the main experiment

```bash
python experiments/run_main.py
# Output: results/raw_main.csv  (appends, resumable)
python experiments/analyze_main.py
```

### 4. Run the supplementary experiment

```bash
python experiments/run_supplement.py
# Output: results/raw_supplement.csv  (appends, resumable)
python experiments/analyze_supplement.py
```

The pre-computed results in `results/` reproduce all tables and figures
in the paper without re-running the experiments.

---

## Algorithm

`src/mhmg.py` implements MHMG (Maximal Homogeneous Metric Granules) greedy
attribute reduction as described in Section III of the paper. The core step:

```python
from src.mhmg import forward
reduct, path = forward(X_train, y_train)   # returns list of column indices
```

`src/nrs_fnrs.py` implements the NRS and FNRS baselines used for comparison.

---

## Licence

The code is released under the MIT Licence.  
The datasets retain their original licences (OpenML CC-BY 4.0; ASU repository).
