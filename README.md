# Saturation of the Global Radius in Neighbourhood Rough-Set Attribute Reduction for High-Dimensional Small-Sample Data

Code and fold-level results for the manuscript

> Thi Thuy Tram Nguyen, Long Giang Nguyen, Manh Hung Nguyen, Thanh Dai Tran.
> *Saturation of the Global Radius in Neighbourhood Rough-Set Attribute Reduction for High-Dimensional Small-Sample Data.*

The study compares the maximal homogeneous metric granules criterion (MHMG, Nguyen et al., IJAR) with neighbourhood
rough-set models (NRS, FNRS, KNN-NRS, WKNRS, GBNRS, adaptive NRS) and with standard selectors on seventeen
gene-expression datasets (3 repetitions of stratified 5-fold cross-validation).

## Contents

| Folder | Content |
|---|---|
| `src/` | self-contained implementations of MHMG (`mhmg.py`), NRS and FNRS (`nrs_fnrs.py`) and a check that they reproduce the recorded reducts (`reproduce_reducts.py`) |
| `paper/e1`, `e3` ... `e8rev` | run scripts with all settings, the fold-level results (`raw_*.csv`, one row per dataset, repetition, fold and method; selection runs list the selected attributes) and the analysis scripts of each experiment |
| `paper/external/` | the re-implementations of KNN-NRS, WKNRS, GBNRS and adaptive NRS used by the run scripts (see its README) |
| `paper/manuscript/` | scripts that generate every table, figure and number macro of the article from the results, and the generated files |
| `data/README.md` | where the datasets come from |

## Reproducing the numbers of the article

```bash
pip install -r requirements.txt
# datasets: see data/README.md  (OpenML ones are downloaded and cached on first use)
python3 src/reproduce_reducts.py          # reducts of MHMG, NRS and FNRS against paper/e1/raw_results.csv
cd paper && bash reproduce_numbers.sh     # regenerates every table, figure and number macro from the results
```

## What has been verified, and what has not

Checked on the authors' machine (Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, SciPy 1.13.1, scikit-learn 1.6.1):

* `reproduce_numbers.sh` regenerates all number macros and tables used in the article from the `raw_*.csv` files, identical to the submitted versions.
* The selectors in `src/` give the recorded reducts in every fold checked (MHMG, NRS with radius 0.1 and 0.3, FNRS: 30 of 30 folds each on three datasets).
* The run scripts of `e6rs` (KNN-NRS, WKNRS, adaptive NRS, GBNRS) and `e8rev` reproduce the recorded size and reduct in every row checked (24 of 24 and 52 of 52, small datasets).
* `e1/run_e1.py` reproduces the recorded reducts and accuracies of every selector except L1-regularised logistic regression (88 rows checked on two small datasets; in two repeat runs, 3 and 4 of the 20 L1 rows differed from the recorded ones). The L1 solver is not deterministic, so the L1 sets of a new run can differ from the recorded ones; the article uses the recorded results.

Not re-run: the complete experiments (several hours on the largest datasets). The recorded results in the `raw_*.csv` files are the ones used in the article.

Scripts that read files of other studies (macros that the article does not use) skip those parts when the files are absent.
Paths to third-party code in the original working directory were replaced by `external/`.
