# external

Re-implementations used by the run scripts of the comparison within the neighbourhood rough-set family.
Each file is a single-purpose module without further dependencies besides NumPy, SciPy and scikit-learn.

| File | Content |
|---|---|
| `run_pilot.py` | reference implementation of the MHMG greedy search used by `e8rev` and `e1/check_rounding.py` |
| `wang_knn_nrs.py` | KNN-NRS (k nearest neighbours as the neighbourhood) |
| `wknrs/impl.py` | WKNRS (weighted k-nearest-neighbourhood rough sets) |
| `gbnrs.py` | GBNRS (granular-ball neighbourhood rough sets, purity 0.9) |
| `adaptive_radius_nrs.py` | adaptive NRS (data-driven radius), greedy search without the swarm optimiser |
| `ccar/nrs.py` | min-max scaling helper imported by `wang_knn_nrs.py` |
