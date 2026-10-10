#!/usr/bin/env bash
# Regenerates every table, figure and number macro of the article from the fold-level results in this folder.
# Needs the 17 datasets (see ../data/README.md): OpenML ones are downloaded and cached in feas/data on first use,
# the scikit-feature .mat files go to data_asu/. Run from this folder: bash reproduce_numbers.sh
set -euo pipefail
cd "$(dirname "$0")"
( cd e6rs && python3 check_saturation.py && python3 check_random_saturation.py )   # sat_numbers.tex, randsat_numbers.tex
( cd manuscript/checks && python3 check_ties_baserate.py )                          # ties_numbers.tex
( cd manuscript && for c in check_theorem1 check_prop2 check_prop2_random check_nrs_target; do python3 $c.py; done )   # numerical checks of the propositions
( cd manuscript && python3 gen_tables.py && python3 gen_cs.py && python3 gen_e8.py && python3 gen_rs.py \
    && python3 gen_size.py && python3 gen_extra.py && python3 make_figs.py )        # numbers, tables, figures
echo "done: see manuscript/*.tex, manuscript/tables, manuscript/figures"
