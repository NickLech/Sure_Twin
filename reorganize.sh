#!/usr/bin/env bash
# reorganize.sh - moves files into the new repository layout (history is preserved with git mv).
#
# Run it from the ROOT of your local clone, on a clean working tree:
#     git checkout -b restructure
#     bash reorganize.sh
#
# `datasets/` and `final_metrics/` are NOT touched.
# Python imports and default output paths are NOT updated here: that is the next step.
set -euo pipefail

if [ ! -d .git ] || [ ! -f train_STID.py ]; then
    echo "Run this script from the root of the Sure_Twin repository." >&2
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree not clean: commit or stash your changes first." >&2
    exit 1
fi

mv_() {  # mv_ <source> <destination>: skip files that are already gone
    if [ -e "$1" ]; then
        mkdir -p "$(dirname "$2")"
        git mv "$1" "$2"
    else
        echo "skip (not found): $1"
    fi
}

# ---------------------------------------------------------------- documentation
# The real project README becomes README.md; the install guide moves to docs/.
mv_ README.md                       docs/SETUP.md
mv_ "README with commands.md"       README.md
# docs/CLUSTER_GUIDE.md and docs/DATA_NOTES.md are already in place.

# ---------------------------------------------------------------- source code
mv_ convert_milan_to_basicts.py     src/data/convert_milan_to_basicts.py
mv_ make_split_view.py              src/data/make_split_view.py

mv_ train_STID.py                   src/training/train_STID.py
mv_ eval_split.py                   src/training/eval_split.py

mv_ analyse_result.py               src/analysis/analyse_result.py
mv_ conformal_eval.py               src/analysis/conformal_eval.py
mv_ schedule_conformal.py           src/analysis/schedule_conformal.py
mv_ schedule_final_metrics.py       src/analysis/schedule_final_metrics.py

mv_ plot_timeseries.py              src/plotting/plot_timeseries.py
mv_ plot_signed_error.py            src/plotting/plot_signed_error.py
mv_ plot_concentration.py           src/plotting/plot_concentration.py
mv_ plot_confident.py               src/plotting/plot_confident.py
mv_ busiest_levels.py               src/plotting/busiest_levels.py

# ---------------------------------------------------------------- shell scripts
mv_ setup.sh                        scripts/setup.sh
mv_ run_train.sh                    scripts/slurm/run_train.sh
mv_ run_eval.sh                     scripts/slurm/run_eval.sh
mv_ git_update.sh                   scripts/git_update.sh

# ---------------------------------------------------------------- results: reports and logs
mv_ results/RESULTS_ANALYSIS.md     results/reports/RESULTS_ANALYSIS.md
mv_ "results/output of analysis"    results/logs/analysis_2026-11-14.txt

# ---------------------------------------------------------------- results: figures by topic
# accuracy of the point forecast
mv_ results/analysis.png            results/figures/accuracy/analysis.png
mv_ results/timeseries.png          results/figures/accuracy/timeseries.png
mv_ results/signed_error.png        results/figures/accuracy/signed_error.png
# conformal upper bounds
mv_ busiest_levels.png              results/figures/conformal/busiest_levels.png
mv_ busiest_confident.png           results/figures/conformal/busiest_confident.png
mv_ confident_pred.png              results/figures/conformal/confident_pred.png
# time-of-day schedule
mv_ schedule_busiest.png            results/figures/schedule/schedule_busiest.png
# data exploration
mv_ concentration.png               results/figures/data/concentration.png

# folder for machine-readable outputs (margin tables, metrics); git needs a file to track it
mkdir -p results/tables
touch results/tables/.gitkeep
git add results/tables/.gitkeep

echo
echo "Done. Review with:  git status   and   git ls-files | grep -v -E '^(datasets|final_metrics)/'"
echo "Then commit:        git commit -m 'Reorganize repository structure'"
