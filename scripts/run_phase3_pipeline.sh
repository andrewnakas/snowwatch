#!/bin/zsh
# End-to-end Phase-3 pipeline, resume-safe against machine restarts.
# Committed to the repo (not scratchpad) so a restart can't wipe it; every
# step is idempotent/resumable via its own state file. Re-run this script
# after any restart and it picks up where it left off.
#
#   nohup zsh scripts/run_phase3_pipeline.sh > phase3_pipeline.log 2>&1 &
#
# Steps: finish GFS-phase (wet-bulb+wind) extraction -> top-up GEFS/HRRR ->
# rebuild pairs (all Phase-3 trees) -> ablation -> final fold retrain.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "MARK $(date +%H:%M) gfs_phase extraction (resumable)"
$PY scripts/backfill_gfs_phase_zarr.py --start-month 2021-11 2>&1 | tail -2

echo "MARK $(date +%H:%M) gefs top-up (idempotent)"
$PY scripts/backfill_gefs_zarr.py --start-month 2022-11 2>&1 | tail -2

echo "MARK $(date +%H:%M) rebuild pairs with Phase-3 feature trees"
$PY scripts/build_training_data.py

echo "MARK $(date +%H:%M) fold-A ablation (base/+ens/+phase/+hrrr_native/all)"
$PY scripts/train_postprocessor.py --ablate

echo "MARK $(date +%H:%M) PHASE3 PIPELINE COMPLETE"
