#!/bin/zsh
# End-to-end Phase-3 pipeline, resume-safe against machine restarts.
# Committed to the repo (not scratchpad) so a restart can't wipe it; every
# step is idempotent/resumable via its own state file. Re-run this script
# after any restart and it picks up where it left off:
#
#   nohup zsh scripts/run_phase3_pipeline.sh > phase3_pipeline.log 2>&1 &
#
# Steps: finish GFS-phase (wet-bulb+wind) extraction -> top-up GEFS -> rebuild
# pairs (all Phase-3 trees) -> ablation. Each step is idempotent, so a crash
# mid-step loses at most that step's in-progress month.
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# Single-instance lock: refuse to start a second copy (relaunch-after-restart
# is safe, relaunch-while-alive would double the extraction load).
LOCK=.phase3_pipeline.lock
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
  echo "MARK $(date +%H:%M) ABORT: pipeline already running (pid $(cat $LOCK))"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# Retry a step a few times (transient S3/network resets) before giving up on
# it; extractions are idempotent so a retry just resumes. Non-fatal: a step
# that can't finish still lets later steps run on whatever data landed.
run_step() {
  local name="$1"; shift
  for attempt in 1 2 3; do
    echo "MARK $(date +%H:%M) $name (attempt $attempt)"
    if "$@"; then return 0; fi
    echo "MARK $(date +%H:%M) $name failed attempt $attempt, retrying"
    sleep 30
  done
  echo "MARK $(date +%H:%M) $name gave up after 3 attempts (continuing)"
  return 1
}

run_step "gfs_phase extraction"  $PY scripts/backfill_gfs_phase_zarr.py --start-month 2021-11
run_step "gefs top-up"           $PY scripts/backfill_gefs_zarr.py --start-month 2022-11
run_step "rebuild pairs"         $PY scripts/build_training_data.py
run_step "fold-A ablation"       $PY scripts/train_postprocessor.py --ablate

echo "MARK $(date +%H:%M) PHASE3 PIPELINE COMPLETE"
