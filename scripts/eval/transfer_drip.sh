#!/bin/bash
#SBATCH --job-name=xfer_drip
#SBATCH --output=slurm/xfer_drip_%j.out
#SBATCH --error=slurm/xfer_drip_%j.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --qos=inferno
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#
# Drip-feed transfer evals under the embers ~50-job submit cap (QOSMaxSubmitJobPerUserLimit).
#
# Runs as a SLURM job on INFERNO, not a login-node `nohup` loop: the previous drips were login-node
# nohup and were killed silently, leaving 0-byte logs and 38 evals unsubmitted for a day. A batch
# job survives logout and is visible in squeue.
#
# It only submits into free embers capacity, and launch_transfers.py is idempotent + queue-aware
# (skips cells whose csv exists or whose job is already queued), so re-running is always safe.
# Exits once nothing is left to submit and nothing is still in flight.
#   Env: [CAP=45] [SLEEP=300] [ONLY_SIZES=a,b] [XFER_OUT=dir]
# ONLY_SIZES/XFER_OUT are forwarded to launch_transfers.py so a drip can target a
# specific capped variant writing to its own output dir.
set -uo pipefail
module load python/3.12.5
cd ~/scratch/SImpL

CAP="${CAP:-45}"
SLEEP="${SLEEP:-300}"

while true; do
  # Count only EMBERS jobs -- this drip runs on inferno and must not count against the cap.
  inflight=$(squeue -u "$USER" -h -o "%q %j" 2>/dev/null | awk '$1=="embers"' | wc -l)
  room=$(( CAP - inflight ))
  left=$(DRY_RUN=1 ONLY_SIZES="${ONLY_SIZES:-}" XFER_OUT="${XFER_OUT:-evaluations/transfer}" python scripts/claude/launch_transfers.py 2>/dev/null | head -1)
  echo "[drip $(date '+%F %T')] embers_inflight=$inflight room=$room | $left"

  if [[ "$left" == 0\ to\ submit* ]] && [[ "$inflight" -eq 0 ]]; then
    echo "[drip] nothing left to submit and nothing in flight -> done"; break
  fi

  if (( room > 0 )); then
    MAX_SUBMIT="$room" ONLY_SIZES="${ONLY_SIZES:-}" XFER_OUT="${XFER_OUT:-evaluations/transfer}" python scripts/claude/launch_transfers.py 2>&1 | grep -E "^\[OK|^\[FAIL|^\[skip|to submit" || true
  else
    echo "[drip] embers full ($inflight/$CAP); waiting"
  fi
  sleep "$SLEEP"
done
echo "[drip] finished at $(date '+%F %T')"
