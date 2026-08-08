#!/bin/bash
#SBATCH --job-name=sweep_watcher
#SBATCH --output=slurm/evals/outputs/sweep_watcher_%j.out
#SBATCH --error=slurm/evals/errors/sweep_watcher_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#
# Autonomous safety-net: every INTERVAL seconds, run the dedup-safe, complete-skipping
# sweeper to resubmit any incomplete eval whose job died (e.g. hit walltime). Exits when
# every run is complete. Circuit breaker: if the eval queue (watcher excluded) ever
# exceeds PANIC_MAX, cancel all PENDING jobs and kill itself. Runs on the cluster -> no
# terminal/window needed.
#   sbatch scripts/eval/sweep_watcher.sh
set -uo pipefail
cd ~/scratch/SImpL
INTERVAL="${INTERVAL:-1800}"
PANIC_MAX="${PANIC_MAX:-50}"

njobs () { squeue -u "$USER" -h -o '%j' | grep -cv '^sweep_watcher' || true; }

panic_check () {  # cancel all pending + self if the eval queue blows past PANIC_MAX
  local n; n="$(njobs)"
  if [[ "$n" -gt "$PANIC_MAX" ]]; then
    echo "[watcher $(date '+%F %T')] PANIC: $n > $PANIC_MAX eval jobs -> cancel all PENDING + self"
    scancel -u "$USER" -t PENDING 2>/dev/null || true
    scancel "${SLURM_JOB_ID:-}" 2>/dev/null || true
    exit 1
  fi
}

count_incomplete () {  # echoes "<incomplete> <total>"
  python3 - <<'PY'
import csv,os,glob
runs=[]
for man in ("evaluations/lsat-re-eval/manifest.tsv","evaluations/race-re-eval/manifest.tsv"):
    if os.path.exists(man):
        for line in open(man):
            rd,out=line.rstrip("\n").split("\t"); runs.append((rd,out))
for s in (24,36,42):
    d=glob.glob(f"oat-output/lsat-ar/Qwen3-4B-Base-final-lsat100-simpl-pe2-us2_simpl-oat_{s}_*")
    if d: runs.append((d[0],f"evaluations/lsat-re-eval/final/final-lsat100-simpl-pe2-us2_simpl-oat_{s}.csv"))
inc=0
for rd,out in runs:
    sm=os.path.join(rd,"saved_models")
    nck=len([x for x in os.listdir(sm) if x.startswith("step_")]) if os.path.isdir(sm) else 0
    ne=(sum(1 for r in csv.DictReader(open(out)) if r.get("step","").startswith("step_")) if os.path.exists(out) else 0)
    if nck and ne<nck: inc+=1
print(inc, len(runs))
PY
}

while true; do
  panic_check
  echo "[watcher $(date '+%F %T')] sweeping..."
  bash scripts/eval/sweep_incomplete_evals.sh || true
  panic_check
  read -r INC TOT < <(count_incomplete)
  echo "[watcher $(date '+%F %T')] incomplete=$INC / $TOT"
  [[ "$INC" -eq 0 ]] && { echo "[watcher] all runs complete -> exiting"; break; }
  sleep "$INTERVAL"
done