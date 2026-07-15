#!/bin/bash
#SBATCH --job-name=eval_orch
#SBATCH --output=slurm/evals/outputs/eval_orch_%j.out
#SBATCH --error=slurm/evals/errors/eval_orch_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=embers
#SBATCH --time=0:20:00
#
# Central eval orchestrator (CPU job). Every hour it: (1) finds manifest lines whose CSV is still
# incomplete, (2) sees which line indices already have a worker queued/running (worker job-name = evL<idx>),
# (3) submits the missing incomplete ones up to MAXJOBS workers, (4) reschedules itself in 1h. This is
# robust to embers preemption -- a killed worker is simply resubmitted on the next pass (eval resumes
# from its flushed CSV). Stops rescheduling once everything is complete.
#   Env: MANIFEST, [MAXJOBS=44], [COT_SAMPLES=8], [REASONING_MAX_TOKENS=1024], [ANSWER_MAX_TOKENS=1024], [PERIOD_MIN=60]
set -uo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL

MANIFEST="${MANIFEST:?set MANIFEST}"
MAXJOBS="${MAXJOBS:-44}"
PERIOD_MIN="${PERIOD_MIN:-60}"
EXCL="atl1-1-03-018-14-0,atl1-1-03-020-11-0"

# line indices that already have a worker in the queue (job-name evL<idx>)
mapfile -t INQ < <(squeue -u "$USER" -h -o "%j" | grep -oP '^evL\K[0-9]+' | sort -un)
NWORK=$(squeue -u "$USER" -h -r -o "%j" | grep -c '^evL' || true)
is_queued() { local x; for x in "${INQ[@]:-}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

INCOMPLETE=$(python scripts/eval/incomplete_lines.py "$MANIFEST")
NREM=$(wc -w <<<"$INCOMPLETE")
echo "[orch] incomplete=$NREM workers_in_queue=$NWORK maxjobs=$MAXJOBS"

slots=$(( MAXJOBS - NWORK ))
sub=0
for idx in $INCOMPLETE; do
  (( slots <= 0 )) && break
  is_queued "$idx" && continue
  sbatch --exclude=$EXCL --job-name="evL${idx}" \
    --export=ALL,MANIFEST="$MANIFEST",LINE_IDX="$idx",COT_SAMPLES="${COT_SAMPLES:-8}",REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-1024}",ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-1024}" \
    scripts/eval/eval_worker.sh >/dev/null 2>&1 && { slots=$((slots-1)); sub=$((sub+1)); }
done
echo "[orch] submitted=$sub"

# Reschedule next pass while work remains.
if (( NREM > 0 )); then
  sbatch --begin="now+${PERIOD_MIN}minutes" --exclude=$EXCL \
    --export=ALL,MANIFEST="$MANIFEST",MAXJOBS="$MAXJOBS",COT_SAMPLES="${COT_SAMPLES:-8}",REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-1024}",ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-1024}",PERIOD_MIN="${PERIOD_MIN}" \
    scripts/eval/eval_orchestrator.sh >/dev/null 2>&1 && echo "[orch] rescheduled in ${PERIOD_MIN}min"
else
  echo "[orch] ALL COMPLETE -- not rescheduling"
fi
