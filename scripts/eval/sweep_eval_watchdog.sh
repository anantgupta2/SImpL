#!/bin/bash
#SBATCH --job-name=sweep_evwd
#SBATCH --output=slurm/evals/outputs/sweep_evwd_%j.out
#SBATCH --error=slurm/evals/errors/sweep_evwd_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=embers
#SBATCH --time=0:15:00
#
# INDEPENDENT preemption safety net for the 24 sweep dev-evals. Does NOT rely on any
# in-job SIGTERM trap (which can be missed on hard preemption). Every ~PERIOD_MIN it:
#   for each config whose TRAINING has finished (train jobid no longer in queue),
#   if its dev CSV does not yet cover all saved checkpoints AND no eval job is currently
#   queued/running for it (job-name evsw_<name>), resubmit a fresh (non-dependency)
#   auto_curve_eval that RESUMES from the flushed CSV. Reschedules itself while any
#   dev eval is still incomplete; stops once all 24 CSVs are complete.
#   Env: [JOBIDS=experiments/sweep_2026-07-02_jobids.txt], [PERIOD_MIN=40]
set -uo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL

JOBIDS="${JOBIDS:-experiments/sweep_2026-07-02_jobids.txt}"
EXCL="atl1-1-03-018-14-0,atl1-1-03-020-11-0"
PERIOD_MIN="${PERIOD_MIN:-40}"

mapfile -t QNAMES < <(squeue -u "$USER" -h -o "%j")
mapfile -t QIDS   < <(squeue -u "$USER" -h -r -o "%A")
in_qname(){ local x; for x in "${QNAMES[@]:-}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }
in_qid(){   local x; for x in "${QIDS[@]:-}";   do [[ "$x" == "$1" ]] && return 0; done; return 1; }

remaining=0; submitted=0; done_cnt=0; training=0
while read -r jid method name; do
  [ -z "${jid:-}" ] && continue
  if [ "$method" = "cot16" ]; then suf="cot-only"; else suf="simpl-nb"; fi
  RUN_PREFIX="Qwen3-4B-sw-lsat50-${name}-t10_${suf}_42"
  OUT="evaluations/sweep_dev/${name}_s42.csv"

  # Still training? (train jobid present in queue) -> the dependency eval owns it; skip.
  if in_qid "$jid"; then training=$((training+1)); remaining=$((remaining+1)); continue; fi

  RUN_DIR="$(ls -dt oat-output/lsat-ar/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)"
  if [ -z "$RUN_DIR" ]; then echo "[wd] WARN train-done but no run dir: $name"; continue; fi
  nck=$(ls "$RUN_DIR/saved_models" 2>/dev/null | grep -c '^step' || true)
  [ "${nck:-0}" -eq 0 ] && { echo "[wd] WARN $name has 0 checkpoints"; continue; }

  ndone=0
  if [ -f "$OUT" ]; then
    ndone=$(python - "$OUT" <<'PY'
import sys, csv, re
s=set()
try:
    for r in csv.DictReader(open(sys.argv[1])):
        m=re.search(r'(\d+)', r.get('step',''))
        if m: s.add(int(m.group(1)))
except Exception: pass
print(len(s))
PY
)
  fi
  if [ "${ndone:-0}" -ge "$nck" ]; then done_cnt=$((done_cnt+1)); continue; fi

  # Incomplete. Count it; resubmit only if nothing is already queued/running for it.
  remaining=$((remaining+1))
  if in_qname "evsw_${name}"; then continue; fi
  ejid=$(sbatch --parsable --exclude=$EXCL --job-name="evsw_${name}" \
    --export=ALL,TRAIN_DS=lsat-ar,DATASET_NAME=lsat-ar,RUN_PREFIX="$RUN_PREFIX",\
DATA_PATH=data/lsat-ar/final_dev.jsonl,OUTPUT_CSV="$OUT",COT_SAMPLES=8,COT_EVAL_ONLY=1,EVAL_TIME=4:00:00 \
    scripts/eval/auto_curve_eval.sh 2>/dev/null) \
    && { echo "[wd] RESUBMIT $name -> $ejid (dev done=$ndone/$nck)"; submitted=$((submitted+1)); }
done < "$JOBIDS"

echo "[wd] complete=$done_cnt still_training=$training remaining_incomplete=$remaining resubmitted=$submitted"
if [ "$remaining" -gt 0 ]; then
  sbatch --begin="now+${PERIOD_MIN}minutes" --exclude=$EXCL \
    --export=ALL,JOBIDS="$JOBIDS",PERIOD_MIN="$PERIOD_MIN" \
    scripts/eval/sweep_eval_watchdog.sh >/dev/null 2>&1 && echo "[wd] rescheduled +${PERIOD_MIN}min"
else
  echo "[wd] ALL 24 dev evals COMPLETE -- stopping"
fi
