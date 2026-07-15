#!/bin/bash
#SBATCH --job-name=finals_evwd
#SBATCH --output=slurm/evals/outputs/finals_evwd_%j.out
#SBATCH --error=slurm/evals/errors/finals_evwd_%j.err
#SBATCH --account=gts-schava6-qcf
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=inferno
#SBATCH --time=0:15:00
#
# Independent preemption safety net for the finals evals (dev+test, LSAT+RACE). Reads a
# manifest (jobname \t ds \t RUN_PREFIX \t DATA_PATH \t OUTPUT_CSV \t train_jid). Every
# ~PERIOD_MIN: for each eval whose TRAINING finished (train_jid gone from queue), if its
# CSV does not cover all saved checkpoints AND no job named <jobname> is queued/running,
# resubmit a fresh (non-dependency) auto_curve_eval that resumes from the flushed CSV.
# Reschedules while any eval is incomplete; stops when all are done. Does not rely on any
# in-job SIGTERM trap.
#   Env: [MANIFEST=evaluations/finals_eval_manifest.tsv], [PERIOD_MIN=40]
set -uo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL

MANIFEST="${MANIFEST:-evaluations/finals_eval_manifest.tsv}"
EXCL="atl1-1-03-018-14-0,atl1-1-03-020-11-0"
PERIOD_MIN="${PERIOD_MIN:-40}"

mapfile -t QNAMES < <(squeue -u "$USER" -h -o "%j")
mapfile -t QIDS   < <(squeue -u "$USER" -h -r -o "%A")
in_qname(){ local x; for x in "${QNAMES[@]:-}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }
in_qid(){   local x; for x in "${QIDS[@]:-}";   do [[ "$x" == "$1" ]] && return 0; done; return 1; }

remaining=0; submitted=0; done_cnt=0; training=0
while IFS=$'\t' read -r JN ds RUN_PREFIX DATA OUT tj; do
  [ -z "${JN:-}" ] && continue
  # Eval checkpoints AS SOON AS they exist -- do not wait for training to finish. A run is
  # only "complete" once training has left the queue AND the CSV covers every checkpoint.
  active=0; in_qid "$tj" && active=1
  RUN_DIR="$(ls -dt oat-output/${ds}/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)"
  if [ -z "$RUN_DIR" ]; then remaining=$((remaining+1)); [ "$active" -eq 1 ] && training=$((training+1)); continue; fi
  nck=$(ls "$RUN_DIR/saved_models" 2>/dev/null | grep -c '^step' || true)
  [ "${nck:-0}" -eq 0 ] && { remaining=$((remaining+1)); [ "$active" -eq 1 ] && training=$((training+1)); continue; }
  ndone=0
  if [ -f "$OUT" ]; then ndone=$(python - "$OUT" <<'PY'
import sys, csv, re
s=set()
try:
    for r in csv.DictReader(open(sys.argv[1])):
        m=re.search(r'(\d+)', r.get('step',''))
        if m: s.add(int(m.group(1)))
except Exception: pass
print(len(s))
PY
) ; fi
  # Complete only when training is done AND all current checkpoints are in the CSV.
  if [ "$active" -eq 0 ] && [ "${ndone:-0}" -ge "$nck" ]; then done_cnt=$((done_cnt+1)); continue; fi
  remaining=$((remaining+1)); [ "$active" -eq 1 ] && training=$((training+1))
  # Queue an eval whenever there are un-evaluated checkpoints and none is already queued
  # (in_qname dedup => at most one eval per CSV at a time, so no concurrent writers).
  if [ "${ndone:-0}" -ge "$nck" ]; then continue; fi   # caught up for now; more ckpts coming
  if in_qname "$JN"; then continue; fi
  ej=$(sbatch --parsable --exclude=$EXCL --job-name="$JN" \
    --export=ALL,TRAIN_DS=$ds,DATASET_NAME=$ds,RUN_PREFIX="$RUN_PREFIX",\
DATA_PATH="$DATA",OUTPUT_CSV="$OUT",COT_SAMPLES=8,COT_EVAL_ONLY=1,EVAL_TIME=4:00:00 \
    scripts/eval/auto_curve_eval.sh 2>/dev/null) \
    && { echo "[fwd] RESUBMIT $JN -> $ej (done=$ndone/$nck)"; submitted=$((submitted+1)); }
done < "$MANIFEST"

echo "[fwd] complete=$done_cnt still_training=$training remaining_incomplete=$remaining resubmitted=$submitted"
if [ "$remaining" -gt 0 ]; then
  sbatch --begin="now+${PERIOD_MIN}minutes" --exclude=$EXCL \
    --export=ALL,MANIFEST="$MANIFEST",PERIOD_MIN="$PERIOD_MIN" \
    scripts/eval/finals_eval_watchdog.sh >/dev/null 2>&1 && echo "[fwd] rescheduled +${PERIOD_MIN}min"
else
  echo "[fwd] ALL finals evals COMPLETE -- stopping"
fi
