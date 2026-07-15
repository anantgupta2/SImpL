#!/bin/bash
#SBATCH --job-name=devtest_eval
#SBATCH --output=slurm/evals/outputs/devtest_%A_%a.out
#SBATCH --error=slurm/evals/errors/devtest_%A_%a.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=48G
#SBATCH --qos=embers
#SBATCH --time=8:00:00
#SBATCH --signal=B:TERM@120
#
# Job-array curve eval driven by a manifest (one line per (run, split)). embers caps submitted jobs
# at 50/user, so instead of one task per line we use NTASKS (<=50) tasks, each processing a STRIDE of
# manifest lines (task i -> lines i, i+NTASKS, i+2*NTASKS, ...). Each line evaluates every checkpoint
# of one run on one data split (dev/test) into one CSV via the FAST native-vLLM-LoRA path (no merge),
# resuming (skips done steps). Self-heals on embers: on preempt/walltime or an incomplete CSV it
# resubmits ONLY its own array index (which re-walks its lines; resume makes that cheap).
#   Env: MANIFEST (tsv: TRAIN_DS \t RUN_PREFIX \t DATA_PATH \t OUTPUT_CSV), NTASKS,
#        [COT_SAMPLES=8], [REASONING_MAX_TOKENS=1024], [ANSWER_MAX_TOKENS=1024], [EVAL_TIME], [RESUB_MAX=30]
set -uo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL

SCRIPT="scripts/eval/array_eval.sh"
RESUB_COUNT="${RESUB_COUNT:-0}"
MANIFEST="${MANIFEST:?set MANIFEST}"
NTASKS="${NTASKS:?set NTASKS}"
IDX="${SLURM_ARRAY_TASK_ID:?must run as a job array}"
TOTAL=$(grep -c '' "$MANIFEST")

_resubmit() {
  local n=$(( RESUB_COUNT + 1 ))
  if [[ "$n" -gt "${RESUB_MAX:-30}" ]]; then echo "[devtest] resubmit cap ($n) hit"; return; fi
  echo "[devtest] resubmitting array index $IDX (attempt $n)"
  sbatch --account=gts-nisha3 --qos=embers --time="${EVAL_TIME:-8:00:00}" \
    --exclude=atl1-1-03-018-14-0,atl1-1-03-020-11-0 --array="${IDX}" \
    --export=ALL,RESUB_COUNT="$n",MANIFEST="$MANIFEST",NTASKS="$NTASKS",COT_SAMPLES="${COT_SAMPLES:-8}",REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-1024}",ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-1024}",EVAL_TIME="${EVAL_TIME:-8:00:00}",RESUB_MAX="${RESUB_MAX:-30}" \
    "$SCRIPT" 2>/dev/null || echo "[devtest] resubmit failed"
}
_on_term() { _resubmit; exit 0; }
trap _on_term TERM

incomplete_any=0
# Process every manifest line assigned to this task index (stride = NTASKS).
for (( LN=IDX; LN<TOTAL; LN+=NTASKS )); do
  LINE="$(sed -n "$((LN+1))p" "$MANIFEST")"
  [[ -z "$LINE" ]] && continue
  IFS=$'\t' read -r TRAIN_DS RUN_PREFIX DATA_PATH OUTPUT_CSV <<<"$LINE"
  echo "[devtest] === line $LN: ds=$TRAIN_DS prefix=$RUN_PREFIX split=$DATA_PATH out=$OUTPUT_CSV ==="
  RUN_DIR="$(ls -dt oat-output/${TRAIN_DS}/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)"
  if [[ -z "$RUN_DIR" ]]; then echo "[devtest] no run dir for ${RUN_PREFIX}; skipping"; continue; fi
  mkdir -p "$(dirname "$OUTPUT_CSV")"

  python -m src.eval_saved_models \
    --dataset_name "$TRAIN_DS" \
    --checkpoint_root "$RUN_DIR" \
    --data_path "$DATA_PATH" \
    --cot_samples "${COT_SAMPLES:-8}" \
    --reasoning_max_tokens "${REASONING_MAX_TOKENS:-1024}" \
    --answer_max_tokens "${ANSWER_MAX_TOKENS:-1024}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.95 \
    --batch_size 128 \
    --output_csv "$OUTPUT_CSV"

  REMAINING=$(python - "$RUN_DIR" "$OUTPUT_CSV" <<'PYEOF'
import sys, os, csv, re, glob
run_dir, csv_path = sys.argv[1], sys.argv[2]
ck=sorted(int(re.search(r'(\d+)',os.path.basename(p)).group(1)) for p in glob.glob(os.path.join(run_dir,'saved_models','step_*')))
done=set()
if os.path.exists(csv_path):
    for r in csv.DictReader(open(csv_path)):
        m=re.search(r'(\d+)', r.get('step',''))
        if m: done.add(int(m.group(1)))
print(len([s for s in ck if s not in done]))
PYEOF
)
  if [[ "${REMAINING:-0}" -gt 0 ]]; then
    echo "[devtest] line $LN incomplete ($REMAINING ckpts left)"; incomplete_any=1
  fi
done

# If any assigned line is still incomplete (e.g. a step errored / partial), resubmit this index.
if [[ "$incomplete_any" -gt 0 ]]; then
  echo "[devtest] some lines incomplete -> resubmitting index $IDX to finish (resume skips done steps)"
  _resubmit
fi
exit 0
