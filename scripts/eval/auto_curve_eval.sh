#!/bin/bash
#SBATCH --job-name=curve_eval_%j
#SBATCH --output=slurm/evals/outputs/curve_eval_%j.out
#SBATCH --error=slurm/evals/errors/curve_eval_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=64G
#SBATCH --qos=embers
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#
# Evaluate EVERY saved checkpoint of a run (avg@N) into ONE CSV (rows = steps) for
# the training curve. eval_saved_models FLUSHES the CSV after every checkpoint and
# resumes (skips done steps), so this is fully restartable. SELF-HEALING on embers:
# on EITHER preemption or walltime, SLURM delivers SIGTERM (embers grace + the
# --signal=B:TERM@120 below); the trap then SUBMITS A FRESH embers job that resumes from
# the flushed CSV. We resubmit (not `scontrol requeue`) because this cluster does NOT
# honor --requeue for preempted jobs -- a fresh job is the reliable path. Capped by
# RESUB_MAX to avoid pathological loops. A single submission thus grinds to completion
# across any number of preemptions, no external watcher needed.
#   Env: TRAIN_DS, RUN_PREFIX, DATASET_NAME, [DATA_PATH], OUTPUT_CSV, [COT_SAMPLES=8],
#        [EVAL_TIME=4:00:00], [RESUB_MAX=30]
set -euo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL

SCRIPT="scripts/eval/auto_curve_eval.sh"
RESUB_COUNT="${RESUB_COUNT:-0}"
# On SIGTERM (preempt or walltime), resubmit a fresh embers job that resumes from the
# flushed CSV, carrying every knob forward so the resume is identical.
_resubmit() {
  local n=$(( RESUB_COUNT + 1 ))
  if [[ "$n" -gt "${RESUB_MAX:-30}" ]]; then echo "[curve_eval] resubmit cap ($n) hit; not resubmitting"; return; fi
  echo "[curve_eval] SIGTERM (preempt/walltime) -> resubmitting embers job (attempt $n), resumes from $OUTPUT_CSV"
  sbatch --account=gts-nisha3 --qos=embers --time="${EVAL_TIME:-4:00:00}" \
    --job-name="${SLURM_JOB_NAME:-curve_eval}" \
    --export=ALL,RESUB_COUNT="$n",TRAIN_DS="${TRAIN_DS:-}",RUN_PREFIX="${RUN_PREFIX:-}",DATASET_NAME="${DATASET_NAME:-}",DATA_PATH="${DATA_PATH:-}",OUTPUT_CSV="${OUTPUT_CSV:-}",COT_SAMPLES="${COT_SAMPLES:-8}",LAST_N_STEPS="${LAST_N_STEPS:-0}",MIN_STEP="${MIN_STEP:-0}",MAX_STEP="${MAX_STEP:-0}",REDO_ALL="${REDO_ALL:-0}",REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-1024}",ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-1024}",EVAL_TIME="${EVAL_TIME:-4:00:00}",RESUB_MAX="${RESUB_MAX:-30}",COT_EVAL_ONLY="${COT_EVAL_ONLY:-0}" \
    "$SCRIPT" 2>/dev/null || echo "[curve_eval] resubmit sbatch failed"
}
_on_term() { _resubmit; exit 0; }
trap _on_term TERM

RUN_DIR="$(ls -dt oat-output/${TRAIN_DS}/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)"
if [[ -z "$RUN_DIR" ]]; then echo "curve_eval: no run dir for ${RUN_PREFIX}" >&2; exit 1; fi
echo "curve_eval: RUN_DIR=$RUN_DIR  steps: $(ls "$RUN_DIR/saved_models" 2>/dev/null | tr '\n' ' ')"

CMD=(
  python -m src.eval_saved_models
  --dataset_name "$DATASET_NAME"
  --checkpoint_root "$RUN_DIR"
  --cot_samples "${COT_SAMPLES:-8}"
  --reasoning_max_tokens "${REASONING_MAX_TOKENS:-1024}"
  --answer_max_tokens "${ANSWER_MAX_TOKENS:-1024}"
  --tensor_parallel_size 1
  --gpu_memory_utilization 0.95
  --batch_size 128
  --output_csv "$OUTPUT_CSV"
)
if [[ -n "${DATA_PATH:-}" ]]; then CMD+=(--data_path "$DATA_PATH"); fi
# Long-context transfer targets (LongBench-v2 up to 128k): YaRN + eager. Pass ROPE_YARN_FACTOR (a
# plain number, NOT JSON -- SLURM --export splits on commas). VLLM_ALLOW_LONG_MAX_MODEL_LEN lets
# vLLM accept a max_model_len above the base 32768 (safe: YaRN is actually active).
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then CMD+=(--max_model_len "$MAX_MODEL_LEN"); fi
if [[ -n "${ROPE_YARN_FACTOR:-}" ]]; then
  CMD+=(--rope_scaling "{\"rope_type\":\"yarn\",\"factor\":${ROPE_YARN_FACTOR},\"original_max_position_embeddings\":32768}")
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi
# Only score plain CoT (skip understanding gen + the two understanding modes) -- ~3-4x faster.
if [[ "${COT_EVAL_ONLY:-0}" == "1" ]]; then CMD+=(--cot_eval_only); fi
# Resume by default (skip checkpoints already in the CSV); set REDO_ALL=1 to re-eval all.
if [[ "${REDO_ALL:-0}" == "1" ]]; then CMD+=(--redo_all); fi
# Only the last N checkpoints (slow datasets, e.g. PW): set LAST_N_STEPS=N.
if [[ "${LAST_N_STEPS:-0}" -gt 0 ]]; then CMD+=(--last_n_steps "$LAST_N_STEPS"); fi
# Partition a dense curve: restrict to a step range (each chunk -> its own CSV).
if [[ "${MIN_STEP:-0}" -gt 0 ]]; then CMD+=(--min_step "$MIN_STEP"); fi
if [[ "${MAX_STEP:-0}" -gt 0 ]]; then CMD+=(--max_step "$MAX_STEP"); fi
# run in background + wait so the SIGTERM trap (walltime) can interrupt and requeue.
"${CMD[@]}" &
# `set -e` would abort on a non-zero wait before we can inspect it; guard with || .
wait $! && EXIT=0 || EXIT=$?

# Completeness check (belt-and-suspenders for the SIGTERM trap). On PREEMPTION, the
# signal can crash vLLM's IPC mid-checkpoint; the Python loop catches that as a normal
# per-step failure ("skipping this step"), writes a PARTIAL CSV, and exits 0 -- so the
# TERM trap never fires and the job dies with most steps unevaluated. So after wait,
# recount: if any saved checkpoint in range is still absent from the CSV, resubmit to
# finish (idempotent -- the resume skips steps already present).
REMAINING=$(python - "$RUN_DIR" "$OUTPUT_CSV" "${MIN_STEP:-0}" "${MAX_STEP:-0}" "${LAST_N_STEPS:-0}" <<'PYEOF'
import sys, os, csv, re, glob
run_dir, csv_path, min_s, max_s, last_n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ck = sorted(int(re.search(r'(\d+)', os.path.basename(p)).group(1))
            for p in glob.glob(os.path.join(run_dir, 'saved_models', 'step_*')))
if min_s > 0: ck = [s for s in ck if s >= min_s]
if max_s > 0: ck = [s for s in ck if s <= max_s]
if last_n > 0: ck = ck[-last_n:]
done = set()
if os.path.exists(csv_path):
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            m = re.search(r'(\d+)', r.get('step', ''))
            if m: done.add(int(m.group(1)))
print(len([s for s in ck if s not in done]))
PYEOF
)
if [[ "${REMAINING:-0}" -gt 0 ]]; then
  echo "[curve_eval] $REMAINING checkpoint(s) still unevaluated after exit=$EXIT (likely preemption swallowed by Python) -> resubmitting to finish"
  _resubmit
fi
exit "${EXIT:-0}"