#!/bin/bash
# General resume-safe re-eval job array. One array task == one run dir; each does a
# full-curve eval (resume-safe: per-ckpt flush, skips done steps) on DATA_PATH, into
# OUT_ROOT/<group>/<run>.csv. Throttled with %CONCURRENCY so it never trips the embers
# 50-job cap. Preempted tasks requeue and resume -> no fragile afterok merges.
#
# Required env: RUN_GLOB  DATA_PATH  OUT_ROOT  ARRAY_NAME
# Optional env: SKIP_RE (regex of run names to exclude), CONCURRENCY=6, TIME=4:00:00,
#               COT_SAMPLES=8
# Usage: RUN_GLOB=... DATA_PATH=... OUT_ROOT=... ARRAY_NAME=... bash scripts/eval/reeval_array.sh [--dry]
set -euo pipefail
cd ~/scratch/SImpL
: "${RUN_GLOB:?}" "${DATA_PATH:?}" "${OUT_ROOT:?}" "${ARRAY_NAME:?}"
[[ -f "$DATA_PATH" ]] || { echo "eval set missing: $DATA_PATH" >&2; exit 1; }
SKIP_RE="${SKIP_RE:-__NOSKIP__}"; CONCURRENCY="${CONCURRENCY:-15}"
TIME="${TIME:-4:00:00}"; COT_SAMPLES="${COT_SAMPLES:-8}"; RUNS_PER_TASK="${RUNS_PER_TASK:-1}"
MANIFEST="${MANIFEST:-$OUT_ROOT/manifest.tsv}"; mkdir -p "$OUT_ROOT"

classify () {
  case "$1" in
    lsat50-hp-*)                          echo hp_grid ;;
    lsat-simpl-base-*)                    echo hp_simpl_base ;;
    lsat-cot-warmup-*|race-cot-warmup-*)  echo hp_warmup ;;
    final-*)                              echo final ;;
    lsat50-*|lsat100-*|lsat200-*)         echo hp_size ;;
    *)                                    echo other ;;
  esac
}

: > "$MANIFEST"
for d in $RUN_GLOB; do
  ls "$d/saved_models" 2>/dev/null | grep -q '^step_' || continue
  bn="$(basename "$d")"
  short="$(echo "${bn#Qwen3-4B-Base-}" | sed -E 's/_[0-9]{4}T[0-9:]+$//')"
  grep -qE "$SKIP_RE" <<<"${short}_" && { echo "skip (active): $short"; continue; }
  grp="$(classify "$short")"
  printf '%s\t%s\n' "${d%/}" "$OUT_ROOT/$grp/${short}.csv" >> "$MANIFEST"
  mkdir -p "$OUT_ROOT/$grp"
done
N=$(wc -l < "$MANIFEST")
NT=$(( (N + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))   # number of array tasks
echo "[$ARRAY_NAME] manifest: $N runs, $RUNS_PER_TASK/task -> $NT tasks -> $MANIFEST"
[[ "$N" -eq 0 ]] && { echo "nothing to eval" >&2; exit 1; }
if [[ "${1:-}" == "--dry" ]]; then column -t "$MANIFEST"; echo "[dry] array 0-$((NT-1))%$CONCURRENCY ($RUNS_PER_TASK runs/task)"; exit 0; fi

mkdir -p slurm/evals/outputs slurm/evals/errors
sbatch --array=0-$((NT-1))%"$CONCURRENCY" --requeue \
  --job-name="$ARRAY_NAME" --account=gts-nisha3 --qos=embers --time="$TIME" \
  --ntasks-per-node=1 --cpus-per-task=8 --gpus=H200:1 --mem-per-cpu=64G \
  --output="slurm/evals/outputs/${ARRAY_NAME}_%A_%a.out" \
  --error="slurm/evals/errors/${ARRAY_NAME}_%A_%a.err" \
  --export=ALL,MANIFEST="$MANIFEST",DATA_PATH="$DATA_PATH",COT_SAMPLES="$COT_SAMPLES",RUNS_PER_TASK="$RUNS_PER_TASK" \
  --wrap='
set -euo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
lo=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
for ((k=0; k<RUNS_PER_TASK; k++)); do
  ln=$(( lo + k + 1 ))
  line=$(sed -n "${ln}p" "$MANIFEST"); [[ -z "$line" ]] && break
  RUN_DIR=$(cut -f1 <<<"$line"); OUT=$(cut -f2 <<<"$line")
  echo "[task $SLURM_ARRAY_TASK_ID.$k] $RUN_DIR -> $OUT"
  python -m src.eval_saved_models \
    --checkpoint_root "$RUN_DIR" --data_path "$DATA_PATH" \
    --cot_samples "$COT_SAMPLES" --reasoning_max_tokens 1024 --answer_max_tokens 1024 \
    --tensor_parallel_size 1 --gpu_memory_utilization 0.95 --batch_size 128 \
    --output_csv "$OUT"
done
'
echo "[$ARRAY_NAME] submitted array 0-$((NT-1))%$CONCURRENCY ($RUNS_PER_TASK runs/task) -> $OUT_ROOT/"