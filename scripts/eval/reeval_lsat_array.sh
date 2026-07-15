#!/bin/bash
# Re-evaluate EVERY trained LSAT run on the pooled valid+test joint set (461 Q) using
# a single throttled SLURM job array (respects the 50-job embers cap via %CONCURRENCY).
# One array task == one run dir; each does a full-curve eval (resume-safe per ckpt) on
# the joint set -> evaluations/lsat-re-eval/<group>/<run>.csv.
#
# Usage:
#   bash scripts/eval/reeval_lsat_array.sh            # build manifest + submit array
#   bash scripts/eval/reeval_lsat_array.sh --dry      # build manifest + print, no submit
#
# Knobs (env): CONCURRENCY=6  TIME=4:00:00  COT_SAMPLES=8
set -euo pipefail
cd ~/scratch/SImpL
JOINT="data/lsat-ar/eval_joint_42_all.jsonl"
[[ -f "$JOINT" ]] || { echo "joint set missing: $JOINT" >&2; exit 1; }
OUTROOT="evaluations/lsat-re-eval"
MANIFEST="$OUTROOT/manifest.tsv"
CONCURRENCY="${CONCURRENCY:-6}"; TIME="${TIME:-4:00:00}"; COT_SAMPLES="${COT_SAMPLES:-8}"
mkdir -p "$OUTROOT"

classify () {
  case "$1" in
    lsat50-hp-*)                   echo hp_grid ;;
    lsat-simpl-base-*)             echo hp_simpl_base ;;
    lsat-cot-warmup-*)             echo hp_warmup ;;
    final-lsat*)                   echo final ;;
    lsat50-*|lsat100-*|lsat200-*)  echo hp_size ;;
    *)                             echo other ;;
  esac
}

# Build manifest: one line per run dir w/ checkpoints -> "<run_dir>\t<group>\t<out_csv>"
: > "$MANIFEST"
for d in oat-output/lsat-ar/*/; do
  ls "$d/saved_models" 2>/dev/null | grep -q '^step_' || continue
  bn="$(basename "$d")"
  short="$(echo "${bn#Qwen3-4B-Base-}" | sed -E 's/_[0-9]{4}T[0-9:]+$//')"
  grp="$(classify "$short")"
  printf '%s\t%s\t%s\n' "${d%/}" "$grp" "$OUTROOT/$grp/${short}.csv" >> "$MANIFEST"
done
N=$(wc -l < "$MANIFEST")
echo "manifest: $N runs -> $MANIFEST  (groups: $(cut -f2 "$MANIFEST" | sort | uniq -c | tr '\n' ' '))"
[[ "$N" -eq 0 ]] && { echo "nothing to eval" >&2; exit 1; }
mkdir -p "$OUTROOT"/{hp_grid,hp_size,hp_simpl_base,hp_warmup,final}

if [[ "${1:-}" == "--dry" ]]; then echo "[dry] would submit array 0-$((N-1))%$CONCURRENCY"; column -t "$MANIFEST"; exit 0; fi

sbatch --array=0-$((N-1))%"$CONCURRENCY" \
  --job-name=lsat_reeval --account=gts-nisha3 --qos=embers --time="$TIME" \
  --ntasks-per-node=1 --cpus-per-task=8 --gpus=H200:1 --mem-per-cpu=64G \
  --output=slurm/evals/outputs/lsat_reeval_%A_%a.out \
  --error=slurm/evals/errors/lsat_reeval_%A_%a.err \
  --export=ALL,MANIFEST="$MANIFEST",JOINT="$JOINT",COT_SAMPLES="$COT_SAMPLES" \
  --wrap='
set -euo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
RUN_DIR=$(cut -f1 <<<"$line"); OUT=$(cut -f3 <<<"$line")
echo "[task $SLURM_ARRAY_TASK_ID] $RUN_DIR -> $OUT"
python -m src.eval_saved_models \
  --checkpoint_root "$RUN_DIR" --data_path "$JOINT" \
  --cot_samples "$COT_SAMPLES" --reasoning_max_tokens 1024 --answer_max_tokens 1024 \
  --tensor_parallel_size 1 --gpu_memory_utilization 0.95 --batch_size 128 \
  --output_csv "$OUT"
'
echo "submitted throttled array (0-$((N-1))%$CONCURRENCY) -> $OUTROOT/"