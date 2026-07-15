#!/bin/bash
#SBATCH --job-name=evL
#SBATCH --output=slurm/evals/outputs/evL_%j.out
#SBATCH --error=slurm/evals/errors/evL_%j.err
#SBATCH --account=gts-nisha3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H200:1
#SBATCH --mem-per-cpu=48G
#SBATCH --qos=embers
#SBATCH --time=6:00:00
#
# Evaluate ONE manifest line (env LINE_IDX) = all checkpoints of one run on one split, into one CSV
# via the native-vLLM-LoRA fast path. RESUMES (skips done steps). NO self-resubmit -- the orchestrator
# (eval_orchestrator.sh) owns resubmission, so a preemption just means the next orchestrator pass picks
# this line up again and resume continues from the flushed CSV.
#   Env: MANIFEST, LINE_IDX, [COT_SAMPLES=8], [REASONING_MAX_TOKENS=1024], [ANSWER_MAX_TOKENS=1024]
set -uo pipefail
module load python/3.12.5 cuda/12.9.1
source ~/r-nisha3-0/llm-env/bin/activate
cd ~/scratch/SImpL

LINE="$(sed -n "$((LINE_IDX+1))p" "$MANIFEST")"
if [[ -z "$LINE" ]]; then echo "no manifest line $LINE_IDX"; exit 0; fi
IFS=$'\t' read -r TRAIN_DS RUN_PREFIX DATA_PATH OUTPUT_CSV <<<"$LINE"
RUN_DIR="$(ls -dt oat-output/${TRAIN_DS}/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)"
if [[ -z "$RUN_DIR" ]]; then echo "no run dir for ${RUN_PREFIX}" >&2; exit 1; fi
mkdir -p "$(dirname "$OUTPUT_CSV")"
echo "[evL $LINE_IDX] ds=$TRAIN_DS prefix=$RUN_PREFIX split=$DATA_PATH -> $OUTPUT_CSV  run=$RUN_DIR"

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
