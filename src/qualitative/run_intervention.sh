#!/bin/bash
# Prompt-intervention probes, to isolate whether visible reasoning helps each 8B model:
#   flatsimpl + reason  -> force the terse Understander to reason before answering
#   cot16     + direct  -> force the verbose Reasoner to answer with no reasoning
# Compare each against its own default-prompt baseline (evaluations/qualitative/<model>_<ds>_s<seed>).
#
#   bash src/qualitative/run_intervention.sh          # embers
#   INFERNO=1 bash src/qualitative/run_intervention.sh
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL

BASE=Qwen/Qwen3-8B-Base
OUTDIR=evaluations/qualitative
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi

declare -A COT_STEP=( [123]=204 [234]=252 [345]=292 )
declare -A FS_STEP=(  [123]=188 [234]=120 [345]=264 )
COT_PREFIX=Qwen3-8B-final-race50-cotn16-lr2e5-clip05-b02-t10-short_cot-only
FS_PREFIX=Qwen3-8B-final-race50-flatsimplv3-lr2e5-clip05-b02-t10-short_simpl-nb
declare -A DATA=( [race-c]=data/race-c/final_test.jsonl [quail]=data/quail/test_42_all.jsonl )

submit () {  # $1 model_tag  $2 seed  $3 step  $4 prefix  $5 dataset  $6 prompt_mode  $7 out_tag
  local mtag=$1 seed=$2 step=$3 prefix=$4 ds=$5 mode=$6 otag=$7
  local rundir; rundir=$(ls -d oat-output/race-c/${prefix}_${seed}_* 2>/dev/null | head -1)
  local step5; step5=$(printf "step_%05d" "$step")
  local ckpt="$rundir/saved_models/$step5"
  [[ -d "$ckpt" ]] || { echo "MISSING ckpt: $ckpt"; return; }
  local tag="${otag}_${ds}_s${seed}"
  local out="$OUTDIR/${tag}.jsonl"
  [[ -s "$out" ]] && { echo "skip (exists): $out"; return; }
  sbatch --job-name="qual_${tag}" \
    --account="$ACCOUNT" --qos="$QOS" --time=2:00:00 \
    --ntasks-per-node=1 --gpus=H200:1 --cpus-per-task=8 --mem-per-cpu=48G \
    --exclude="$EXCLUDE" \
    --output="slurm/qual/${tag}_%j.out" --error="slurm/qual/${tag}_%j.err" \
    --wrap="module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate; \
      export HF_HUB_DISABLE_XET=1; cd /storage/scratch1/1/agupta886/SImpL; \
      python -m src.qualitative.token_probe --checkpoint '$ckpt' --base_model '$BASE' \
        --dataset_name '$ds' --data_path '${DATA[$ds]}' --out '$out' --prompt_mode '$mode'"
  echo "submitted $tag  (mode=$mode step=$step)"
}

# MODE selects which intervention set to submit: "all" (default) = the original weak-reason +
# direct; "reason_first" = only the strong reason-first Understander re-run.
MODE_SEL="${MODE:-all}"
for ds in race-c quail; do
  for seed in 123 234 345; do
    if [[ "$MODE_SEL" == "all" ]]; then
      submit flatsimpl "$seed" "${FS_STEP[$seed]}"  "$FS_PREFIX"  "$ds" reason "flatsimpl-reason"
      submit cot16     "$seed" "${COT_STEP[$seed]}" "$COT_PREFIX" "$ds" direct "cot16-direct"
    fi
    if [[ "$MODE_SEL" == "reason_first" || "$MODE_SEL" == "all" ]]; then
      # Understander forced to reason FIRST (prefill prevents answer-at-token-0)
      submit flatsimpl "$seed" "${FS_STEP[$seed]}" "$FS_PREFIX" "$ds" reason_first "flatsimpl-reasonfirst"
    fi
  done
done
