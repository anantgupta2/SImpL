#!/bin/bash
# Token/qualitative probe on LSAT-AR (analytical-reasoning puzzles) for the 8B Reasoner (cot16) vs
# Understander (flatsimpl). LSAT is the reasoning-heavy counterpoint to RACE/QuAIL: does the
# Understander stay terse here too, or does it actually reason?
#
#   INFERNO=1 bash src/qualitative/run_token_probe_lsat.sh            # default-prompt baseline
#   MODE=reason_first INFERNO=1 bash src/qualitative/run_token_probe_lsat.sh   # + interventions
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL

BASE=Qwen/Qwen3-8B-Base
OUTDIR=evaluations/qualitative
DATA=data/lsat-ar/final_test.jsonl
DSN=lsat-ar
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi

declare -A COT_STEP=( [123]=152 [234]=164 [345]=176 )
declare -A FS_STEP=(  [123]=316 [234]=240 [345]=292 )
COT_PREFIX=Qwen3-8B-final-lsat50-cotn16-lr2e5-clip05-b02-t10_cot-only
FS_PREFIX=Qwen3-8B-final-lsat50-flatsimpl-lr2e5-clip05-b02-t10_simpl-nb

submit () {  # $1 model_tag $2 seed $3 step $4 prefix $5 prompt_mode $6 out_tag
  local mtag=$1 seed=$2 step=$3 prefix=$4 mode=$5 otag=$6
  local rundir; rundir=$(ls -d oat-output/lsat-ar/${prefix}_${seed}_* 2>/dev/null | head -1)
  local step5; step5=$(printf "step_%05d" "$step")
  local ckpt="$rundir/saved_models/$step5"
  [[ -d "$ckpt" ]] || { echo "MISSING ckpt: $ckpt"; return; }
  local tag="${otag}_lsat-ar_s${seed}"
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
        --dataset_name '$DSN' --data_path '$DATA' --out '$out' --prompt_mode '$mode'"
  echo "submitted $tag  (mode=$mode step=$step)"
}

MODE_SEL="${MODE:-baseline}"
for seed in 123 234 345; do
  if [[ "$MODE_SEL" == "baseline" || "$MODE_SEL" == "all" ]]; then
    submit cot16     "$seed" "${COT_STEP[$seed]}" "$COT_PREFIX" default "cot16"
    submit flatsimpl "$seed" "${FS_STEP[$seed]}"  "$FS_PREFIX"  default "flatsimpl"
  fi
  if [[ "$MODE_SEL" == "reason_first" || "$MODE_SEL" == "all" ]]; then
    submit flatsimpl "$seed" "${FS_STEP[$seed]}" "$FS_PREFIX" reason_first "flatsimpl-reasonfirst"
    submit cot16     "$seed" "${COT_STEP[$seed]}" "$COT_PREFIX" direct "cot16-direct"
  fi
done
