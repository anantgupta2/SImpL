#!/bin/bash
# Token/qualitative probe for the 8B u4c12 (25% understanding) Understander on RACE-C, in the
# default and reason_first prompt modes. Checks whether the lower-understanding-share arm behaves
# like flatsimpl (8:8) -- i.e. also answers directly -- or differently.
#
#   INFERNO=1 bash src/qualitative/run_token_probe_u4c12.sh
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL

BASE=Qwen/Qwen3-8B-Base
OUTDIR=evaluations/qualitative
DATA=data/race-c/final_test.jsonl
DSN=race-c
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi

declare -A U_STEP=( [123]=152 [234]=228 [345]=116 )
U_PREFIX=Qwen3-8B-final-race50-flatsplitv3-u4c12-lr2e5-clip05-b02-t10-short_simpl-split

submit () {  # $1 seed $2 step $3 prompt_mode $4 out_tag
  local seed=$1 step=$2 mode=$3 otag=$4
  local rundir; rundir=$(ls -d oat-output/race-c/${U_PREFIX}_${seed}_* 2>/dev/null | head -1)
  local step5; step5=$(printf "step_%05d" "$step")
  local ckpt="$rundir/saved_models/$step5"
  [[ -d "$ckpt" ]] || { echo "MISSING ckpt: $ckpt"; return; }
  local tag="${otag}_race-c_s${seed}"
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

for seed in 123 234 345; do
  submit "$seed" "${U_STEP[$seed]}" default      "u4c12"
  submit "$seed" "${U_STEP[$seed]}" reason_first "u4c12-reasonfirst"
done
