#!/bin/bash
# 4B RACE token probe: v3 baseline for the Reasoner (cot16) and Understander (flatsimpl), so the
# incoming 4B v4 model has a same-scale comparison. Default prompt only.
#   INFERNO=1 bash src/qualitative/run_token_probe_4b.sh
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL
BASE=Qwen/Qwen3-4B-Base
OUTDIR=evaluations/qualitative
DATA=data/race-c/final_test.jsonl
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi

declare -A COT_STEP=( [123]=248 [234]=64 [345]=76 )
declare -A FS_STEP=(  [123]=212 [234]=104 [345]=160 )
COT_PREFIX=Qwen3-4B-sw-race50-cotn16-lr2e5-clip05-b02-t10_cot-only
FS_PREFIX=Qwen3-4B-sw-race50-flatsimplv3-lr2e5-clip05-b02-t10_simpl-nb

submit () {  # $1 out_tag $2 seed $3 step $4 prefix
  local otag=$1 seed=$2 step=$3 prefix=$4
  local rundir; rundir=$(ls -d oat-output/race-c/${prefix}_${seed}_* 2>/dev/null | head -1)
  local step5; step5=$(printf "step_%05d" "$step")
  local ckpt="$rundir/saved_models/$step5"
  [[ -d "$ckpt" ]] || { echo "MISSING ckpt: $ckpt"; return; }
  local tag="${otag}_race-c_s${seed}"; local out="$OUTDIR/${tag}.jsonl"
  [[ -s "$out" ]] && { echo "skip (exists): $out"; return; }
  sbatch --job-name="qual_${tag}" --account="$ACCOUNT" --qos="$QOS" --time=2:00:00 \
    --ntasks-per-node=1 --gpus=H200:1 --cpus-per-task=8 --mem-per-cpu=48G --exclude="$EXCLUDE" \
    --output="slurm/qual/${tag}_%j.out" --error="slurm/qual/${tag}_%j.err" \
    --wrap="module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate; \
      export HF_HUB_DISABLE_XET=1; cd /storage/scratch1/1/agupta886/SImpL; \
      python -m src.qualitative.token_probe --checkpoint '$ckpt' --base_model '$BASE' \
        --dataset_name race-c --data_path '$DATA' --out '$out' --prompt_mode default"
  echo "submitted $tag (step $step)"
}
for seed in 123 234 345; do
  submit cot16-4b     "$seed" "${COT_STEP[$seed]}" "$COT_PREFIX"
  submit flatsimpl-4b "$seed" "${FS_STEP[$seed]}"  "$FS_PREFIX"
done
