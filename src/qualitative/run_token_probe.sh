#!/bin/bash
# Launch the token/qualitative probe for the 8B Reasoner (cot16) vs Understander (flatsimpl) on
# RACE-C and QuAIL. One GPU job per (model, seed, dataset). Deployed steps are the per-seed
# dev-argmax that the paper reports (uncapped).
#
#   bash src/qualitative/run_token_probe.sh            # submit the 12 baseline probes
#   EXTRA=1 bash src/qualitative/run_token_probe.sh    # ALSO submit the "think longer" cot16 arm
#
# Runs on embers (eval). NEVER run the python directly on the login node.
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL

BASE=Qwen/Qwen3-8B-Base
OUTDIR=evaluations/qualitative
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
# INFERNO=1 routes to the training cluster (gts-schava6-qcf/inferno) instead of embers.
if [[ "${INFERNO:-0}" == "1" ]]; then
  ACCOUNT=gts-schava6-qcf; QOS=inferno
else
  ACCOUNT=gts-nisha3; QOS=embers
fi

# run_dir prefix (seed + timestamp appended by a glob) and per-seed deployed step
declare -A COT_STEP=( [123]=204 [234]=252 [345]=292 )
declare -A FS_STEP=(  [123]=188 [234]=120 [345]=264 )
COT_PREFIX=Qwen3-8B-final-race50-cotn16-lr2e5-clip05-b02-t10-short_cot-only
FS_PREFIX=Qwen3-8B-final-race50-flatsimplv3-lr2e5-clip05-b02-t10-short_simpl-nb

# dataset -> test file
declare -A DATA=( [race-c]=data/race-c/final_test.jsonl [quail]=data/quail/test_42_all.jsonl )

submit () {  # $1 model_tag  $2 seed  $3 step  $4 prefix  $5 dataset  [$6 extra_tag]
  local mtag=$1 seed=$2 step=$3 prefix=$4 ds=$5 extra_tag=${6:-}
  local rundir; rundir=$(ls -d oat-output/race-c/${prefix}_${seed}_* 2>/dev/null | head -1)
  if [[ -z "$rundir" ]]; then echo "MISSING run dir: ${prefix}_${seed}"; return; fi
  local step5; step5=$(printf "step_%05d" "$step")
  local ckpt="$rundir/saved_models/$step5"
  if [[ ! -d "$ckpt" ]]; then echo "MISSING ckpt: $ckpt"; return; fi
  local tag="${mtag}${extra_tag}_${ds}_s${seed}"
  local out="$OUTDIR/${tag}.jsonl"
  if [[ -s "$out" ]]; then echo "skip (exists): $out"; return; fi

  local extra_arg=()
  if [[ -n "$extra_tag" ]]; then
    extra_arg=(--extra_instruction "First, in a few sentences, lay out the key inferences the passage forces that it does not state outright -- what must be true, the author's stance, the meaning of any pivotal word. Then reason step by step to the answer.")
  fi

  sbatch --job-name="qual_${tag}" \
    --account="$ACCOUNT" --qos="$QOS" --time=2:00:00 \
    --ntasks-per-node=1 --gpus=H200:1 --cpus-per-task=8 --mem-per-cpu=48G \
    --exclude="$EXCLUDE" \
    --output="slurm/qual/${tag}_%j.out" --error="slurm/qual/${tag}_%j.err" \
    --wrap="module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate; \
      export HF_HUB_DISABLE_XET=1; cd /storage/scratch1/1/agupta886/SImpL; \
      python -m src.qualitative.token_probe --checkpoint '$ckpt' --base_model '$BASE' \
        --dataset_name '$ds' --data_path '${DATA[$ds]}' --out '$out' ${extra_arg[@]+\"${extra_arg[@]}\"}"
  echo "submitted $tag  (step $step)"
}

for ds in race-c quail; do
  for seed in 123 234 345; do
    submit cot16     "$seed" "${COT_STEP[$seed]}" "$COT_PREFIX" "$ds"
    submit flatsimpl "$seed" "${FS_STEP[$seed]}"  "$FS_PREFIX"  "$ds"
    if [[ "${EXTRA:-0}" == "1" ]]; then
      # "think longer" intervention on the Reasoner only
      submit cot16 "$seed" "${COT_STEP[$seed]}" "$COT_PREFIX" "$ds" "-long"
    fi
  done
done
