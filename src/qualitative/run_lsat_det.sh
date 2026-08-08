#!/bin/bash
# LSAT deterministic answering panel: the LSAT-trained Reasoner (cot16) vs Understander (flatsimpl),
# both on the plain-answer path (prompt_mode=default), greedy (n=1, temp 0), over LSAT-AR, 3 seeds.
# Unlike RACE, the LSAT Understander is expected to REASON here (its understanding only sets up the
# board), so we compare normal-vs-normal -- no direct/reason_after arms.
#   INFERNO=1 bash src/qualitative/run_lsat_det.sh
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL
OUTDIR=evaluations/qualitative_deterministic
DATA=data/lsat-ar/final_test.jsonl; DSN=lsat-ar; BASE=Qwen/Qwen3-8B-Base
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi
INFLIGHT=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

COT_PREFIX=Qwen3-8B-final-lsat50-cotn16-lr2e5-clip05-b02-t10_cot-only
FS_PREFIX=Qwen3-8B-final-lsat50-flatsimpl-lr2e5-clip05-b02-t10_simpl-nb
declare -A COT_STEP=( [123]=152 [234]=164 [345]=176 )
declare -A FS_STEP=(  [123]=316 [234]=240 [345]=292 )

submit () {  # $1 prefix $2 seed $3 step $4 out_tag
  local prefix=$1 seed=$2 step=$3 otag=$4
  local rundir; rundir=$(ls -d oat-output/lsat-ar/${prefix}_${seed}_* 2>/dev/null | head -1)
  [[ -n "$rundir" ]] || { echo "MISSING rundir ${prefix}_${seed}"; return; }
  local ckpt="$rundir/saved_models/$(printf "step_%05d" "$step")"
  [[ -d "$ckpt" ]] || { echo "MISSING ckpt $ckpt"; return; }
  local tag="${otag}_lsat-ar_s${seed}"; local out="$OUTDIR/${tag}.jsonl"
  [[ -s "$out" ]] && { echo "skip (exists): $out"; return; }
  grep -qx "qual_${tag}" <<< "$INFLIGHT" && { echo "skip (inflight): $tag"; return; }
  sbatch --job-name="qual_${tag}" --account="$ACCOUNT" --qos="$QOS" --time=2:00:00 \
    --ntasks-per-node=1 --gpus=H200:1 --cpus-per-task=8 --mem-per-cpu=48G --exclude="$EXCLUDE" \
    --output="slurm/qual/${tag}_%j.out" --error="slurm/qual/${tag}_%j.err" \
    --wrap="module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate; \
      export HF_HUB_DISABLE_XET=1; cd /storage/scratch1/1/agupta886/SImpL; \
      python -m src.qualitative.token_probe --checkpoint '$ckpt' --base_model '$BASE' \
        --dataset_name '$DSN' --data_path '$DATA' --out '$out' --prompt_mode default \
        --max_model_len 4096 --cot_samples 1 --cot_temperature 0.0" >/dev/null
  echo "submitted $tag (step $step)"
}

for seed in 123 234 345; do
  submit "$COT_PREFIX" "$seed" "${COT_STEP[$seed]}" "lsat-cot16"
  submit "$FS_PREFIX"  "$seed" "${FS_STEP[$seed]}"  "lsat-flatsimpl"
done