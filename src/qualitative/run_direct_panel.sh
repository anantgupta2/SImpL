#!/bin/bash
# The prompt-optimized-baseline control: is the Understander still ahead once the Reasoner is
# allowed to answer directly? Runs THREE conditions through the SAME token_probe pipeline so the
# comparison has no cross-harness offset:
#     cot16     + default   (Reasoner as reported)
#     cot16     + direct    (Reasoner, prompt-optimized)   <- the control
#     flatsimpl + default   (Understander; its default IS direct)
# over the 6-target RC + long-context panel, 4B and 8B, 3 seeds. Idempotent (skips existing).
#
#   INFERNO=1 bash src/qualitative/run_direct_panel.sh
#   INFERNO=1 ONLY_SCALE=8b MAX_SUBMIT=30 bash src/qualitative/run_direct_panel.sh
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL
# PANEL_OUT/COT_SAMPLES/COT_TEMP let the same matrix run deterministically (greedy, n=1) into a
# separate directory: PANEL_OUT=evaluations/qualitative_deterministic COT_SAMPLES=1 COT_TEMP=0.0
OUTDIR="${PANEL_OUT:-evaluations/qualitative}"
SAMPLES="${COT_SAMPLES:-8}"
TEMP="${COT_TEMP:-0.6}"
JOBPFX="${JOBPFX:-qual}"
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi
MAX_SUBMIT="${MAX_SUBMIT:-999}"; N=0
# Queue-aware guard: the output file only appears when a job FINISHES, so without this a re-run
# resubmits every still-running cell as a duplicate. Snapshot the in-flight job names once.
INFLIGHT=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

# target -> dataset_name|data_path|max_model_len   (quality ~6.5k and lbsmall 10-30k need the
# native 32k window; 4096 would silently truncate the passage)
declare -A TGT=(
  [race-c]="race-c|data/race-c/final_test.jsonl|4096"
  [quail]="quail|data/quail/test_42_all.jsonl|4096"
  [cosmosqa]="cosmosqa|data/cosmosqa/test_42_all.jsonl|4096"
  [lsatrc]="lsat-rc|data/lsat-rc/test_42_all.jsonl|4096"
  [quality]="quality|data/quality/test_42_all.jsonl|32768"
  [lbsmall]="longbench-v2-small|data/longbench-v2-small/test_42_all.jsonl|32768"
)
# scale -> base model
declare -A BASEM=( [4b]=Qwen/Qwen3-4B-Base [8b]=Qwen/Qwen3-8B-Base )
# scale -> run-dir prefixes
declare -A COT_PREFIX=(
  [4b]=Qwen3-4B-sw-race50-cotn16-lr2e5-clip05-b02-t10_cot-only
  [8b]=Qwen3-8B-final-race50-cotn16-lr2e5-clip05-b02-t10-short_cot-only )
declare -A FS_PREFIX=(
  [4b]=Qwen3-4B-sw-race50-flatsimplv3-lr2e5-clip05-b02-t10_simpl-nb
  [8b]=Qwen3-8B-final-race50-flatsimplv3-lr2e5-clip05-b02-t10-short_simpl-nb )
# u4c12 (25%-understanding-share split Understander), race-50. Its "default" path answers directly
# like flatsimpl. Enable the default arm with INCLUDE_U4C12=1; the "answer-then-reason" arm (probe
# whether eliciting post-hoc reasoning changes the direct-answerer) with INCLUDE_U4C12_REASON=1.
declare -A U4_PREFIX=(
  [4b]=Qwen3-4B-sw-race50-flatsplitv3-u4c12-lr2e5-clip05-b02-t10_simpl-split
  [8b]=Qwen3-8B-final-race50-flatsplitv3-u4c12-lr2e5-clip05-b02-t10-short_simpl-split )
# scale,seed -> deployed (dev-argmax) step
declare -A COT_STEP=( [4b,123]=248 [4b,234]=64  [4b,345]=76  [8b,123]=204 [8b,234]=252 [8b,345]=292 )
declare -A FS_STEP=(  [4b,123]=212 [4b,234]=104 [4b,345]=160 [8b,123]=188 [8b,234]=120 [8b,345]=264 )
declare -A U4_STEP=(  [4b,123]=136 [4b,234]=64  [4b,345]=64  [8b,123]=152 [8b,234]=228 [8b,345]=116 )

submit () {  # $1 scale $2 seed $3 prefix $4 step $5 mode $6 out_tag $7 target
  local sc=$1 seed=$2 prefix=$3 step=$4 mode=$5 otag=$6 tgt=$7
  IFS='|' read -r dsn data mml <<< "${TGT[$tgt]}"
  local rundir; rundir=$(ls -d oat-output/race-c/${prefix}_${seed}_* 2>/dev/null | head -1)
  [[ -n "$rundir" ]] || { echo "MISSING rundir ${prefix}_${seed}"; return; }
  local ckpt="$rundir/saved_models/$(printf "step_%05d" "$step")"
  [[ -d "$ckpt" ]] || { echo "MISSING ckpt $ckpt"; return; }
  local tag="${otag}_${tgt}_s${seed}"; local out="$OUTDIR/${tag}.jsonl"
  [[ -s "$out" ]] && return
  grep -qx "${JOBPFX}_${tag}" <<< "$INFLIGHT" && return   # already queued/running
  (( N >= MAX_SUBMIT )) && { echo "[throttle] at MAX_SUBMIT=$MAX_SUBMIT"; return; }
  sbatch --job-name="${JOBPFX}_${tag}" --account="$ACCOUNT" --qos="$QOS" --time=3:00:00 \
    --ntasks-per-node=1 --gpus=H200:1 --cpus-per-task=8 --mem-per-cpu=48G --exclude="$EXCLUDE" \
    --output="slurm/qual/${tag}_%j.out" --error="slurm/qual/${tag}_%j.err" \
    --wrap="module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate; \
      export HF_HUB_DISABLE_XET=1; cd /storage/scratch1/1/agupta886/SImpL; \
      python -m src.qualitative.token_probe --checkpoint '$ckpt' --base_model '${BASEM[$sc]}' \
        --dataset_name '$dsn' --data_path '$data' --out '$out' --prompt_mode '$mode' \
        --max_model_len $mml --cot_samples $SAMPLES --cot_temperature $TEMP" >/dev/null
  N=$((N+1)); echo "submitted $tag ($sc mode=$mode step=$step ctx=$mml)"
}

SCALES="${ONLY_SCALE:-4b 8b}"
for sc in $SCALES; do
  # 8B race-c/quail default+direct already exist from the earlier probes; the skip handles them.
  sfx=""; [[ "$sc" == "4b" ]] && sfx="-4b"
  for tgt in race-c quail cosmosqa lsatrc quality lbsmall; do
    for seed in 123 234 345; do
      submit "$sc" "$seed" "${COT_PREFIX[$sc]}" "${COT_STEP[$sc,$seed]}" default      "cot16${sfx}"          "$tgt"
      submit "$sc" "$seed" "${COT_PREFIX[$sc]}" "${COT_STEP[$sc,$seed]}" direct       "cot16-direct${sfx}"   "$tgt"
      submit "$sc" "$seed" "${FS_PREFIX[$sc]}"  "${FS_STEP[$sc,$seed]}"  default      "flatsimpl${sfx}"      "$tgt"
      if [[ "${INCLUDE_U4C12:-0}" == "1" && -n "${U4_PREFIX[$sc]:-}" ]]; then
        submit "$sc" "$seed" "${U4_PREFIX[$sc]}" "${U4_STEP[$sc,$seed]}" default      "u4c12${sfx}"          "$tgt"
      fi
      if [[ "${INCLUDE_U4C12_REASON:-0}" == "1" && -n "${U4_PREFIX[$sc]:-}" ]]; then
        submit "$sc" "$seed" "${U4_PREFIX[$sc]}" "${U4_STEP[$sc,$seed]}" reason_after "u4c12-reasonafter${sfx}" "$tgt"
      fi
    done
  done
done
echo "submitted $N jobs"
