#!/bin/bash
# Sample the UNDERSTANDINGS from the 8B RACE and LSAT understanders (deployed checkpoints, seed 123).
# One seed, capped passages -- this is for reading, not statistics.
#   INFERNO=1 bash src/qualitative/run_understanding_probe.sh
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL
OUTDIR=evaluations/qualitative_understandings
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi
MAXP="${MAX_PASSAGES:-60}"

# tag | base | dataset | data_path | run_dir_glob | step
RUNS=(
  "race-8b|Qwen/Qwen3-8B-Base|race-c|data/race-c/final_test.jsonl|oat-output/race-c/Qwen3-8B-final-race50-flatsimplv3-lr2e5-clip05-b02-t10-short_simpl-nb_123_*|188"
  "lsat-8b|Qwen/Qwen3-8B-Base|lsat-ar|data/lsat-ar/final_test.jsonl|oat-output/lsat-ar/Qwen3-8B-final-lsat50-flatsimpl-lr2e5-clip05-b02-t10_simpl-nb_123_*|316"
  # CONTROL: the cot-only Reasoner (never trained on the understanding objective) given the SAME
  # understanding prompt -- isolates what the understanding TRAINING added vs the base capability.
  "race-cot16-8b|Qwen/Qwen3-8B-Base|race-c|data/race-c/final_test.jsonl|oat-output/race-c/Qwen3-8B-final-race50-cotn16-lr2e5-clip05-b02-t10-short_cot-only_123_*|204"
  # u4c12: the 25%-understanding-share split Understander (vs the 8:8 flatsimpl race-8b above), same
  # RACE-C passages -- lets us read whether a lower understanding share yields a different artifact.
  "u4c12-8b|Qwen/Qwen3-8B-Base|race-c|data/race-c/final_test.jsonl|oat-output/race-c/Qwen3-8B-final-race50-flatsplitv3-u4c12-lr2e5-clip05-b02-t10-short_simpl-split_123_*|152"
  "u4c12-4b|Qwen/Qwen3-4B-Base|race-c|data/race-c/final_test.jsonl|oat-output/race-c/Qwen3-4B-sw-race50-flatsplitv3-u4c12-lr2e5-clip05-b02-t10_simpl-split_123_*|136"
  # 4B RACE understander (flatsimplv3 8:8) + Reasoner (cot16), for the 4B understanding-token table.
  "race-flatsimpl-4b|Qwen/Qwen3-4B-Base|race-c|data/race-c/final_test.jsonl|oat-output/race-c/Qwen3-4B-sw-race50-flatsimplv3-lr2e5-clip05-b02-t10_simpl-nb_123_*|212"
  "race-cot16-4b|Qwen/Qwen3-4B-Base|race-c|data/race-c/final_test.jsonl|oat-output/race-c/Qwen3-4B-sw-race50-cotn16-lr2e5-clip05-b02-t10_cot-only_123_*|248"
)
for spec in "${RUNS[@]}"; do
  IFS='|' read -r tag base dsn data glob step <<< "$spec"
  rundir=$(ls -d $glob 2>/dev/null | head -1)
  ckpt="$rundir/saved_models/$(printf 'step_%05d' "$step")"
  [[ -d "$ckpt" ]] || { echo "MISSING ckpt: $ckpt"; continue; }
  out="$OUTDIR/${tag}_s123.jsonl"
  [[ -s "$out" ]] && { echo "skip (exists): $out"; continue; }
  sbatch --job-name="uprobe_${tag}" --account="$ACCOUNT" --qos="$QOS" --time=1:00:00 \
    --ntasks-per-node=1 --gpus=H200:1 --cpus-per-task=8 --mem-per-cpu=48G --exclude="$EXCLUDE" \
    --output="slurm/qual/uprobe_${tag}_%j.out" --error="slurm/qual/uprobe_${tag}_%j.err" \
    --wrap="module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate; \
      export HF_HUB_DISABLE_XET=1; cd /storage/scratch1/1/agupta886/SImpL; \
      python -m src.qualitative.understanding_probe --checkpoint '$ckpt' --base_model '$base' \
        --dataset_name '$dsn' --data_path '$data' --out '$out' --n 2 --max_passages $MAXP"
  echo "submitted uprobe_${tag}  (step $step, $MAXP passages)"
done
