#!/bin/bash
# Token probe for the v4 (no-brevity understanding prompt) models. Derives the deployed checkpoint
# from the v4 DEV curve at submit time, so run this only once ev4_*_dev has enough steps.
# Compare its tok/sample against the v3 baseline at the same scale:
#     4B v3 flatsimpl = 28 tok / 78% direct     8B v3 flatsimpl = 9 tok / 95% direct
# If v4 is much less terse, the terseness was caused by the "be brief" instruction.
#
#   INFERNO=1 bash src/qualitative/run_token_probe_v4.sh
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL
OUTDIR=evaluations/qualitative
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi
INFLIGHT=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
SEED=123

# scale -> base | run prefix | dev curve csv
declare -A SPEC=(
  [4b]="Qwen/Qwen3-4B-Base|Qwen3-4B-sw-race50-flatsimplv3-V4-lr2e5-clip05-b02-t10_simpl-nb|evaluations/finals_dev/race_flatsimplv3-V4_s123.csv"
  [8b]="Qwen/Qwen3-8B-Base|Qwen3-8B-final-race50-flatsimplv3-V4-lr2e5-clip05-b02-t10-short_simpl-nb|evaluations/final_8b/dev/race_flatsimplv3-V4-short_s123.csv"
)

for sc in ${ONLY_SCALE:-4b 8b}; do
  IFS='|' read -r base prefix devcsv <<< "${SPEC[$sc]}"
  [[ -s "$devcsv" ]] || { echo "SKIP $sc: no dev curve yet ($devcsv)"; continue; }
  # deployed step = argmax of the dev curve (same convention as the paper)
  step=$(python3 -c "
import csv,sys
rows=[r for r in csv.DictReader(open('$devcsv')) if r.get('cot_accuracy')]
if not rows: sys.exit(1)
best=max(rows,key=lambda r: float(r['cot_accuracy']))
print(int(best['step'].replace('step_','')))
" 2>/dev/null) || { echo "SKIP $sc: dev curve empty"; continue; }
  nsteps=$(( $(wc -l < "$devcsv") - 1 ))
  # Guard: dev-argmax is only meaningful once training is essentially done. Without this the
  # launcher happily deploys step 12 of a 300-step run (it did, once).
  maxstep=$(python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$devcsv')) if r.get('step')]
print(max(int(r['step'].replace('step_','')) for r in rows) if rows else 0)")
  MIN_MAX_STEP="${MIN_MAX_STEP:-260}"
  if (( maxstep < MIN_MAX_STEP )); then
    echo "SKIP $sc: dev curve only reaches step $maxstep (< $MIN_MAX_STEP); training still early"
    continue
  fi
  rundir=$(ls -d oat-output/race-c/${prefix}_${SEED}_* 2>/dev/null | head -1)
  ckpt="$rundir/saved_models/$(printf "step_%05d" "$step")"
  [[ -d "$ckpt" ]] || { echo "SKIP $sc: ckpt missing $ckpt"; continue; }
  tag="flatsimpl-v4-${sc}_race-c_s${SEED}"; out="$OUTDIR/${tag}.jsonl"
  [[ -s "$out" ]] && { echo "skip (exists): $out"; continue; }
  grep -qx "qual_${tag}" <<< "$INFLIGHT" && { echo "skip (in flight): $tag"; continue; }
  sbatch --job-name="qual_${tag}" --account="$ACCOUNT" --qos="$QOS" --time=2:00:00 \
    --ntasks-per-node=1 --gpus=H200:1 --cpus-per-task=8 --mem-per-cpu=48G --exclude="$EXCLUDE" \
    --output="slurm/qual/${tag}_%j.out" --error="slurm/qual/${tag}_%j.err" \
    --wrap="module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate; \
      export HF_HUB_DISABLE_XET=1; cd /storage/scratch1/1/agupta886/SImpL; \
      python -m src.qualitative.token_probe --checkpoint '$ckpt' --base_model '$base' \
        --dataset_name race-c --data_path data/race-c/final_test.jsonl --out '$out' \
        --prompt_mode default --max_model_len 4096" >/dev/null
  echo "submitted $tag  (step $step, chosen from $nsteps dev points)"
done
