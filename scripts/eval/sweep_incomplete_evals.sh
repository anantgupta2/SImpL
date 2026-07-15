#!/bin/bash
# Submit/resubmit a resume-eval for every INCOMPLETE run that has no active job.
# Dedupe is by per-run job name ("ev_<run>") checked against squeue -- NOT by CSV
# mtime -- so a still-pending job is never double-submitted. Safe to run repeatedly:
# the first run submits all incomplete evals; later runs only resubmit ones whose job
# has ended (e.g. hit walltime) while still incomplete. Resume-evals skip done steps.
#   bash scripts/eval/sweep_incomplete_evals.sh
set -euo pipefail
cd ~/scratch/SImpL
JOINT="data/lsat-ar/eval_joint_42_all.jsonl"
RACE_TEST="data/race-c/test_42_all.jsonl"
# Exact dedup: the set of OUTPUT_CSVs that ANY of my queued/running jobs is working on.
# (read from each job's --export, so it works regardless of job name.) A run in this set
# already has a job -> never resubmit.
ACTIVE_OUTS="$(for j in $(squeue -u "$USER" -h -o '%i'); do scontrol show job "$j" 2>/dev/null | grep -oE 'OUTPUT_CSV=[^ ,]*' | cut -d= -f2 || true; done | sort -u)"
MAX_JOBS="${MAX_JOBS:-49}"            # never let the eval queue exceed this (watcher excluded)
CURJOBS="$(squeue -u "$USER" -h -o '%j' | grep -cv '^sweep_watcher' || true)"

consider () {  # $1 run_dir  $2 out_csv  $3 train_ds  $4 dataset  $5 data_path
  local rd="$1" out="$2" tds="$3" dsn="$4" dp="$5"
  [[ -d "$rd/saved_models" ]] || return 0
  local nck=$(ls "$rd/saved_models" 2>/dev/null | grep -c '^step_')
  [[ "$nck" -eq 0 ]] && return 0
  local ne=0; [[ -f "$out" ]] && ne=$(( $(wc -l < "$out") - 1 )); (( ne<0 )) && ne=0
  [[ "$ne" -ge "$nck" ]] && { COMPLETE=$((COMPLETE+1)); return 0; }     # already complete -> never rerun
  grep -Fxq "$out" <<<"$ACTIVE_OUTS" && { ACTIVECNT=$((ACTIVECNT+1)); return 0; }  # a job already exists for this run
  if [[ "$CURJOBS" -ge "$MAX_JOBS" ]]; then CAPPED=$((CAPPED+1)); return 0; fi      # respect the cap
  local prefix="$(basename "$rd" | sed -E 's/_[0-9]{4}T[0-9:]+$//')"
  local jn="ev_${prefix#Qwen3-4B-Base-}"
  sbatch --parsable --job-name="$jn" --account=gts-nisha3 --qos=embers --time="${EVAL_TIME:-4:00:00}" \
    --export=ALL,TRAIN_DS="$tds",RUN_PREFIX="$prefix",DATASET_NAME="$dsn",DATA_PATH="$dp",OUTPUT_CSV="$out",COT_SAMPLES=8 \
    scripts/eval/auto_curve_eval.sh >/dev/null \
    && { echo "  submit ($ne/$nck): $jn"; SUBMIT=$((SUBMIT+1)); CURJOBS=$((CURJOBS+1)); }
}

SUBMIT=0; ACTIVECNT=0; COMPLETE=0; CAPPED=0
while IFS=$'\t' read -r rd out; do [[ -n "$rd" ]] && consider "$rd" "$out" lsat-ar lsat-ar "$JOINT"; done < evaluations/lsat-re-eval/manifest.tsv
while IFS=$'\t' read -r rd out; do [[ -n "$rd" ]] && consider "$rd" "$out" race-c race-c "$RACE_TEST"; done < evaluations/race-re-eval/manifest.tsv
for s in 24 36 42; do
  d=$(ls -d oat-output/lsat-ar/Qwen3-4B-Base-final-lsat100-simpl-pe2-us2_simpl-oat_${s}_* 2>/dev/null | head -1) || true
  [[ -n "$d" ]] && consider "$d" "evaluations/lsat-re-eval/final/final-lsat100-simpl-pe2-us2_simpl-oat_${s}.csv" lsat-ar lsat-ar "$JOINT"
done
# beta sweep (joint set) -> evaluations/bsweep/  (afterany evals hold these slots until
# training ends, so this only acts as a self-healing fallback if one fails)
for d in oat-output/lsat-ar/Qwen3-4B-Base-bsweep-lsat50-*/; do
  [[ -d "$d/saved_models" ]] || continue
  stem=$(basename "$d" | sed -E 's/^Qwen3-4B-Base-bsweep-(lsat50-[a-z0-9-]+)_(cot-only|simpl-oat)_.*/\1/')
  consider "${d%/}" "evaluations/bsweep/${stem}.csv" lsat-ar lsat-ar "$JOINT"
done
echo "sweep: submitted=$SUBMIT  already-active=$ACTIVECNT  complete=$COMPLETE  capped=$CAPPED"