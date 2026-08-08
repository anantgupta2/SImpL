#!/bin/bash
# Claude's EVAL run log for the SImpL paper experiments (branch: simpl-paper-experiments).
# Mirrors scripts/evals_current.sh but logged via `clog` into experiments/LEDGER.md.
#
#   bash scripts/claude/claude_evals.sh
#   DRY_RUN=1 bash scripts/claude/claude_evals.sh
#
# HP-search evals run on the DEV split (data/race-c/dev_holdout_100.jsonl) via DATA_PATH,
# writing CSVs to evaluations/<dataset>/hp_dev/ so they are clearly separated from final
# test-set numbers (evaluations/<dataset>/final_only/).

cd ~/scratch/SImpL
source scripts/claude/submit.sh

DEV_PATH="data/race-c/dev_holdout_100.jsonl"

# Reusable wrapper: eval a run's final checkpoint on the dev split.
# Usage: eval_dev <WS> "<purpose>" <run_name>
eval_dev() {
    local ws="$1" purpose="$2" run="$3"
    DATA_PATH="$DEV_PATH" DATASET_NAME="race-c" \
        OUTPUT_DIR="evaluations/race-c/hp_dev" \
        clog "$ws" "$purpose" \
        sbatch --export=ALL,DATASET_NAME=race-c,DATA_PATH="$DEV_PATH",OUTPUT_DIR=evaluations/race-c/hp_dev \
            scripts/eval/run_eval_final_saved_model.sh "$run"
}

# =====================================================================================
# WS1 — DEV evals for the LR search, submitted as afterok dependencies of the train jobs
# (so they auto-run on completion). Exact timestamped run dirs are known because oat
# creates the dir at startup. Submitted 2026-06-03:
#
#   sbatch --dependency=afterok:9411110 \
#     --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/dev_holdout_100.jsonl,OUTPUT_DIR=evaluations/race-c/hp_dev \
#     scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-hp-lr4e6_cot-only_42_0603T14:15:14   # eval job 9411184
#   sbatch --dependency=afterok:9411148 \
#     --export=ALL,DATASET_NAME=race-c,DATA_PATH=data/race-c/dev_holdout_100.jsonl,OUTPUT_DIR=evaluations/race-c/hp_dev \
#     scripts/eval/run_eval_final_saved_model.sh Qwen3-4B-Base-cot-hp-lr8e6_cot-only_42_0603T14:17:31   # eval job 9411185
#
# Pattern for future runs (resolve exact dir from oat-output/<ds>/ first, then):
# eval_dev WS1 "CoT HP lr=8e-6 dev eval" Qwen3-4B-Base-cot-hp-lr8e6_cot-only_42_<TS>
