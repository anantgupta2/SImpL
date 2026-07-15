#!/bin/bash
# Claude's TRAINING run log for the SImpL paper experiments (branch: simpl-paper-experiments).
# Mirrors scripts/trains_current.sh but every line is launched via `clog` so it is recorded in
# experiments/LEDGER.md with its purpose, job id, and timestamp. Uncomment a block to run it.
#
#   bash scripts/claude/claude_train.sh        # runs whatever is uncommented below
#   DRY_RUN=1 bash scripts/claude/claude_train.sh   # print, don't submit
#
# Conventions:
#   * SImpL is run with --simpl (NOT --combined): combined underperforms (see PAPER_PLAN).
#   * HP search trains on the 200 training passages, SELECTS on data/race-c/dev_holdout_100.jsonl,
#     and never touches test*.jsonl.
#   * Output run dirs land in oat-output/<dataset>/<wb_run_name>_<mode>_<seed>_<timestamp>.

cd ~/scratch/SImpL
source scripts/claude/submit.sh

# =====================================================================================
# SMOKE TEST — confirm the pipeline runs end-to-end before spending a full grid.
# (short run; uncomment to use)
# =====================================================================================
# clog SMOKE "smoke: CoT 1 prompt-epoch, verify config+output path" \
#     sbatch --time=1:00:00 scripts/run/cot_oat.sh hp/cot-lr4e6 42 qwen

# =====================================================================================
# WS1 — MINIMAL HYPERPARAMETER SEARCH for CoT-only (one seed=42, SELECT on dev).
# Tune the baseline; the tuned HP is then reused for SImpL (conservative/fair).
# Kept small: learning rate (3 points) + GRPO group size (1 extra point). lora_rank stays 64.
# num_prompt_epoch is intentionally NOT tuned here (it is the WS2 "more epochs won't help CoT"
# control axis). g16 is run at the default lr=4e-6 (LR and group size treated as separable).
# =====================================================================================
clog WS1 "CoT-only HP: lr=4e-6 seed42 (dev-select, default)" sbatch scripts/run/cot_oat.sh hp/cot-lr4e6 42 qwen
clog WS1 "CoT-only HP: lr=8e-6 seed42 (dev-select)"          sbatch scripts/run/cot_oat.sh hp/cot-lr8e6 42 qwen
clog WS1 "CoT-only HP: lr=2e-6 seed42 (dev-select)"          sbatch scripts/run/cot_oat.sh hp/cot-lr2e6 42 qwen
clog WS1 "CoT-only HP: group=16 lr=4e-6 seed42 (dev-select)" sbatch scripts/run/cot_oat.sh hp/cot-g16   42 qwen
