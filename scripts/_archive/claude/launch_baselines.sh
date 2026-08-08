#!/bin/bash
# Fresh baseline (untrained base model) evals -- avg@8, deterministic (--eval_seed 42),
# into evaluations/baselines_v2/. Matrix = models x {LSAT-AR, RACE-C, ProofWriter-d5}.
# Models already cached (4B/8B) launch immediately; downloaded ones (32B/Octo) go afterok
# the download job (pass DOWNLOAD_JID=<id>).
#   bash scripts/claude/launch_baselines.sh           # submit
#   DRY_RUN=1 bash scripts/claude/launch_baselines.sh # preview
set -uo pipefail
cd ~/scratch/SImpL
LEDGER="experiments/LEDGER.md"
# Output layout: evaluations/<dataset>/baselines/<model>.csv

# model tag | HF id | needs-download? (1 = afterok the download job)
MODELS=(
  "qwen3-4b-base|Qwen/Qwen3-4B-Base|0"
  "qwen3-8b-base|Qwen/Qwen3-8B-Base|0"
  "qwen3-32b-base|Qwen/Qwen3-32B-Base|1"
  "octothinker-3b-hybrid-base|OctoThinker/OctoThinker-3B-Hybrid-Base|1"
  "octothinker-8b-hybrid-base|OctoThinker/OctoThinker-8B-Hybrid-Base|1"
)
# dataset tag | DATASET_NAME | DATA_PATH
DATASETS=(
  "lsat|lsat-ar|data/lsat-ar/test_42_all.jsonl"
  "race|race-c|data/race-c/test_42_all.jsonl"
  "pwd5|proofwriter-d5|data/proofwriter-d5/test_42_300.jsonl"
)

for mrow in "${MODELS[@]}"; do
  IFS='|' read -r mtag mid needs_dl <<<"$mrow"
  dep=()
  if [[ "$needs_dl" == "1" && -n "${DOWNLOAD_JID:-}" ]]; then dep=(--dependency=afterok:"$DOWNLOAD_JID"); fi
  for drow in "${DATASETS[@]}"; do
    IFS='|' read -r dtag dname dpath <<<"$drow"
    csv="evaluations/${dname}/baselines/${mtag}.csv"; mkdir -p "$(dirname "$csv")"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "[dry] $mtag x $dtag -> $csv (dep=${dep[*]:-none})"; continue; fi
    out="$(sbatch "${dep[@]}" \
      --export=ALL,BASE_MODEL="$mid",DATASET_NAME="$dname",DATA_PATH="$dpath",COT_SAMPLES=8,OUTPUT_CSV="$csv",BASELINE_TAG="${mtag}_v2",IS_INSTRUCT=0 \
      scripts/eval/run_eval_untrained.sh 2>&1)"
    jid="$(grep -oE '[0-9]+' <<<"$out"|tail -n1)"
    echo "[baseline] $mtag x $dtag -> $jid"
    echo "| $(date '+%F %T') | WS-BASELINE | $jid | baseline-v2 $mtag x $dtag avg@8 | \`run_eval_untrained\` |" >> "$LEDGER"
  done
done