#!/bin/bash
# Warm-start PAIR run (2026-06-08): 1:1 understanding:cot. Each (passage,question) pair
# generates its OWN understanding (scored on ALL the passage's questions) AND a cot on
# that paired question -> every cot has a matching understanding. Warm-start from per-seed
# cot-merged base, 4 epochs (72 steps/epoch -> 288 steps; ~old 24-epoch exposure for both
# objectives). 2 arms x 3 seeds:
#   lsat-pair-uall-warm (understanding ON, scored on all q)   lsat-pair-cot-warm (OFF)
# Each train -> afterok in-domain LSAT avg@8 full-curve eval.
#   bash scripts/claude/launch_pairwarm.sh   |   DRY_RUN=1 ... to preview
set -uo pipefail
cd ~/scratch/SImpL
OUT="evaluations/lsat-ar/pairwarm_curves"; mkdir -p "$OUT"
CONFIGS=(lsat-pair-uall-warm lsat-pair-cot-warm)
SEEDS=(42 24 36)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    base="oat-output/staged-bases/lsat-cot-merged_seed${seed}"
    prefix="Qwen3-4B-Base-${cfg}_simpl-oat_${seed}"
    train_cmd=(sbatch scripts/run/simpl_spice_oat.sh "main/${cfg}" "$seed" qwen "$base")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "[dry] ${train_cmd[*]} -> curve $prefix"; continue; fi
    out="$("${train_cmd[@]}" 2>&1)"; echo "$out"
    tid="$(grep -oE '[0-9]+' <<<"$out"|tail -n1)"; [[ -z "$tid" ]] && { echo "FAILED $cfg $seed" >&2; tid=FAILED; }
    cout="$(sbatch --dependency=afterok:"$tid" \
      --export=ALL,TRAIN_DS=lsat-ar,RUN_PREFIX="$prefix",DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_CSV="$OUT/${prefix}_allsteps.csv",COT_SAMPLES=8 \
      scripts/eval/auto_curve_eval.sh 2>&1)"; cid="$(grep -oE '[0-9]+' <<<"$cout"|tail -n1)"
    echo "| $(date '+%F %T') | WS-PAIRWARM | $tid | pairwarm $cfg seed$seed (warm,ep4,1:1 understand-all) + curve $cid | \`launch_pairwarm.sh\` |" >> experiments/LEDGER.md
    echo "[pairwarm] $cfg seed$seed -> train $tid curve $cid"
  done
done