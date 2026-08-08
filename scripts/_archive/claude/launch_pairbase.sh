#!/bin/bash
# From-BASE pair run (2026-06-08): same 1:1 pair-uall setting as the warm grid, but
# starting from Qwen3-4B-Base (no warm-start) -- to see the data-efficiency picture
# from scratch alongside the warm version. 4 epochs (288 steps). 2 arms x 3 seeds:
#   lsat-pair-uall-base (understanding ON, scored on all q)   lsat-pair-cot-base (OFF)
#   bash scripts/claude/launch_pairbase.sh   |   DRY_RUN=1 ... to preview
set -uo pipefail
cd ~/scratch/SImpL
OUT="evaluations/lsat-ar/pairbase_curves"; mkdir -p "$OUT"
CONFIGS=(lsat-pair-uall-base lsat-pair-cot-base)
SEEDS=(42 24 36)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    prefix="Qwen3-4B-Base-${cfg}_simpl-oat_${seed}"
    train_cmd=(sbatch scripts/run/simpl_spice_oat.sh "main/${cfg}" "$seed" qwen)   # from base -> no pretrain override
    if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "[dry] ${train_cmd[*]} -> curve $prefix"; continue; fi
    out="$("${train_cmd[@]}" 2>&1)"; echo "$out"
    tid="$(grep -oE '[0-9]+' <<<"$out"|tail -n1)"; [[ -z "$tid" ]] && { echo "FAILED $cfg $seed" >&2; tid=FAILED; }
    cout="$(sbatch --dependency=afterok:"$tid" \
      --export=ALL,TRAIN_DS=lsat-ar,RUN_PREFIX="$prefix",DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_CSV="$OUT/${prefix}_allsteps.csv",COT_SAMPLES=8 \
      scripts/eval/auto_curve_eval.sh 2>&1)"; cid="$(grep -oE '[0-9]+' <<<"$cout"|tail -n1)"
    echo "| $(date '+%F %T') | WS-PAIRBASE | $tid | pairbase $cfg seed$seed (base,ep4,1:1 understand-all) + curve $cid | \`launch_pairbase.sh\` |" >> experiments/LEDGER.md
    echo "[pairbase] $cfg seed$seed -> train $tid curve $cid"
  done
done