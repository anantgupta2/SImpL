#!/bin/bash
# Pair-dataset run (data-efficiency framing, 2026-06-08): flattened (passage,question)
# pairs (<=8 q/passage), 2 prompt-epochs, FROM BASE Qwen3-4B. 3 arms x 3 seeds:
#   lsat-pair-simpl  (understanding ON,  N=8)   <- the SImpL arm
#   lsat-pair-cot    (understanding OFF, N=8)   <- same-data, same-compute cot
#   lsat-pair-cot2x  (understanding OFF, N=16)  <- compute-matched cot control
# Each train job is followed (afterok) by an in-domain LSAT avg@8 full-curve eval
# (fixed auto_curve_eval.sh: embers/2h, 1024 tok, flush-per-step, resumable).
#
#   bash scripts/claude/launch_pair.sh           # submit
#   DRY_RUN=1 bash scripts/claude/launch_pair.sh # print only
set -uo pipefail
cd ~/scratch/SImpL
LEDGER="experiments/LEDGER.md"
OUT="evaluations/lsat-ar/pair_curves"; mkdir -p "$OUT"

CONFIGS=(lsat-pair-simpl lsat-pair-cot lsat-pair-cot2x)
SEEDS=(42 24 36)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    prefix="Qwen3-4B-Base-${cfg}_simpl-oat_${seed}"
    train_cmd=(sbatch scripts/run/simpl_spice_oat.sh "main/${cfg}" "$seed" qwen)   # from base -> no pretrain override
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[dry] ${train_cmd[*]}  -> curve $prefix"; continue
    fi
    out="$("${train_cmd[@]}" 2>&1)"; echo "$out"
    tid="$(grep -oE '[0-9]+' <<<"$out" | tail -n1)"; [[ -z "$tid" ]] && { echo "FAILED $cfg $seed" >&2; tid=FAILED; }
    cout="$(sbatch --dependency=afterok:"$tid" \
      --export=ALL,TRAIN_DS=lsat-ar,RUN_PREFIX="$prefix",DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_CSV="$OUT/${prefix}_allsteps.csv",COT_SAMPLES=8 \
      scripts/eval/auto_curve_eval.sh 2>&1)"; echo "$cout"
    cid="$(grep -oE '[0-9]+' <<<"$cout" | tail -n1)"
    echo "| $(date '+%F %T') | WS-PAIR | $tid | pair $cfg seed$seed (base,ep2,pairs<=8,N) + curve $cid | \`launch_pair.sh\` |" >> "$LEDGER"
    echo "[launch_pair] $cfg seed$seed -> train $tid curve $cid"
  done
done