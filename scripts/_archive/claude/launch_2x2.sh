#!/bin/bash
# Launch the corrected LSAT 2x2 grid:
#   selection {rotate (no-curriculum, K=1 round-robin), frontier (curriculum, K=1)}
#   x understanding {off (cot, N=16 to compute-match), on (N=8)}
# 3 seeds each, warm-start from per-seed merged cot base, keep-all checkpoints (ep12).
# Each train job is followed (afterok) by an in-domain LSAT avg@8 curve eval over every
# saved checkpoint (fan-out: one cheap embers eval per step).
#
#   bash scripts/claude/launch_2x2.sh           # submit
#   DRY_RUN=1 bash scripts/claude/launch_2x2.sh # print only
set -uo pipefail
cd ~/scratch/SImpL
LEDGER="experiments/LEDGER.md"

CONFIGS=(lsat-2x2-cot-all lsat-2x2-simpl-all lsat-2x2-cot-front lsat-2x2-spice)
SEEDS=(42 24 36)
DS="lsat-ar"

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    base="oat-output/staged-bases/lsat-cot-merged_seed${seed}"
    prefix="Qwen3-4B-Base-${cfg}_simpl-oat_${seed}"   # = final wb_run_name (config name + appended CLI name)
    train_cmd=(sbatch scripts/run/simpl_spice_oat.sh "main/${cfg}" "$seed" qwen "$base")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[dry] ${train_cmd[*]}"
      echo "[dry] curve afterok -> RUN_PREFIX=$prefix DATASET_NAME=$DS"
      continue
    fi

    out="$("${train_cmd[@]}" 2>&1)"; echo "$out"
    tid="$(grep -oE '[0-9]+' <<<"$out" | tail -n1)"
    if [[ -z "$tid" ]]; then echo "[launch_2x2] FAILED to submit $cfg seed$seed" >&2; tid="FAILED"; fi

    cout="$(sbatch --dependency=afterok:"$tid" \
      --export=ALL,TRAIN_DS="$DS",RUN_PREFIX="$prefix",DATASET_NAME="$DS",COT_SAMPLES=8 \
      scripts/eval/fanout_curve_eval.sh 2>&1)"; echo "$cout"
    cid="$(grep -oE '[0-9]+' <<<"$cout" | tail -n1)"

    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "| $ts | WS-2x2v2 | $tid | 2x2v2 $cfg seed$seed (warm,ep12,K=1,keep-all) + curve $cid | \`launch_2x2.sh\` |" >> "$LEDGER"
    echo "[launch_2x2] $cfg seed$seed -> train $tid, curve $cid"
  done
done