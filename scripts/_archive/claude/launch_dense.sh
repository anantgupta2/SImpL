#!/bin/bash
# Dense-cot run (2026-06-08): per-passage structure -> ONE understanding/passage scored
# on ALL questions (the "old" understanding), and cot trained on ALL questions of the
# passage (dense, "this version of cot"). Warm-start from per-seed cot-merged base,
# 16 prompt-epochs (~200 steps). 2 arms x 3 seeds:
#   lsat-dense-simpl (understanding ON)   lsat-dense-cot (understanding OFF)
# Each train job -> afterok in-domain LSAT avg@8 full-curve eval (embers, fixed).
#   bash scripts/claude/launch_dense.sh   |   DRY_RUN=1 ... to preview
set -uo pipefail
cd ~/scratch/SImpL
OUT="evaluations/lsat-ar/dense_curves"; mkdir -p "$OUT"
CONFIGS=(lsat-dense-simpl lsat-dense-cot)
SEEDS=(42 24 36)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    base="oat-output/staged-bases/lsat-cot-merged_seed${seed}"
    prefix="Qwen3-4B-Base-${cfg}_simpl-oat_${seed}"
    train_cmd=(sbatch --time=16:00:00 scripts/run/simpl_spice_oat.sh "main/${cfg}" "$seed" qwen "$base")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "[dry] ${train_cmd[*]} -> curve $prefix"; continue; fi
    out="$("${train_cmd[@]}" 2>&1)"; echo "$out"
    tid="$(grep -oE '[0-9]+' <<<"$out"|tail -n1)"; [[ -z "$tid" ]] && { echo "FAILED $cfg $seed" >&2; tid=FAILED; }
    cout="$(sbatch --dependency=afterok:"$tid" \
      --export=ALL,TRAIN_DS=lsat-ar,RUN_PREFIX="$prefix",DATASET_NAME=lsat-ar,DATA_PATH=data/lsat-ar/test_42_all.jsonl,OUTPUT_CSV="$OUT/${prefix}_allsteps.csv",COT_SAMPLES=8 \
      scripts/eval/auto_curve_eval.sh 2>&1)"; cid="$(grep -oE '[0-9]+' <<<"$cout"|tail -n1)"
    echo "| $(date '+%F %T') | WS-DENSE | $tid | dense $cfg seed$seed (warm,ep16,K=all,understand-all) + curve $cid | \`launch_dense.sh\` |" >> experiments/LEDGER.md
    echo "[dense] $cfg seed$seed -> train $tid curve $cid"
  done
done