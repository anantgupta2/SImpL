#!/bin/bash
# Two qualitative jobs the user asked for:
#   TASK 1 -- LSAT three-way UNDERSTANDINGS (understander vs base vs cot16) on the LSAT-AR test,
#             parallel to the RACE-on-CosmosQA three-way in qualitative_outputs.md. The understander
#             file (lsat-8b_s123.jsonl, step 316) already exists; this fills in BASE and COT16.
#   TASK 2 -- Full UNDERSTANDING + ANSWER pipeline on CosmosQA (dataset=race-c prompt, the RACE
#             generalization target) for all three RACE models -- dumps the understanding AND the
#             answer it produces, per question.
#   INFERNO=1 bash src/qualitative/run_lsat_and_ua.sh
set -euo pipefail
cd /storage/scratch1/1/agupta886/SImpL
OUTDIR=evaluations/qualitative_understandings
mkdir -p "$OUTDIR" slurm/qual
EXCLUDE="atl1-1-03-018-14-0,atl1-1-03-020-11-0,atl1-1-01-007-28-0,atl1-1-01-009-16-0,atl1-1-01-009-9-0,atl1-1-03-019-2-0,atl1-1-03-020-18-0"
if [[ "${INFERNO:-0}" == "1" ]]; then ACCOUNT=gts-schava6-qcf; QOS=inferno; else ACCOUNT=gts-nisha3; QOS=embers; fi
BASE=Qwen/Qwen3-8B-Base
EMPTY=evaluations/_base_empty            # base model: no adapter_config.json -> runs the raw base
MAXP="${MAX_PASSAGES:-50}"

LSAT_UND=$(ls -d oat-output/lsat-ar/Qwen3-8B-final-lsat50-flatsimpl-lr2e5-clip05-b02-t10_simpl-nb_123_* | head -1)
LSAT_COT=$(ls -d oat-output/lsat-ar/Qwen3-8B-final-lsat50-cotn16-lr2e5-clip05-b02-t10_cot-only_123_* | head -1)
RACE_UND=$(ls -d oat-output/race-c/Qwen3-8B-final-race50-flatsimplv3-lr2e5-clip05-b02-t10-short_simpl-nb_123_* | head -1)
RACE_COT=$(ls -d oat-output/race-c/Qwen3-8B-final-race50-cotn16-lr2e5-clip05-b02-t10-short_cot-only_123_* | head -1)

submit () {  # name | module (u=understanding-only, ua=understanding+answer) | ckpt | dataset | data | out
  local name="$1" mode="$2" ckpt="$3" dsn="$4" data="$5" out="$6"
  [[ "$ckpt" == "$EMPTY" || -d "$ckpt" ]] || { echo "MISSING ckpt: $ckpt"; return; }
  [[ -s "$out" ]] && { echo "skip (exists): $out"; return; }
  local py
  if [[ "$mode" == "u" ]]; then
    py="python -m src.qualitative.understanding_probe --checkpoint '$ckpt' --base_model '$BASE' \
        --dataset_name '$dsn' --data_path '$data' --out '$out' --n 2 --max_passages $MAXP"
  else
    py="python -m src.qualitative.understanding_answer_probe --checkpoint '$ckpt' --base_model '$BASE' \
        --dataset_name '$dsn' --data_path '$data' --out '$out' --max_passages $MAXP"
  fi
  sbatch --job-name="q_${name}" --account="$ACCOUNT" --qos="$QOS" --time=1:30:00 \
    --ntasks-per-node=1 --gpus=H200:1 --cpus-per-task=8 --mem-per-cpu=48G --exclude="$EXCLUDE" \
    --output="slurm/qual/q_${name}_%j.out" --error="slurm/qual/q_${name}_%j.err" \
    --wrap="module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate; \
      export HF_HUB_DISABLE_XET=1; cd /storage/scratch1/1/agupta886/SImpL; $py"
  echo "submitted q_${name} -> $out"
}

# --- TASK 1: LSAT three-way understandings (LSAT-AR test) ---
submit "lsat_BASE"  u "$EMPTY"                          lsat-ar data/lsat-ar/final_test.jsonl "$OUTDIR/lsat-BASE_s123.jsonl"
submit "lsat_COT16" u "$LSAT_COT/saved_models/step_00152" lsat-ar data/lsat-ar/final_test.jsonl "$OUTDIR/lsat-COT16_s123.jsonl"

# --- TASK 2: understanding + answer on CosmosQA (race-c prompt) ---
submit "ua_race_UND"   ua "$RACE_UND/saved_models/step_00188" race-c data/cosmosqa/test_42_all.jsonl "$OUTDIR/ua-cosmosqa-UND_s123.jsonl"
submit "ua_race_BASE"  ua "$EMPTY"                            race-c data/cosmosqa/test_42_all.jsonl "$OUTDIR/ua-cosmosqa-BASE_s123.jsonl"
submit "ua_race_COT16" ua "$RACE_COT/saved_models/step_00204" race-c data/cosmosqa/test_42_all.jsonl "$OUTDIR/ua-cosmosqa-COT16_s123.jsonl"
