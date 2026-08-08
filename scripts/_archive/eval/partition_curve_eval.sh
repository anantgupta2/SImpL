#!/bin/bash
# Partition a dense curve eval into CHUNKS parallel jobs (each a disjoint step range
# -> its own CSV), then afterok-merge into OUTPUT_CSV. Use when one run has too many
# checkpoints to evaluate inside a single embers window.
#   Env: TRAIN_DS, RUN_PREFIX, DATASET_NAME, [DATA_PATH], OUTPUT_CSV, [COT_SAMPLES=8],
#        [CHUNKS] (auto if unset), [MODELS_PER_CHUNK=15]
#   bash scripts/eval/partition_curve_eval.sh
# Part CSVs (+their .json sidecars) are written to a parts/ subfolder next to OUTPUT_CSV
# and DELETED by the merge job on success, so only the final merged CSV is left behind.
# Number of chunks is chosen from the checkpoint count (~MODELS_PER_CHUNK models each)
# unless CHUNKS is set explicitly.
set -euo pipefail
cd ~/scratch/SImpL
MODELS_PER_CHUNK="${MODELS_PER_CHUNK:-15}"
RUN_DIR="$(ls -dt oat-output/${TRAIN_DS}/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)"
[[ -z "$RUN_DIR" ]] && { echo "partition: no run dir for $RUN_PREFIX" >&2; exit 1; }
mapfile -t STEPS < <(ls "$RUN_DIR/saved_models" 2>/dev/null | sed -n 's/^step_0*\([0-9]\+\)$/\1/p' | sort -n)
N=${#STEPS[@]}
[[ "$N" -eq 0 ]] && { echo "partition: no checkpoints" >&2; exit 1; }
# auto chunk count: ceil(N / MODELS_PER_CHUNK), at least 1, unless CHUNKS given
CHUNKS="${CHUNKS:-$(( (N + MODELS_PER_CHUNK - 1) / MODELS_PER_CHUNK ))}"
[[ "$CHUNKS" -lt 1 ]] && CHUNKS=1
echo "partition: $N checkpoints -> $CHUNKS chunks (~$MODELS_PER_CHUNK models/chunk)"

# parts live in a subfolder so the eval dir stays uncluttered
OUT_DIR="$(dirname "$OUTPUT_CSV")"; OUT_BASE="$(basename "${OUTPUT_CSV%.csv}")"
PARTS_DIR="$OUT_DIR/parts"; mkdir -p "$PARTS_DIR"

# QOS/account for the eval jobs: default embers (preemptible), override to inferno via env.
EVAL_ACCOUNT="${EVAL_ACCOUNT:-gts-nisha3}"; EVAL_QOS="${EVAL_QOS:-embers}"; EVAL_TIME="${EVAL_TIME:-2:00:00}"
per=$(( (N + CHUNKS - 1) / CHUNKS ))
deps=()
parts=()
for ((c=0; c<CHUNKS; c++)); do
  lo_idx=$(( c * per )); hi_idx=$(( lo_idx + per - 1 ))
  [[ "$lo_idx" -ge "$N" ]] && break
  [[ "$hi_idx" -ge "$N" ]] && hi_idx=$(( N - 1 ))
  MN=${STEPS[$lo_idx]}; MX=${STEPS[$hi_idx]}
  partcsv="$PARTS_DIR/${OUT_BASE}.part$((c+1)).csv"; parts+=("$partcsv")
  jid="$(sbatch --parsable --account="$EVAL_ACCOUNT" --qos="$EVAL_QOS" --time="$EVAL_TIME" --export=ALL,TRAIN_DS="$TRAIN_DS",RUN_PREFIX="$RUN_PREFIX",DATASET_NAME="$DATASET_NAME",DATA_PATH="${DATA_PATH:-}",OUTPUT_CSV="$partcsv",COT_SAMPLES="${COT_SAMPLES:-8}",MIN_STEP="$MN",MAX_STEP="$MX" scripts/eval/auto_curve_eval.sh)"
  deps+=("$jid"); echo "  chunk $((c+1)): steps [$MN,$MX] -> $partcsv (job $jid)"
done
# merge job (afterok all chunks): concat part CSVs, keep header once, dedupe by step,
# then delete the part CSVs and their .json sidecars on success.
dep=$(IFS=:; echo "${deps[*]}")
mkdir -p slurm/evals/merge_logs
sbatch --parsable --dependency=afterok:"$dep" --account="$EVAL_ACCOUNT" --qos="$EVAL_QOS" --time=00:10:00 \
  --job-name=merge --output=slurm/evals/merge_logs/merge_%j.out --error=slurm/evals/merge_logs/merge_%j.err \
  --export=ALL,OUTCSV="$OUTPUT_CSV",PARTS="${parts[*]}" --wrap='
python3 - <<PY
import csv,os
out=os.environ["OUTCSV"]; parts=os.environ["PARTS"].split()
rows={}; header=None
for p in parts:
    if not os.path.exists(p): continue
    with open(p) as f:
        r=list(csv.reader(f))
    if not r: continue
    header=r[0]
    si=header.index("step")
    for row in r[1:]: rows[row[si]]=row
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(header)
    for s in sorted(rows, key=lambda x:int(x.split("_")[1])): w.writerow(rows[s])
print(f"merged {len(rows)} steps -> {out}")
# clean up parts + their json sidecars now that the merge succeeded
for p in parts:
    for f in (p, p[:-4]+".json"):
        try: os.remove(f)
        except OSError: pass
print(f"cleaned {len(parts)} part files")
PY' >/dev/null && echo "  merge job (afterok ${dep}) -> $OUTPUT_CSV (parts auto-deleted on success)"