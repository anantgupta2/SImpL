#!/bin/bash
# Helper sourced by claude_train.sh / claude_evals.sh so every job Claude launches is
# auto-recorded in experiments/LEDGER.md (timestamp, workstream, purpose, job id, command).
#
# Usage inside a *_current-style script:
#     source scripts/claude/submit.sh
#     clog WS1 "CoT LR sweep lr=8e-6 seed42, dev-select" sbatch scripts/run/cot_oat.sh hp/cot-lr8e6 42
#
# Set DRY_RUN=1 to print what would be submitted (and log nothing).

LEDGER="${LEDGER:-experiments/LEDGER.md}"

clog() {
    local ws="$1"; shift
    local purpose="$1"; shift
    local cmd=("$@")
    local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry-run][$ws] $purpose :: ${cmd[*]}"
        return 0
    fi

    mkdir -p "$(dirname "$LEDGER")"
    if [[ ! -f "$LEDGER" ]]; then
        {
            echo "# Experiment ledger (Claude runs)"
            echo
            echo "Auto-appended by scripts/claude/submit.sh. One row per submitted job."
            echo
            echo "| submitted | WS | job id | purpose | command |"
            echo "|---|---|---|---|---|"
        } > "$LEDGER"
    fi

    local out; out="$("${cmd[@]}" 2>&1)"
    local rc=$?
    echo "$out"
    local jobid; jobid="$(grep -oE '[0-9]+' <<<"$out" | tail -n1)"
    if [[ $rc -ne 0 || -z "$jobid" ]]; then
        echo "[clog] WARNING: submit failed or no job id parsed (rc=$rc); logging as FAILED" >&2
        jobid="FAILED"
    fi
    echo "| $ts | $ws | $jobid | $purpose | \`${cmd[*]}\` |" >> "$LEDGER"
    echo "[clog] logged job $jobid -> $LEDGER"
}
