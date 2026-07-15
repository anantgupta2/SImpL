#!/usr/bin/env python3
"""Submit the cross-dataset TRANSFER evals: deployed checkpoint of a run -> unseen target dataset.

The deployed checkpoint = argmax of that seed's SOURCE dev curve (the canonical per-seed dev-argmax
convention), evaluated on the target's test set with COT_EVAL_ONLY=1, avg@8.

IDEMPOTENT: skips any cell whose output csv already exists, so it is safe to re-run after the embers
submit cap (QOSMaxSubmitJobPerUserLimit) rejects part of a batch. auto_curve_eval.sh self-heals on
preemption/walltime, so no external drip loop is needed -- just re-run this until it reports 0 missing.

  DRY_RUN=1 python scripts/claude/launch_transfers.py   # preview
  python scripts/claude/launch_transfers.py             # submit
  MAX_SUBMIT=20 python scripts/claude/launch_transfers.py  # throttle under the embers cap
"""
import csv, glob, os, re, subprocess, sys

sys.path.insert(0, "scripts/eval")
from make_final_tables import load_curve  # noqa: E402

OUT = "evaluations/transfer"
DEV = {"4B": "evaluations/finals_dev", "8B": "evaluations/final_8b/dev"}
TRAIN_DS = {"lsat": "lsat-ar", "race": "race-c"}

# (size, src) -> {xfer_method: (finals_dev_method, run_prefix_template)}
RUNS = {
    ("4B", "lsat"): {
        "cotn16": ("cotn16", "Qwen3-4B-sw-lsat50-cotn16-lr2e5-clip05-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplit-u4c12", "Qwen3-4B-sw-lsat50-flatsplit-u4c12-lr2e5-clip05-b02-t10_simpl-split_{s}"),
    },
    ("4B", "race"): {
        "cotn16": ("cotn16", "Qwen3-4B-sw-race50-cotn16-lr2e5-clip05-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplitv3-u4c12", "Qwen3-4B-sw-race50-flatsplitv3-u4c12-lr2e5-clip05-b02-t10_simpl-split_{s}"),
    },
    ("8B", "lsat"): {
        "cotn16": ("cotn16", "Qwen3-8B-final-lsat50-cotn16-lr2e5-clip05-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplit-u4c12", "Qwen3-8B-final-lsat50-flatsplit-u4c12-lr2e5-clip05-b02-t10_simpl-split_{s}"),
    },
    # 8B-race uses the SHORT runs, not -long: the long cot16 baseline is uneven (s234 died at
    # step 248 vs 496 for s123/s345, and its dev csv is contaminated with rows from the abandoned
    # pre-OOM run dir). The short runs are all equal-length and matched to the 4B budget.
    ("8B", "race"): {
        "cotn16": ("cotn16-short", "Qwen3-8B-final-race50-cotn16-lr2e5-clip05-b02-t10-short_cot-only_{s}"),
        "u4c12": ("flatsplitv3-u4c12-short", "Qwen3-8B-final-race50-flatsplitv3-u4c12-lr2e5-clip05-b02-t10-short_simpl-split_{s}"),
    },
}
# target -> (DATASET_NAME, DATA_PATH, kind)
#   kind "passage"   -> scripts/eval/auto_curve_eval.sh (the normal passage+MCQ harness)
#   kind "nopassage" -> scripts/eval/run_eval_nopassage.sh (src/eval_nopassage.py): ARC has an empty
#                       article and GSM8K is not MCQ at all, so the passage harness cannot run them.
# proofwriter-d2 DROPPED as a transfer target: the base model already scores 76 (4B) / 82 (8B) on it,
# so there is no headroom, and seed variance is enormous (SEM 5.4 at 4B) -- it cannot support a claim.
TARGETS = {
    "reclor": ("reclor", "data/reclor/test_42_all.jsonl", "passage"),
    "quail": ("quail", "data/quail/test_42_all.jsonl", "passage"),
    # The other two LSAT sections. Verified ZERO article/question overlap with the lsat-ar
    # train/dev/test pools, so they are genuinely unseen despite sharing exam ids.
    "lsatlr": ("lsat-lr", "data/lsat-lr/test_42_all.jsonl", "passage"),
    "lsatrc": ("lsat-rc", "data/lsat-rc/test_42_all.jsonl", "passage"),
    "clutrr": ("clutrr", "data/clutrr/test_42_all.jsonl", "passage"),
    "clutrrmc4": ("clutrr-mc4", "data/clutrr-mc4/test_42_all.jsonl", "passage"),
    "folio": ("folio", "data/folio/test_42_all.jsonl", "passage"),
    "cosmosqa": ("cosmosqa", "data/cosmosqa/test_42_all.jsonl", "passage"),
    "bbhld": ("bbh-logical-deduction", "data/bbh-logical-deduction/test_42_all.jsonl", "passage"),
    # True LSAT-AR analogues (analytical reasoning / logic games).
    "zebra": ("zebralogic", "data/zebralogic/test_42_all.jsonl", "passage"),
    "bbhtrack": ("bbh-tracking", "data/bbh-tracking/test_42_all.jsonl", "passage"),
    "arc": ("arc-challenge", "data/arc-challenge/test_42_all.jsonl", "nopassage"),
    "gsm8k": ("gsm8k", None, "nopassage"),
}
# reclor is LSAT-shaped and quail is RACE-shaped, so each stays with its natural source; every new
# target is run from BOTH sources (they are unseen by both).
_BOTH = ["lsatlr", "lsatrc", "clutrr", "clutrrmc4", "folio", "cosmosqa", "bbhld",
         "zebra", "bbhtrack", "arc", "gsm8k"]
SRC_TARGETS = {"lsat": ["reclor"] + _BOTH, "race": ["quail"] + _BOTH}
SEEDS = ["123", "234", "345"]


def deployed_step(size, src, dev_method, seed):
    c = load_curve(os.path.join(DEV[size], f"{src}_{dev_method}_s{seed}.csv"))
    return max(c, key=lambda k: c[k]) if c else None


def resolve_ckpt(train_ds, run_prefix, step):
    """Absolute checkpoint dir for a step. auto_curve_eval.sh resolves RUN_PREFIX itself, but the
    nopassage runner takes an explicit --checkpoint_dir, so mirror its `ls -dt | head -1` (newest
    wins -- matters where an OOM requeue left two dirs for the same seed)."""
    hits = sorted(glob.glob(f"oat-output/{train_ds}/{run_prefix}_*"), key=os.path.getmtime,
                  reverse=True)
    if not hits:
        return None
    ck = os.path.join(hits[0], "saved_models", f"step_{step:05d}")
    return ck if os.path.isdir(ck) else None


def queued_jobnames():
    """Names of this user's queued/running jobs. A cell whose job is already in flight must NOT be
    resubmitted: the csv does not exist yet, so the existence check alone would double-submit, and
    two jobs writing the same OUTPUT_CSV race each other."""
    try:
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                           capture_output=True, text=True, timeout=60)
        return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
    except Exception as e:
        print(f"[warn] squeue failed ({e}); cannot guard against double-submit -- aborting")
        sys.exit(1)


def main():
    dry = os.environ.get("DRY_RUN", "0") == "1"
    max_submit = int(os.environ.get("MAX_SUBMIT", "999"))
    inflight = queued_jobnames()
    todo, n, skipped = [], 0, 0
    for (size, src), methods in RUNS.items():
        for xm, (dev_method, pref) in methods.items():
            for seed in SEEDS:
                step = deployed_step(size, src, dev_method, seed)
                for tgt in SRC_TARGETS[src]:
                    csvp = f"{OUT}/{size}_{src}_{xm}_to_{tgt}_s{seed}.csv"
                    job = f"xfer_{size}_{src}_{xm}_to_{tgt}_s{seed}"
                    if os.path.exists(csvp):
                        continue
                    if job in inflight:
                        skipped += 1
                        continue
                    if step is None:
                        print(f"[skip] no dev curve: {size} {src} {dev_method} s{seed}")
                        continue
                    todo.append((size, src, xm, tgt, seed, step, pref.format(s=seed), csvp))

    print(f"{len(todo)} to submit ({skipped} already in flight, skipped)")
    for (size, src, xm, tgt, seed, step, pref, csvp) in todo:
        if n >= max_submit:
            print(f"[throttle] stopping at MAX_SUBMIT={max_submit}; re-run to submit the rest")
            break
        dsn, dpath, kind = TARGETS[tgt]
        base = "Qwen/Qwen3-4B-Base" if size == "4B" else "Qwen/Qwen3-8B-Base"
        job = f"xfer_{size}_{src}_{xm}_to_{tgt}_s{seed}"

        if kind == "nopassage":
            ck = resolve_ckpt(TRAIN_DS[src], pref, step)
            if ck is None:
                print(f"[skip] no checkpoint for {pref} @step{step}")
                continue
            task = "gsm8k" if tgt == "gsm8k" else "mcq_nopassage"
            env = (f"ALL,TASK={task},BASE_MODEL={base},CHECKPOINT_DIR={ck},"
                   f"OUTPUT_CSV={csvp},COT_SAMPLES=8,RUN_NAME={pref},STEP=step_{step:05d}")
            if dpath:
                env += f",DATA_PATH={dpath}"
            script = "scripts/eval/run_eval_nopassage.sh"
        else:
            env = (f"ALL,TRAIN_DS={TRAIN_DS[src]},RUN_PREFIX={pref},DATASET_NAME={dsn},"
                   f"DATA_PATH={dpath},OUTPUT_CSV={csvp},COT_SAMPLES=8,COT_EVAL_ONLY=1,"
                   f"MIN_STEP={step},MAX_STEP={step}")
            script = "scripts/eval/auto_curve_eval.sh"

        cmd = ["sbatch", f"--job-name={job}", f"--export={env}", script]
        if dry:
            print(f"[dry] {size} {src}->{tgt:<9} {xm:<6} s{seed} @step{step:<4} [{kind}] -> {csvp}")
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        jid = re.findall(r"\d+", r.stdout or "")
        ok = "OK " + jid[-1] if jid else "FAIL " + (r.stderr or r.stdout).strip()[:70]
        print(f"[{ok}] {size} {src}->{tgt} {xm} s{seed} @step{step}")
        n += 1


if __name__ == "__main__":
    main()
