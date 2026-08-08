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

OUT = os.environ.get("XFER_OUT", "evaluations/transfer")
# nodes that have hung or OOM'd jobs; passed to sbatch --exclude (commas are fine here -- this is a
# direct sbatch arg, NOT --export, which truncates at the first comma). XFER_EXCLUDE= to disable.
BAD_NODES = os.environ.get("XFER_EXCLUDE", ",".join([
    "atl1-1-03-018-14-0", "atl1-1-03-020-11-0", "atl1-1-01-007-28-0", "atl1-1-01-009-16-0",
    "atl1-1-01-009-9-0", "atl1-1-03-019-2-0", "atl1-1-03-020-18-0",
]))
DEV = {"4B": "evaluations/finals_dev", "8B": "evaluations/final_8b/dev",
       "1p7B": "evaluations/final_1p7b/dev",
       "4Bs100": "evaluations/scale100_dev",
       # step-capped variant: same dev curves, but selection searches only the first N steps.
       "4Bs100c256": "evaluations/scale100_dev",
       "4Bs100c200": "evaluations/scale100_dev", "4Bc200": "evaluations/finals_dev",
       "4Babl": "evaluations/finals_dev", "8Babl": "evaluations/final_8b/dev",
       "1p7Babl": "evaluations/final_1p7b/dev",
       "4Bv4": "evaluations/finals_dev", "8Bv4": "evaluations/final_8b/dev"}
WIN3_SIZES = set()
STEP_CAP = {"4Bs100c256": 256, "4Bs100c200": 200, "4Bc200": 200}   # size -> max step eligible for dev-argmax
# NOTE cap 256: RACE-50 needs no capped variant -- all its deployed steps are <=248, so
# evaluations/transfer/4B_race_* IS the cap-256 result. Only RACE-100 moves (5/6 cells).
# capped variant reuses the uncapped result whenever the cap does not move the step,
# so we only spend GPU on the cells the cap actually changes.
REUSE_FROM = {"4Bs100c256": "4Bs100", "4Bs100c200": "4Bs100", "4Bc200": "4B"}
TRAIN_DS = {"lsat": "lsat-ar", "race": "race-c"}
BASE_MODEL = {"8B": "Qwen/Qwen3-8B-Base", "8Babl": "Qwen/Qwen3-8B-Base", "8Bv4": "Qwen/Qwen3-8B-Base",
              "1p7B": "Qwen/Qwen3-1.7B-Base", "1p7Babl": "Qwen/Qwen3-1.7B-Base"}  # else 4B

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
    ("1p7B", "lsat"): {
        "cotn16": ("cotn16", "Qwen3-1p7B-final-lsat50-cotn16-lr8e5-clip1-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplit-u4c12", "Qwen3-1p7B-final-lsat50-flatsplit-u4c12-lr8e5-clip1-b02-t10_simpl-split_{s}"),
    },
    ("1p7B", "race"): {
        "cotn16": ("cotn16", "Qwen3-1p7B-final-race50-cotn16-lr8e5-clip1-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplitv3-u4c12", "Qwen3-1p7B-final-race50-flatsplitv3-u4c12-lr8e5-clip1-b02-t10_simpl-split_{s}"),
    },
    # RACE-100 (s100) 4B: the data-scaling comparison vs race-50. dev curves in scale100_dev are
    # named race_<method>_sN (src stays "race"); "4Bs100" only selects that dev dir + csv prefix.
    ("4Bs100", "race"): {
        "cotn16": ("cotn16", "Qwen3-4B-sw-race100-cotn16-lr2e5-clip05-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplitv3-u4c12", "Qwen3-4B-sw-race100-flatsplitv3-u4c12-lr2e5-clip05-b02-t10_simpl-split_{s}"),
    },
    # LSAT-100 (s100) 4B: the other half of the data-scaling control. NOTE u4c12 s345 timed out at
    # step 260/649, so its dev curve is short -- its deployed step comes from that truncated range.
    ("4Bs100", "lsat"): {
        "cotn16": ("cotn16", "Qwen3-4B-sw-lsat100-cotn16-lr2e5-clip05-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplit-u4c12", "Qwen3-4B-sw-lsat100-flatsplit-u4c12-lr2e5-clip05-b02-t10_simpl-split_{s}"),
    },
    # STEP-CAPPED selection: search only the first STEP_CAP steps. Only RACE-100 cot16 differs
    # from the uncapped choice (its dev peaked at 440-456; capped it lands at 220/128/380).
    # cap-256: only RACE-100 needs it (RACE-50's steps are all <=248 -> uncapped == capped).
    ("4Bs100c256", "race"): {
        "cotn16": ("cotn16", "Qwen3-4B-sw-race100-cotn16-lr2e5-clip05-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplitv3-u4c12", "Qwen3-4B-sw-race100-flatsplitv3-u4c12-lr2e5-clip05-b02-t10_simpl-split_{s}"),
    },
    # cap-200 variants (kept running alongside 256 as a second data point)
    ("4Bs100c200", "race"): {
        "cotn16": ("cotn16", "Qwen3-4B-sw-race100-cotn16-lr2e5-clip05-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplitv3-u4c12", "Qwen3-4B-sw-race100-flatsplitv3-u4c12-lr2e5-clip05-b02-t10_simpl-split_{s}"),
    },
    ("4Bc200", "race"): {
        "cotn16": ("cotn16", "Qwen3-4B-sw-race50-cotn16-lr2e5-clip05-b02-t10_cot-only_{s}"),
        "u4c12": ("flatsplitv3-u4c12", "Qwen3-4B-sw-race50-flatsplitv3-u4c12-lr2e5-clip05-b02-t10_simpl-split_{s}"),
    },
    # RACE split-ablation arms on transfer (compare against the existing 4B_race_cotn16_* baseline).
    # u12c4 = 75% understanding, u16c0 = understanding-only, flatsimpl = 8:8.
    ("4Babl", "race"): {
        "u12c4": ("flatsplitv3-u12c4", "Qwen3-4B-sw-race50-flatsplitv3-u12c4-lr2e5-clip05-b02-t10_simpl-split_{s}"),
        "u16c0": ("flatsplitv3-u16c0", "Qwen3-4B-sw-race50-flatsplitv3-u16c0-lr2e5-clip05-b02-t10_simpl-split_{s}"),
        "flatsimpl": ("flatsimplv3", "Qwen3-4B-sw-race50-flatsimplv3-lr2e5-clip05-b02-t10_simpl-nb_{s}"),
    },
    # LSAT-trained 8:8 arm. Compares against the existing 4B_lsat_{cotn16,u4c12}_* cells.
    ("4Babl", "lsat"): {
        "flatsimpl": ("flatsimpl", "Qwen3-4B-sw-lsat50-flatsimpl-lr2e5-clip05-b02-t10_simpl-nb_{s}"),
    },
    # 1.7B RACE 8:8 arm -- the capacity-floor check: does a larger understanding share rescue the
    # arm that fails at this scale? Compares against 1p7B_race_{cotn16,u4c12}_*.
    ("1p7Babl", "race"): {
        "flatsimpl": ("flatsimplv3", "Qwen3-1p7B-final-race50-flatsimplv3-lr8e5-clip1-b02-t10_simpl-nb_{s}"),
    },
    # 8B LSAT 8:8 arm -- appendix table only (LSAT-RC, LSAT-LR, ARC). Uses the non-"-long" runs to
    # match the 8B_lsat_{cotn16,u4c12} baselines it is compared against.
    ("8Babl", "lsat"): {
        "flatsimpl": ("flatsimpl", "Qwen3-8B-final-lsat50-flatsimpl-lr2e5-clip05-b02-t10_simpl-nb_{s}"),
    },
    # v4 understanding prompt (brevity demand removed). Compare against the v3 flatsimpl cells.
    ("4Bv4", "race"): {
        "flatsimplv4": ("flatsimplv3-V4", "Qwen3-4B-sw-race50-flatsimplv3-V4-lr2e5-clip05-b02-t10_simpl-nb_{s}"),
    },
    ("8Bv4", "race"): {
        "flatsimplv4": ("flatsimplv3-V4-short", "Qwen3-8B-final-race50-flatsimplv3-V4-lr2e5-clip05-b02-t10-short_simpl-nb_{s}"),
    },
    # 8B RACE 8:8 arm, on the RC + long-context panel. Compares against 8B_race_{cotn16,u4c12}_*.
    ("8Babl", "race"): {
        "flatsimpl": ("flatsimplv3-short", "Qwen3-8B-final-race50-flatsimplv3-lr2e5-clip05-b02-t10-short_simpl-nb_{s}"),
    },
}
# skip the YaRN long-context targets (they floor under rope scaling); keep lbsmall (native 32k).
_YARN_SKIP = {"longbench", "lbmedium", "lblong", "lbshort", "lbmed"}
_RC_KEEP = {"quail", "cosmosqa", "lsatrc", "quality", "lbsmall", "lsatlr"}
SIZE_SKIP = {"1p7B": _YARN_SKIP, "4Bs100": _YARN_SKIP, "4Bs100c256": _YARN_SKIP,
             "4Bs100c200": _YARN_SKIP, "4Bc200": _YARN_SKIP,
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
    # GSM-Symbolic (apple/GSM-Symbolic "main"): 20 held-out TEMPLATES x 50 instances
    # = 1000 numeric questions, flattened from the Roles project build.
    "gsmsym": ("gsm-symbolic", "data/gsm-symbolic/test_42_all.jsonl", "nopassage"),
    # RACE -> LSAT-AR: the symmetric complement of the original LSAT->RACE finding.
    # AR is a FORMAL target; not run from lsat (in-domain).
    "lsatar": ("lsat-ar", "data/lsat-ar/final_test.jsonl", "passage"),
    # QuALITY: long-document RC, the understanding stress test (~6.5k-token passages,
    # NOT truncated at eval). Run from all sources.
    "quality": ("quality", "data/quality/test_42_all.jsonl", "passage"),
    # LongBench-v2: 10k-128k-token RC. The far end of the passage-length ladder. Needs YaRN (Qwen3
    # past 32k) -> flagged in LONG_CTX below so the submit adds MAX_MODEL_LEN + ROPE_YARN_FACTOR.
    "longbench": ("longbench-v2", "data/longbench-v2/test_42_all.jsonl", "passage"),
    # Length TIERS: each uses the smallest window that holds it (cheaper GPU + isolates whether the
    # below-chance floor on the full set is a long-context coherence break, not a task limit).
    "lbsmall": ("longbench-v2-small", "data/longbench-v2-small/test_42_all.jsonl", "passage"),
    "lbmedium": ("longbench-v2-medium", "data/longbench-v2-medium/test_42_all.jsonl", "passage"),
    "lblong": ("longbench-v2-long", "data/longbench-v2-long/test_42_all.jsonl", "passage"),
    # DATASET-NATIVE length bands (word-based, from the `length` field). short(178)/medium(125)
    # are what fits in 128k; the native "long" band (median ~573k tok) is unrunnable. Distinct from
    # the token-size tiers above -- these use the paper's own categories.
    "lbshort": ("longbench-lb-short", "data/longbench-lb-short/test_42_all.jsonl", "passage"),
    "lbmed": ("longbench-lb-medium", "data/longbench-lb-medium/test_42_all.jsonl", "passage"),
}
# Targets whose passages exceed Qwen3's native 32768 ctx: eval needs YaRN rope scaling + a raised
# max_model_len (and eager mode, handled in eval_saved_models). Env-driven via auto_curve_eval.sh.
# small tier fits native 32k -> NOT here (no YaRN, uses fast compiled kernels).
LONG_CTX = {"longbench": ("131072", "4.0"),   # tgt -> (MAX_MODEL_LEN, ROPE_YARN_FACTOR)
            "lbmedium": ("65536", "2.0"),
            "lblong": ("131072", "4.0"),
            # dataset-native bands: short reaches 94k, medium 129k -> both need the 128k window.
            "lbshort": ("131072", "4.0"),
            "lbmed": ("131072", "4.0")}
# reclor is LSAT-shaped and quail is RACE-shaped, so each stays with its natural source; every new
# target is run from BOTH sources (they are unseen by both).
_BOTH = ["lsatlr", "lsatrc", "clutrr", "clutrrmc4", "folio", "cosmosqa", "bbhld",
         "zebra", "bbhtrack", "arc", "gsm8k", "quality", "gsmsym"]
SRC_TARGETS = {"lsat": ["reclor"] + _BOTH,
               "race": ["quail", "lsatar", "reclor", "longbench", "lbsmall", "lbmedium", "lblong",
                        "lbshort", "lbmed"] + _BOTH}
# ablation is scoped to the RC + long-context panel (TARGETS must exist first).
SIZE_SKIP["4Babl"] = set(TARGETS) - _RC_KEEP
SIZE_SKIP["8Babl"] = set(TARGETS) - _RC_KEEP
# 1.7B also skips the YaRN long-context tiers (they floor on this base model too).
SIZE_SKIP["1p7Babl"] = (set(TARGETS) - _RC_KEEP) | _YARN_SKIP
# gsmsym is a targeted generalization probe: only the main 4B/8B LSAT and RACE models (plus their
# 8:8 arms via EXTRA_TARGETS). Skip it everywhere else -- otherwise membership in _BOTH drags it
# into 1.7B / data-scaling / capped sources too (250 cells instead of 36).
# NOTE: build a NEW set per size; several SIZE_SKIP values alias the same _YARN_SKIP object.
for _sz in ("1p7B", "1p7Babl", "4Bs100", "4Bs100c256", "4Bs100c200", "4Bc200"):
    SIZE_SKIP[_sz] = set(SIZE_SKIP.get(_sz, ())) | {"gsmsym"}
# ...except where we deliberately widen it. SIZE_SKIP is per-size, so without this the formal
# sweep below would also drag u12c4/u16c0 along. Keyed (size, src, xfer_method).
_FORMAL = {"reclor", "folio", "bbhld", "bbhtrack", "clutrr", "clutrrmc4", "zebra"}
EXTRA_TARGETS = {
    # 8:8 on the formal family, plus the GSM-Symbolic generalization probe
    ("4Babl", "race", "flatsimpl"): _FORMAL | {"gsmsym"},
    ("4Babl", "lsat", "flatsimpl"): set(SRC_TARGETS["lsat"]),  # 8:8 on every LSAT-source target
    # arc and gsmsym sit outside _RC_KEEP, so they must be exempted explicitly
    ("8Babl", "lsat", "flatsimpl"): {"arc", "gsmsym"},
    ("8Babl", "race", "flatsimpl"): {"gsmsym"},
}
# Hard restriction (vs EXTRA_TARGETS, which only widens): run ONLY these targets for this cell.
# Needed because SIZE_SKIP is per-size, so 8Babl inherits the RACE panel's RC keep-set.
_V4_PANEL = {"quail", "cosmosqa", "lsatrc", "quality", "lbsmall", "gsmsym"}
ONLY_TARGETS = {
    ("8Babl", "lsat", "flatsimpl"): {"lsatrc", "lsatlr", "arc", "gsmsym"},  # appendix cols + gsmsym
    # v4: RC + long-context panel, plus gsmsym (does the less-terse model keep its math?)
    ("4Bv4", "race", "flatsimplv4"): _V4_PANEL,
    ("8Bv4", "race", "flatsimplv4"): _V4_PANEL,
}

SEEDS = ["123", "234", "345"]


def deployed_step(size, src, dev_method, seed):
    c = load_curve(os.path.join(DEV[size], f"{src}_{dev_method}_s{seed}.csv"))
    if not c:
        return None
    if size in WIN3_SIZES:
        return _win3_step(c)
    cap = STEP_CAP.get(size)
    if cap is not None:
        c = {k: v for k, v in c.items() if k <= cap}
        if not c:
            return None
    return max(c, key=lambda k: c[k])


def _win3_step(c):
    """Rolling-window-3: dev-mean picks the window, dev-argmax within it picks the checkpoint."""
    from statistics import mean as _mean
    steps = sorted(c)
    if len(steps) < 3:
        return max(steps, key=lambda k: c[k])
    bi = max(range(len(steps) - 2), key=lambda i: _mean(c[steps[j]] for j in (i, i + 1, i + 2)))
    return max(steps[bi:bi + 3], key=lambda k: c[k])


def single_step(size, src, dev_method, seed):
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
    # accept "," or ":" -- SLURM --export splits on commas, so a comma-separated value
    # gets truncated at the first item when passed through sbatch.
    only = {x for x in re.split(r"[,:]", os.environ.get("ONLY_SIZES", "")) if x}
    # ONLY_TGT restricts to specific TARGETS (same ':' or ',' syntax). Needed because several
    # capped/scaling cells legitimately live in transfers_capped/, so a plain run wants to
    # (re)submit them into evaluations/transfer/; filtering by target avoids that.
    only_tgt = {x for x in re.split(r"[,:]", os.environ.get("ONLY_TGT", "")) if x}
    todo, n, skipped = [], 0, 0
    for (size, src), methods in RUNS.items():
        if only and size not in only:
            continue
        for xm, (dev_method, pref) in methods.items():
            for seed in SEEDS:
                step = deployed_step(size, src, dev_method, seed)
                # capped variant: if the cap did not move this seed's step, the uncapped csv is
                # identical -> skip (aggregation reads it from the base size instead).
                base_size = REUSE_FROM.get(size)
                if base_size is not None and step == deployed_step(base_size, src, dev_method, seed):
                    continue
                extra = EXTRA_TARGETS.get((size, src, xm), ())
                only_t = ONLY_TARGETS.get((size, src, xm))
                for tgt in SRC_TARGETS[src]:
                    if only_tgt and tgt not in only_tgt:
                        continue
                    if only_t is not None and tgt not in only_t:
                        continue
                    if tgt in SIZE_SKIP.get(size, ()) and tgt not in extra:
                        continue
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
        base = BASE_MODEL.get(size, "Qwen/Qwen3-4B-Base")
        job = f"xfer_{size}_{src}_{xm}_to_{tgt}_s{seed}"

        if kind == "nopassage":
            ck = resolve_ckpt(TRAIN_DS[src], pref, step)
            if ck is None:
                print(f"[skip] no checkpoint for {pref} @step{step}")
                continue
            task = "gsm8k" if tgt in ("gsm8k", "gsmsym") else "mcq_nopassage"
            env = (f"ALL,TASK={task},BASE_MODEL={base},CHECKPOINT_DIR={ck},"
                   f"OUTPUT_CSV={csvp},COT_SAMPLES=8,RUN_NAME={pref},STEP=step_{step:05d}")
            if dpath:
                env += f",DATA_PATH={dpath}"
            script = "scripts/eval/run_eval_nopassage.sh"
        else:
            env = (f"ALL,TRAIN_DS={TRAIN_DS[src]},RUN_PREFIX={pref},DATASET_NAME={dsn},"
                   f"DATA_PATH={dpath},OUTPUT_CSV={csvp},COT_SAMPLES=8,COT_EVAL_ONLY=1,"
                   f"MIN_STEP={step},MAX_STEP={step}")
            if tgt in LONG_CTX:   # long-context: YaRN + raised ctx (plain numbers, no commas)
                mml, factor = LONG_CTX[tgt]
                env += f",MAX_MODEL_LEN={mml},ROPE_YARN_FACTOR={factor}"
            script = "scripts/eval/auto_curve_eval.sh"

        cmd = ["sbatch", f"--job-name={job}", f"--export={env}"]
        if BAD_NODES:      # a flatsimpl->cosmosqa eval once hung 90min on 03-018-14-0 writing nothing
            cmd.append(f"--exclude={BAD_NODES}")
        cmd.append(script)
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
