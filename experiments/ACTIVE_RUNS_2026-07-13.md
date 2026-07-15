# SImpL — active runs & where everything lives (2026-07-13)

Live tracker of everything launched in the scale/convention/ablation/transfer phase. Canonical
results write-up: `SCALE_RESULTS_2026-07-12.md`. Aggregate with `scripts/eval/aggregate_finals.py`
(in-domain) and `scripts/eval/aggregate_transfer.py` (OOD). Canonical selection convention =
**single-step dev-argmax, per-seed** (the deployed checkpoint).

## Eval output locations
| What | Dev/Test CSVs | Aggregate with |
|---|---|---|
| 4B in-domain (LSAT/RACE/QUAIL) + a128 + 4B split-sweep | `evaluations/finals_{dev,test}/` | `aggregate_finals.py` (default dirs) |
| 8B in-domain (LSAT/RACE, incl -long & -short) | `evaluations/final_8b/{dev,test}/` | `AGG_DEV_DIR=evaluations/final_8b/dev AGG_TEST_DIR=evaluations/final_8b/test ... --step-cap 600` |
| 1.7B in-domain (LSAT/RACE) | `evaluations/final_1p7b/{dev,test}/` | `AGG_DEV_DIR=.../final_1p7b/dev AGG_TEST_DIR=.../final_1p7b/test ...` |
| Cross-dataset transfer (OOD) | `evaluations/transfer/` | `aggregate_transfer.py` |
| Zero-shot base-model baselines | `evaluations/baselines/base_{4B,8B}_{ds}.csv` | (read directly) |

## Jobid manifests (training)
- `experiments/finals_2026-07-03.tsv` — 4B finals (cot16/nbmarg/flatsimpl/flatsimplv3/u4c12/…).
- `experiments/scale_finals_2026-07-07_jobids.txt` — 8B & 1.7B finals (incl -long, -short, requeues).
- `experiments/sweep_8b_1p7b_2026-07-05_jobids.txt` — the 8B/1.7B HP sweep (seed 42).
- `experiments/a128_2026-07-12_jobids.txt` — 4B lora_alpha=128 (LSAT).
- `experiments/ablation_2026-07-12_jobids.txt` — 4B split sweep u12c4/u16c0 (LSAT + RACE, v3).

## Eval manifests (watchdog-driven) + watchdogs
- `evaluations/finals_eval_manifest.tsv` — 4B finals + a128 + 4B split-sweep + transfer of finals.
- `evaluations/scale_finals_eval_manifest.tsv` — 8B/1.7B finals incl -long/-short.
- Watchdog: `scripts/eval/finals_eval_watchdog.sh` (now runs on **inferno**), one instance per manifest,
  reschedules every 40min, self-stops when a manifest is fully evaluated. Relaunch:
  `sbatch --export=ALL,MANIFEST=<manifest>,PERIOD_MIN=40 scripts/eval/finals_eval_watchdog.sh`.

## In flight (2026-07-13)
- **4B split sweep** (total=16, vary u:c): u4c12(done +1.0) / u8c8(=flatsimpl, done +1.1) / **u12c4**(running) /
  **u16c0**(running, understanding-ONLY, cot=0) — LSAT and RACE(v3). Methods `flatsplit[-v3]-u12c4/-u16c0`.
- **8B-RACE native short** (epoch9 ~300 steps, matched to 4B budget): `cotn16-short`, `flatsplitv3-u4c12-short`
  (3 seeds) — cleaner matched-budget than capping the -long runs. OOM guard on (tbspd4 + expandable_segments).
- **Transfer/OOD** (deployed checkpoint of u4c12 vs cot16, 4B & 8B, 3 seeds):
  - LSAT-trained → **ReClor** (done), RACE-trained → **QuAIL** (done).
  - Both sources → **ProofWriter-d2** (deductive) and **ARC-Challenge** (science, no passage) — drip-submitting.
  - Drip-submitters (background, fill as embers slots free): `slurm/pw_drip.log`, `slurm/arc_drip.log`;
    deployed-step plan `experiments/pw_transfer_plan.txt`.
- **Baselines**: base 4B/8B zero-shot on all 8 datasets (lsat-ar, race-c/high, quail, reclor,
  proofwriter-d2/d3/d5) + ARC — `evaluations/baselines/`.

## Datasets registered (`src/utils/preprocess_data.py`)
lsat-ar, race-c, race-high, quail, reclor, proofwriter-d2/d3/d5, **arc-challenge** (new).
**LogiQA dropped** (no usable HF source; canonical is a deprecated script dataset). ARC needs
`HF_HUB_DISABLE_XET=1` at *build* time only (evals read the local `data/arc-challenge/test_42_all.jsonl`).

## Gotchas (see SCALE_RESULTS_2026-07-12.md for detail)
- Embers has a ~50 job submit cap → use the drip-submitter pattern for large eval batches.
- 8B simpl needs `-long` (epoch15) + `max_save_num>=150`; 8B-RACE understanding & cot16(ns16) need the OOM guard.
- fail_fast catches OOM and scancels → shows as bare "SIGNAL Terminated"; real cause in `fail_fast_<jid>.log`.
