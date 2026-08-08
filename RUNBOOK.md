# SImpL operational runbook

How to make configs, launch training, launch/aggregate evals, and run the dev-based convergence
selection. Results/findings live in `experiments/` (SUMMARY_*, FINDINGS.md, RESULTS_joint.md); data
splits in `data/SPLITS.md`; this file is the "how to run it" reference.

---

## 0. Environment (hard constraints)

- **Never run on the login node.** CPU work: `$CPU_DEFAULT_SALLOC srun ...`. GPU work: `sbatch`.
- Two venvs (different module loads):
  - **Training** (oat): `module load python/3.10.10 cuda/12.6.1; source ~/r-nisha3-0/oat-env/bin/activate`
  - **Eval / data / aggregation** (llm): `module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate`
    - For one-off python (aggregation), just call `~/r-nisha3-0/llm-env/bin/python`.
- Working dir: `~/scratch/SImpL`.
- SLURM accounts/QOS:
  - **Training → inferno** (reliable, non-preemptible): `--account=gts-schava6-qcf --qos=inferno` (already in the train scripts).
  - **Eval → embers** (cheap, preemptible): `--account=gts-nisha3 --qos=embers` (already in `auto_curve_eval.sh`).
- **Flaky nodes — always exclude** on both train and eval: `--exclude=atl1-1-03-018-14-0,atl1-1-03-020-11-0`.
- Seeds: **123 / 234 / 345** = main 3-seed CI runs; **42** = exploratory / sweeps.

---

## 0b. Repo layout (post-consolidation, 2026-08-08)

Live surface only — everything else is under an `_archive`/`archive` dir and is kept for reference,
not for running.

| path | what |
|---|---|
| `src/algorithm/{cot_only,simpl_no_bias,simpl_split}_oat.py` | the three trainers. `simpl_split` **subclasses** `simpl_no_bias`, so neither can be dropped |
| `src/algorithm/archive/` | `simpl_oat.py` (the pre-0614 rotate trainer) — dead, `--simpl` mode removed from `run_with_config.py` |
| `scripts/run/` | `cot_oat.sh`, `simpl_no_bias_oat.sh`, `simpl_split_oat.sh` |
| `scripts/eval/` | `auto_curve_eval.sh` (the worker), the two watchdogs, `eval_orchestrator`+`eval_worker`, `array_eval`, `run_eval_{nopassage,untrained}`, `transfer_drip`, `sweep_incomplete_evals`, table/aggregation `.py` |
| `scripts/claude/` | dataset builders + `launch_transfers.py` |
| `scripts/_archive/` | superseded launchers/wrappers (incl. `run_eval_final_saved_model.sh` — the live path is `python -m src.eval_saved_models`, called by `auto_curve_eval.sh`) |
| `configs/qwen/{final,sweep}/` | see §2 |

Selection rule used: a config/script is "live" iff a current eval CSV's `run_name` resolves to it, or
a slurm job-name log shows it ran in the June–Aug 2026 window.

## 1. Data (see data/SPLITS.md for the full protocol)

- Train files follow the loader convention `data/<ds>/<split>_<seed>_<num>.jsonl`
  (LSAT seed 142, RACE seed 92). One row = one passage `{example_id, article, questions:[...]}`.
- Nested train subsets: LSAT `train_142_{25,50,100}`, RACE `train_92_{25,50,100}`.
- Eval splits: `data/<ds>/dev.jsonl` (convergence) and `data/<ds>/test.jsonl` (report). ~1:1, passage-
  disjoint from train_100.
- **GOTCHA:** the lsat-ar HF cache is corrupted for oat-env's older `datasets` (`Feature type 'List'
  not found`). Do NOT re-download — preprocess loads an existing `data/lsat-ar/<...>.jsonl` if present
  (`create_or_load_preprocessed_data` checks `os.path.exists`). To make a new train subset, derive it
  from an existing larger file (e.g. `head -N`/nested subset) rather than re-running HF preprocess.

---

## 2. Making a config

Two config folders, both under `configs/qwen/` (the run scripts resolve `configs/<family>/<arg>.json`,
so the arg you pass to `sbatch` is `final/<name>` or `sweep/<name>`):

- **`configs/qwen/final/`** (43) — the runs behind every reported number: `final-8b-*`, `final-1p7b-*`,
  the 4B `flatsplit-*` / `flatsimpl-*` / `cotn16-*` ablations, `race-*`, `quail-*`, `scale100-*`.
  There is no "transfer" config set — transfer/OOD is an *eval* of an already-trained checkpoint (§9).
  The joint LSAT+RACE runs were retired 2026-08-08 (out of the RC + long-context paper scope):
  configs `git rm`'d, evals moved to `evaluations/_archive/joint/` (see its README),
  `data/joint-lsat-race/` deleted, joint entries stripped from `launch_transfers.py`.
- **`configs/qwen/sweep/`** (45) — the LR × clip HP grid (`{cotn16,flatsimpl}-{,1p7b-,8b-}lr*-clip*-b*`)
  read off dev only. **All sweep runs are seed 42** (RUNBOOK convention: 42 = exploratory/sweep,
  123/234/345 = the 3-seed CI runs) — don't mistake them for stray seeds and delete them.

Name-suffix decoder:
- **`-t10`** — `temperature: 1.0`, the *training* rollout temp. On every run and config; not a variant.
  (Eval temp is 0.6 — see [[simpl-eval-conventions]].)
- **`-long`** — identical to the base recipe but `num_prompt_epoch` 9→15 and `max_sgd_steps` 300→600.
  The extra epochs turned out not to be needed; kept as the training-length control.
- **`-short`** — the 8B RACE arms matched to the 4B step budget. These, not `-long`, are what the
  transfer panel deploys (the long cot16 baseline is uneven across seeds).
- **`v3`/`v4`** — understanding-prompt variants (`understanding_prompt_version`), not algorithm changes.

Clone an existing one and edit `oat_args`. Clone pattern:

```python
import json
c = json.load(open("configs/qwen/final/final-8b-flatsplit-u4c12-lr2e5-clip05-b02.json"))
c["oat_args"]["learning_rate"] = 3.2e-5        # lr
c["oat_args"]["max_norm"]      = 1.0           # *** "clip" == max_norm (grad clip). clip05=0.5, clip1=1.0
c["oat_args"]["wb_run_name"]   = "Qwen3-8B-final-lsat50-flatsplit-u4c12-clip1lr32"   # see naming bug below
c["preprocess"]["num_samples"] = 25            # train subset size (25/50/100)
c["oat_args"]["num_prompt_epoch"] = 72         # *** DOUBLE this when you HALVE num_samples (keeps step count ~constant; capped by max_sgd_steps)
json.dump(c, open("configs/qwen/final/<newname>.json","w"), indent=4)
```

Key `oat_args` fields:
- `learning_rate`: both 4B & 8B peak at **6.4e-5** under clip05 ("lr64" runs).
- `max_norm` (= "clip"): clip05 recipe = 0.5; the 8B retune = clip1 = 1.0.
- `beta`=0.02 (KL-to-ref), `reward_scale`=1, `understanding_reward_scale`=2 — the clip05 "clean recipe".
- `reasoning_num_samples`: **8** for cot, **16** for cot16 (the compute-matched control).
- `num_prompt_epoch`: passes over the train set. lsat50 cot=6, nbmarg=36; halve data ⇒ double epoch.
- `max_sgd_steps`: caps training steps (217 in current configs). `save_steps`=4 (checkpoint cadence).
- `train_batch_size_per_device`: **8** normally; **4** for 8B-RACE-nbmarg (avoids the entropy-softmax OOM).
- nbmarg-only: `difficulty_weighting`, `baseline_with_passage`, `flatten_cot=false`, `selection_mode="rotate"`,
  `train_understanding=true`, top-level `"simpl_no_bias": true`.

**NAMING BUG (hit 3+ times):** the eval `RUN_PREFIX` is derived from `wb_run_name`, NOT the filename.
Put the scale ("8b") and recipe in `wb_run_name` itself, or eval globs nothing.

Recipes cheat-sheet (these three trainers are the whole live surface; see §8 for split):
| name | mode | key | script |
|---|---|---|---|
| cot | CoT-only, 8 samples | reasoning_num_samples=8 | cot_oat.sh |
| cot16 | CoT-only, 16 samples (compute control) | reasoning_num_samples=16 | cot_oat.sh |
| nbmarg / flatsimpl | SImpL no-bias (8:8) | simpl_no_bias=true | simpl_no_bias_oat.sh |
| u4c12 (headline) | SImpL-split 4 und : 12 cot | simpl_split=true, num_understanding_rollouts=4, num_cot_rollouts=12 | simpl_split_oat.sh |

---

## 3. Launching training

```bash
cd ~/scratch/SImpL
EXCL="atl1-1-03-018-14-0,atl1-1-03-020-11-0"
# cot / cot16:  args = <config-relpath-without-.json> <seed>
sbatch --time=8:00:00 --exclude=$EXCL scripts/run/cot_oat.sh final/final-8b-cotn16-lr2e5-clip05-b02 123
# nbmarg / flatsimpl:
sbatch --time=10:00:00 --exclude=$EXCL scripts/run/simpl_no_bias_oat.sh final/final-8b-flatsimpl-lr2e5-clip05-b02 123
# u4c12 (the headline split recipe):
sbatch --time=10:00:00 --exclude=$EXCL scripts/run/simpl_split_oat.sh final/final-8b-flatsplit-u4c12-lr2e5-clip05-b02 123
# 8B-RACE-nbmarg ONLY: add the OOM env (config already has tbspd=4) and 12h:
sbatch --time=12:00:00 --exclude=$EXCL --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  scripts/run/simpl_no_bias_oat.sh final/final-8b-race-flatsimplv3-short-lr2e5-clip05-b02 123
```
Walltime guide: 4B ~3–7h; 8B LSAT cot ~2.5h / cot16 ~4h / nbmarg ~7h; 8B RACE cot/cot16 ~2.5–4h /
nbmarg ~10–12h. Output run dir: `oat-output/<ds>/<wb_run_name>_<clisuffix>_<seed>_<timestamp>/`
where clisuffix = `cot-only` (cot/cot16) or `simpl-nb` (nbmarg).

---

## 4. Launching evals (curve over all checkpoints → one CSV)

`scripts/eval/auto_curve_eval.sh` evaluates every saved checkpoint (avg@8) into one CSV (rows=steps,
descending), FLUSHES + RESUMES per step (restartable), and **self-heals on embers** (resubmits a fresh
job on preemption/timeout; also a post-run completeness check resubmits if any checkpoint is still
missing — closes the "preemption swallowed by vLLM ZMQError → partial CSV" hole).

```bash
EXCL="atl1-1-03-018-14-0,atl1-1-03-020-11-0"
sbatch --dependency=afterany:<TRAIN_JOBID> --exclude=$EXCL --job-name=ev_<tag> \
  --export=ALL,\
TRAIN_DS="lsat-ar",DATASET_NAME="lsat-ar",\
RUN_PREFIX="Qwen3-8B-Base-clip1lr32-lsat50-nbmarg_simpl-nb_123",\
DATA_PATH="data/lsat-ar/dev.jsonl",\
OUTPUT_CSV="evaluations/<folder>/<run>_s123.csv",\
EVAL_TIME="4:00:00" \
  scripts/eval/auto_curve_eval.sh
```
- **RUN_PREFIX** = `<wb_run_name>_<clisuffix>_<seed>` (the script globs `oat-output/$TRAIN_DS/${RUN_PREFIX}_*`,
  newest match). Derive it from wb_run_name (see naming bug).
- **DATA_PATH**: omit to use the dataset default; set it to `dev.jsonl` for convergence selection and
  `test.jsonl` for the final report. (Run the eval twice — once per split, into separate CSVs.)
- Optional env: `LAST_N_STEPS=N` (only last N ckpts), `MIN_STEP`/`MAX_STEP` (range), `REDO_ALL=1`
  (re-eval all), `COT_SAMPLES=8`, `RESUB_MAX=30`.
- CSV columns: `cot_accuracy,...,run_name,step,checkpoint_path,...` (step like `step_00184`).

---

## 5. Aggregation

Generic seed-averaged aggregation at fixed steps (use `~/r-nisha3-0/llm-env/bin/python`):

```python
import csv, glob, re
def load(p):
    r={}
    for row in csv.DictReader(open(p)):
        try: r[int(re.search(r'(\d+)',row['step']).group(1))]=float(row['cot_accuracy'])*100
        except: pass
    return r
def curve(D,prefix,drop42=True):
    s={}
    for p in sorted(glob.glob(f"{D}/{prefix}_s*.csv")):
        sd=re.search(r'_s(\d+)\.csv',p).group(1)
        if drop42 and sd=="42": continue
        d=load(p)
        if d: s[sd]=d
    return s
def at(seeds,step):
    v=[d[step] for d in seeds.values() if step in d]
    return (sum(v)/len(v), v) if v else (None,[])
# fixed-step view (184/216 were the agreed points)
for m in ["cot","cotn16","nbmarg"]:
    s=curve("evaluations/clip8b", f"clip1lr32-8b-lsat50-{m}")
    print(m, at(s,184)[0], at(s,216)[0], {k:len(v) for k,v in [("n",s)]})
```
Notes:
- `cot_accuracy` is the head we report (both arms scored by their own cot output).
- **Reporting step:** the agreed view is a FIXED step (184 & 216), NOT per-method-best (cherry-pick)
  NOR cot-convergence (cot-biased). The new protocol (§6) replaces this with dev-selected steps.
- Rolling-3 peak of a curve `c` (dict step→acc): `max(range(1,len(ks)-1), key=lambda i:(c[ks[i-1]]+c[ks[i]]+c[ks[i+1]])/3)` where `ks=sorted(c)`.

---

## 6. Dev-based convergence selection (the current protocol)

Goal: pick the reporting step contamination-free.
1. **Dev evals:** run §4 for every run with `DATA_PATH=data/<ds>/final_dev.jsonl` → `evaluations/<...>_dev/...csv`.
2. **CANONICAL convention (as of 2026-07-12): single-step dev-argmax, PER-SEED** — each run deploys
   its OWN best-dev checkpoint and you read test there (the realistic deploy scenario); average over
   seeds. (Earlier we used rolling-3 seed-avg-center; it undersold simpl because 8B cot16 dev peaks
   early then declines. `aggregate_finals.py` reports both, but `DEVARGMAX(ps)` is the headline.)
3. **Test evals:** run §4 with `DATA_PATH=data/<ds>/final_test.jsonl` → `evaluations/<...>_test/...csv`.
4. **Report:** `python scripts/eval/aggregate_finals.py [--step-cap N]`. Columns: DEVARGMAX(ps)=canonical,
   devargmax(sa)=seed-avg variant, seed-tmax=per-seed test ceiling. Env `AGG_DEV_DIR`/`AGG_TEST_DIR`
   point it at a size's folder (`evaluations/final_{8b,1p7b}/{dev,test}`); 8B uses `--step-cap 600`.

Existing trained runs are all valid against these splits (train anchored on their passages; dev/test
disjoint), so no retraining is needed — just the dev+test eval passes.

---

## 8. simpl_split (decoupled understanding/cot rollouts, ANY split)
`src/algorithm/simpl_split_oat.py` (dispatch `--simpl_split` / `"simpl_split": true`; run script
`scripts/run/simpl_split_oat.sh`, clisuffix `simpl-split`). Adds `num_understanding_rollouts` (n_u)
and `num_cot_rollouts` (n_c) — independent per-passage counts; `-1`=default to reasoning_num_samples,
**explicit 0 turns that role off** (u16c0 = understanding-only, c16u0 = cot-only), sum must be ≥1.
Also `understanding_every_k` (run understanding only every k-th round). Correct per-group GRPO
advantages via `info["grp_start"]` flags — verified bit-identical to base GRPO when n_u==n_c.

## 9. Cross-dataset transfer / OOD eval
Eval a trained run's **deployed** (dev-argmax) checkpoint on a dataset it was NOT trained on:
`auto_curve_eval.sh` with `TRAIN_DS=<source ds>` (where the run dir lives), `RUN_PREFIX=<run>`,
`MIN_STEP=MAX_STEP=<deployed step>`, `DATASET_NAME=<target>`, `DATA_PATH=data/<target>/test*.jsonl`,
`COT_EVAL_ONLY=1`. Output → `evaluations/transfer/<label>_to_<target>_s<seed>.csv`. Aggregate with
`scripts/eval/aggregate_transfer.py` (u4c12 vs cot16, Δ per size×source→target). OOD targets used:
ReClor (LSAT-like), QuAIL (RC), ProofWriter-d2 (deductive), ARC-Challenge (science, no passage).

## 10. Zero-shot baselines
`scripts/eval/run_eval_untrained.sh <BASE_MODEL>` with env `DATASET_NAME`, `DATA_PATH`,
`OUTPUT_CSV=evaluations/baselines/base_<tag>_<ds>.csv`, `COT_SAMPLES=8`, `COT_EVAL_ONLY=1`. Runs the
untrained base model (Qwen3-4B/8B-Base) on a dataset's test set → the floor for every transfer cell.

## 11. Drip-submitter (embers ~50-job submit cap)
Large eval batches hit `QOSMaxSubmitJobPerUserLimit`. Pattern: a `nohup bash -c '... for iter; do
<for each planned job>: skip if done (csv has rows) or queued (jobname in squeue); else submit if
squeue count < 46; sleep 120; done'` loop on the LOGIN node (lightweight; only sleeps + sbatch).
See `slurm/{pw,arc}_drip.log`. Plans live in `experiments/*_plan.txt`.

## 12. Datasets
Registered (`src/utils/preprocess_data.py` DATASET_REGISTRY): lsat-ar, race-c, race-high, quail,
reclor, proofwriter-d2/d3/d5, arc-challenge. Add one = registry entry (hf_dataset/subset/format/
split_map) + a `_standardize_<fmt>` (produces `[{article, questions:[{question,options,answer}]}]`)
+ a dispatch branch. **ARC build needs `HF_HUB_DISABLE_XET=1`** (xet CAS 401); evals read the local
jsonl so no HF at eval time. LogiQA dropped (no usable HF source).

---

## 7. Misc gotchas seen this project
- 8B-RACE-nbmarg OOM = entropy softmax over full vocab during policy update → tbspd=4 +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + 12h walltime.
- `merged_eval_models/` under `oat-output/` is regenerable temp (LoRA→full merges); safe to wipe when
  the queue is empty (reclaimed 159G once).
- A stuck training job = frozen in DeepSpeed init on a flaky node, 0 checkpoints, log mtime stale for
  hours → scancel + rerun with `--exclude` (and cancel its dependent eval, resubmit fresh).
```
