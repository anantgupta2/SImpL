# SImpL — Handoff (2026-06-14)

Self-contained pickup doc for a new agent. Branch: **`simpl-paper-experiments`**.
Companions: `FINDINGS.md` (older consolidated findings — partly superseded by this), `LEDGER.md` (job log).
**Read this first.**

---

## 0. TL;DR — where we are

**Thesis:** co-training an *understanding* objective alongside CoT (shared weights, RL) extracts
more learning signal per datum than plain CoT — i.e. a **data-efficiency** claim.

**The result that holds (clean, v2):** from **base** Qwen3-4B-Base, matched steps, matched lr,
no warm-up confound — **simpl beats cot in-domain on LSAT-AR**, and the edge is largest when data
is scarce:

| dataset | gap (simpl − cot), converged | n |
|---|---|---|
| LSAT-50 | **+1.6pp** | 3 |
| LSAT-100 | **+1.7pp** | 3 |
| LSAT-200 | ambiguous (+1.8pp at simpl's plateau, tie at far tail; cot not converged) | 2 |

**Honest narrative:** understanding helps **when the task is hard / data is scarce** (LSAT, headroom),
**ties** when CoT already works (RACE, near-saturated 0.83). An earlier "+3.8pp growing with data" claim
was a **single-step cherry-pick artifact** — killed by reading the converged tail. **Transfer (LSAT→RACE/PW)
did NOT work** → we **dropped transfer** and pitch the in-domain data-efficiency result only.

**Right now (in flight):** a *final, clean* comparison block at locked HP (see §4), including an
`understanding_reward_scale` sweep and a stronger RACE prompt test.

---

## 1. Method (the consolidated implementation)

- **simpl** = `src/algorithm/simpl_oat.py` (`--simpl`). Per passage, one step:
  1. generate N understanding rollouts from the passage alone;
  2. score each by answering **all** the passage's questions (difficulty-weighted marginal reward);
  3. select **one** question (`selection_mode=rotate` round-robin, or `random`) and generate N CoT
     rollouts on it (correctness reward).
  → emits 1 understanding GRPO group + 1 cot group of N (contiguous; Dr.GRPO recovers via
  `view(-1, num_samples)`). Single-actor guard for `rotate` (gpus=1).
- **cot baseline** = `src/algorithm/cot_only_oat.py` (`--cot_only`). **Flattened** dataset (one row per
  question — the *standard* way these datasets are trained). So "simpl vs cot" isolates the
  understanding objective. NOTE: simpl uses a *per-passage* dataset, cot a *flattened* one → they hit
  **matched STEPS** at different epoch counts (sized per dataset, see §4).
- Everything else (SImpL_oat/marginal/spice, understanding_only, cot_trl, pair/dense modes) is
  **archived** under `src/algorithm/archive/` and `configs/qwen/main/archive/`. Don't resurrect.
- `run_with_config.py` accepts only `--simpl` / `--cot_only`.

### Reward scaling (important nuance)
- `reward_scale` (global) ≈ an effective-lr multiplier → we **fixed it (=1.0)** and tune lr instead.
- `understanding_reward_scale` is a **separate** knob applied **on top**: understanding reward =
  `base × reward_scale × understanding_reward_scale`; cot = `base × reward_scale`. So `=2` up-weights
  understanding 2× vs cot (≈ trains understanding at 2× effective lr). The established +1.7pp used
  **`=2`** — this is a potential confound, hence the sweep {1,2,4} in flight (§5).

---

## 2. Environment & how to run (NEVER run on the login node)

- Working dir: `~/scratch/SImpL` (= `/storage/scratch1/1/agupta886/...`).
- **Two venvs:**
  - training: `~/r-nisha3-0/oat-env` (py3.10) — `module load python/3.10.10 cuda/12.6.1`
  - eval/data: `~/r-nisha3-0/llm-env` (py3.12) — `module load python/3.12.5 cuda/12.9.1`
    (use llm-env for HF preprocessing — oat-env has a pyarrow `concat_tables(promote_options=...)` bug)
- CPU work: `$CPU_DEFAULT_SALLOC srun bash <script>`. GPU: `sbatch`. Compute nodes have internet
  (vLLM downloads models on load — no separate download step needed).
- **Run scripts** (`scripts/run/`): `cot_oat.sh`, `simpl_oat.sh` — args `<config-under-main/> <seed>
  qwen [<pretrain-override>]`. Default walltime 8h (cot) — **simpl heavy runs need `--time=16:00:00`**
  (400+ step simpl runs time out at 8h). LSAT-200 simpl & long runs = 16h.
- Config = JSON with `preprocess` (dataset_name, seed, num_samples) + `oat_args`. Unknown oat_args
  are dropped with a warning (so old configs still load).

### Eval (`src/eval_saved_models.py` via `scripts/eval/auto_curve_eval.sh`)
- avg@8 (`COT_SAMPLES=8`), **deterministic** (`--eval_seed 42`: seeds vLLM engine + every sampling +
  reseeds python/np/torch on each model load). MUST pass `--reasoning/answer_max_tokens 1024`
  (auto_curve_eval does; the parser defaults 384/256 **truncate RACE CoT → fake-low ~0.58**).
- Writes the CSV **incrementally** (flush per checkpoint) and **resumes** (skips steps already in the
  CSV; `REDO_ALL=1` to force). So a preempted eval just needs requeuing.
- Knobs: `LAST_N_STEPS=N` (only last N ckpts, for slow PW), `MIN_STEP`/`MAX_STEP` (step range).
- **Partitioning** (`scripts/eval/partition_curve_eval.sh`, env `CHUNKS=4`): splits a dense run into
  K parallel step-range eval jobs (`.partN.csv`) + afterok merge → final CSV. Use for runs with many
  (>~20) checkpoints so no single embers job has to grind the whole curve.
- Env for auto_curve_eval: `TRAIN_DS` (= save subdir, e.g. lsat-ar), `RUN_PREFIX` (= wb_run_name, the
  run-dir prefix), `DATASET_NAME`, `DATA_PATH`, `OUTPUT_CSV`, `COT_SAMPLES`. It globs the newest
  matching run dir.

---

## 3. Methodology decisions (use these when reporting)

- **Read at CONVERGENCE, tail-averaged** — mean of the last ~3–5 checkpoints, for BOTH arms, at
  **matched step** within a size. NEVER a single cherry-picked step (that gave the false +3.8pp).
- **Dense saves** (`save_steps` ~4–12) so the plateau is visible / stability is judgeable.
- **Show the cot curve plateaus** at the read point — if cot is still climbing (as LSAT-200 cot was),
  "converged" is shaky and the comparison is unfair. Larger data converges later (more steps).
- Compare cot vs simpl at **matched STEPS** (= matched gradient updates), not matched epochs.
- Seeds vary the RL stage only → claim "robust across RL seeds", not "across initializations".

---

## 4. Locked hyperparameters (v2 final)

Tuned on LSAT-50 cot, 3 epochs, dense saves (grid in `evaluations/lsat-ar/hp_grid/`):
- **lr = 3.2e-5**, **ppo_epochs = 4**, **reward_scale = 1.0**.
- Findings: ppo_epochs=4 clearly beats 1/2 (more updates/rollout); **high lr × high ppo_epochs
  collapses** (6.4e-5 + pe4 decays); 3.2e-5 + pe4 is best stable. (reward_scale 2→1 halved effective
  lr; pe 2→4 quadrupled updates/rollout — net ~2× gradient work/rollout vs old runs.)

**Step budgets (matched within size):**
- LSAT-50: cot 6 ep / simpl 36 ep → **216 steps**
- LSAT-100: cot 6 ep (~432) / simpl 25 ep (~300) — compare at common step (~300)
- RACE-100: cot 3 ep / simpl 17 ep → **~205 steps**
- (steps/epoch: cot-flattened ≈ 0.58×N_passages/8×... ; simpl per-passage = N_passages/8. LSAT 50→6,
  100→12, 200→25 simpl steps/epoch. cot 50→36, 100→72.)

---

## 5. What is RUNNING NOW (2026-06-14) → `evaluations/final/`

All at lr3.2e-5 / pe4 / rs1.0, dense saves, **evals are 4-chunk partitioned** (afterok train → partition
→ merge):

- **understanding_reward_scale sweep** (LSAT-50 simpl, **108 steps** = 18 ep, seed 42, dense):
  `final-lsat50-simpl-us{1,2,4}` → `lsat50_simpl_us{1,2,4}_s42.csv`. **Pick the scale here**, then run
  full simpl. (Is the established `=2` load-bearing or a confound vs `=1`?)
- **Final cot baselines (3 seeds, 42/24/36):**
  - `final-lsat50-cot` (216 steps) → `lsat50_cot_s*.csv`
  - `final-lsat100-cot` (~432 steps) → `lsat100_cot_s*.csv`
  - `final-lsat50-cotn16` (216, **N=16 compute-matched ablation**) → `lsat50_cotn16_s*.csv`

**Pending (do after the above land):**
1. Aggregate the scale sweep (converged-tail) → choose `understanding_reward_scale`.
2. Launch **full simpl, 3 seeds**: LSAT-50, LSAT-100, **RACE-100** (RACE uses the new stronger
   understanding prompt — see §6) at the chosen scale → `evaluations/final/`.
3. Aggregate final table: cot vs simpl (+ cot-N16 ablation, + scale) at converged tail, LSAT-50/100 +
   RACE-100.

**Held for user decision:** LSAT-200 final (needs dense saves + cot trained long enough to actually
plateau — cot-200 was still climbing). **8B validation** (re-run winning setups on Qwen3-8B-Base; do a
quick lr re-check first) — the planned finish line for "this part."

---

## 6. RACE — the open sub-question

RACE in-domain is a **tie** (cot ≈ simpl, ~0.82–0.835) under the old understanding prompt — likely
because (a) the old prompt was a generic *summary*, (b) RACE questions are independent (less shared
structure), (c) near-saturation. We **rewrote `race_understanding_prompt`** in
`src/utils/oat_prompt_templates.py` to *pre-solve* (main idea, per-paragraph function, inferences,
author stance, contrasts — "commit, don't summarize"), mirroring the LSAT prompt that works. The
RACE-100 simpl run in §5 step 2 uses it. **If RACE moves above cot → understanding helps on both task
types (strong claim). If still a tie → RACE's structure caps it; lean on LSAT.**

---

## 7. Baselines (untrained, avg@8) — `evaluations/<ds>/baselines/`

| model | LSAT | RACE | PW-d5 |
|---|---|---|---|
| Qwen3-4B-Base | 0.210 | 0.731 | 0.628 |
| Qwen3-8B-Base | 0.254 | 0.813 | 0.714 |
| Qwen3-32B-Base | (rerun) | | |
| OctoThinker-3B/8B-Hybrid-Base | 0.12/0.03 | 0.29/0.08 | 0.18/0.06 |

⚠️ **OctoThinker bases score below random** = a **format failure** (base models don't emit parseable
boxed answers). They need a format-adapted prompt / few warm-up steps before they're usable. Flag.

---

## 8. Datasets

- **LSAT-AR** (MAIN): analytical reasoning, base near floor (~0.21, random 0.20) → headroom → where
  understanding helps. Train sets: `data/lsat-ar/train_142_{50,100,200}.jsonl` (seed 142, nested),
  `train_42_all` (272). Test: `test_42_all.jsonl` (40 passages, 230 q). ~5.8 q/passage.
- **RACE-C**: reading comprehension, near-saturated (base 0.73→0.83). `train_142_100`, `test_42_all`
  (708 q). ~5.5 q/passage.
- **ProofWriter-d5**: True/False/Unknown deduction (boxed MC). `train_142_100` (created seed 142),
  `test_42_300`. **~24 q/passage** (heavy for simpl understanding-scoring; eval is slow → use
  `LAST_N_STEPS`). Transfer-only so far.

---

## 9. Gotchas / lessons (don't relearn these)

- **Single-step reads lie** — always converged-tail. (The +3.8pp LSAT-200 artifact.)
- **reward_scale ≈ lr** (redundant); `understanding_reward_scale` is the real, separate up-weight knob.
- **Heavy simpl runs (≥~400 steps) time out at 8h** → `--time=16:00:00`. If one times out, its
  partial checkpoints are usable (eval them) or retrain longer.
- **Eval default tokens (384/256) truncate RACE** → always 1024. Evals are deterministic now.
- **Dense-save evals get preempted as one big job** → use the 4-chunk **partition** tooling.
- **HF preprocessing**: run in **llm-env** (oat-env pyarrow bug). Compute nodes have internet.
- **`run_eval_untrained.sh`** reads `BASE_MODEL` from env now (it used to hardcode Qwen2.5-7B — that
  bug silently evaluated the wrong model).
- **Disk:** old run checkpoints were a huge waste — `oat-output/archive_pre0610` models were **deleted**
  (~226G freed); `evaluations/archive_pre0610` (CSVs) **kept** as the record. Keep `oat-output/staged-bases/`
  (warm-start merges, if ever needed).
- Octo/format: base models that don't emit boxed answers score ~0 — not a capability signal.

---

## 10. Quick orientation commands

```
squeue -u $USER -o "%.10i %.18j %.8T %M %R"          # what's running
ls evaluations/final/                                 # latest results land here
tail experiments/LEDGER.md                            # every job submitted
# aggregate a curve at converged tail (mean of last 3 ckpts), matched step within size — see prior
# python snippets in chat history; read cot_accuracy column from the *_s<seed>.csv files.
```

---

# RUNBOOK — how to run everything (updated 2026-06-21)

## 0. Environment (hard rules)
- **Never run on the login node.** GPU work → `sbatch`; CPU/data work → `$CPU_DEFAULT_SALLOC srun ...`.
- Two venvs: **oat-env** (training) `module load python/3.10.10 cuda/12.6.1; source ~/r-nisha3-0/oat-env/bin/activate`;
  **llm-env** (eval + HF data prep) `module load python/3.12.5 cuda/12.9.1; source ~/r-nisha3-0/llm-env/bin/activate`.
- HF dataset preprocessing **must** run in llm-env (oat-env pyarrow bug). Compute nodes have internet.
- Working dir `~/scratch/SImpL`. Branch `simpl-paper-experiments`.

## 1. Configs  (`configs/qwen/main/*.json`)
One JSON per run: `{preprocess:{dataset_name,num_samples,seed}, oat_args:{...}}`. Key knobs:

| knob | meaning |
|---|---|
| `pretrain` | base model (Qwen/Qwen3-4B-Base, Qwen/Qwen3-8B-Base, …) |
| `learning_rate` (lr) | e.g. 3.2e-5 |
| `num_ppo_epochs` (pe) | gradient passes per rollout (2 = locked) |
| `reward_scale` (rs) | scales reward {0,1}→{0,rs}; in Dr.GRPO = advantage/grad-magnitude knob (rs=2 = headline) |
| `understanding_reward_scale` (us) | simpl only; understanding reward ×= rs×us |
| `beta` | KL-to-ref coefficient. **−1 sentinel → default 0.04**; set explicitly to sweep |
| `max_norm` | grad-norm clip (default 1.0; we tested 0.5) |
| `reasoning_num_samples` (rns) | CoT samples/group (8 std; 16 = cot-N16 control; 4 = cheaper simpl) |
| `num_prompt_epoch` | controls steps: n=50 → 36 (simpl)/6 (cot-flattened) ≈ 216 steps; n=100 → ~432 |
| `selection_mode` | simpl = `rotate` (K=1 question/passage/epoch) |
| `save_steps` | 4 (standardized) |

Make a new config in Python by cloning a sibling and overriding `oat_args` + `preprocess` + `wb_run_name`.

## 2. Training
```
# cot:    sbatch [--time=Hh] scripts/run/cot_oat.sh   main/<config-name> <seed> qwen
# simpl:  sbatch [--time=Hh] scripts/run/simpl_oat.sh main/<config-name> <seed> qwen
```
- Args: `$1`=config (relative to `configs/qwen/`, so `main/<name>`), `$2`=seed, `$3`=family(qwen).
- Default qos=inferno; cot 5h / simpl 8h — **override `--time=12:00:00` for full 216-step / 8B runs.**
- **Run dir** lands at: `oat-output/<dataset_name>/<wb_run_name>_<cot-only|simpl-oat>_<seed>_<timestamp>/`
  (launcher appends `_cot-only_<seed>` or `_simpl-oat_<seed>` to the config's `wb_run_name`).
  ⚠️ The eval `RUN_PREFIX` must match this **exactly** (config filename ≠ wb_run_name — check the JSON).

## 3. Evals  →  **use inferno + resume for curve evals** (embers preemption kills them)
Eval set paths: LSAT **joint 461-Q** `data/lsat-ar/eval_joint_42_all.jsonl`; RACE **708-Q** `data/race-c/test_42_all.jsonl`.
```
sbatch --account=gts-schava6-qcf --qos=inferno --time=8:00:00 --requeue \
  --export=ALL,TRAIN_DS=<lsat-ar|race-c>,\
RUN_PREFIX="<wb_run_name>_<cot-only|simpl-oat>_<seed>",\
DATASET_NAME=<lsat-ar|race-c>,DATA_PATH=<eval-set path>,\
OUTPUT_CSV="evaluations/<dir>/<name>.csv",COT_SAMPLES=8 \
  scripts/eval/auto_curve_eval.sh
```
- `auto_curve_eval.sh` globs the run dir by `RUN_PREFIX_*`, evals **every** checkpoint into one CSV
  (rows = steps), **flushes per checkpoint and resumes** (skips steps already in the CSV).
- Chain after training with `--dependency=afterany:<trainjid>` (afterany so it fires even if a seed fails).
- One CSV column set: `cot_accuracy`, `understanding_plus_cot_accuracy`, `u_and_a_accuracy`, `step`, …
- **Lesson:** embers `--requeue` does NOT recover from preemption here (jobs go PREEMPTED-terminal) and
  4h is too short for 55–100-ckpt simpl evals → they stall. **Inferno + 8h + resume is the reliable path.**
  (`auto_curve_eval.sh` also has an embers self-requeue trap, but inferno is what works.)
- Regenerate the joint LSAT set: `$CPU_DEFAULT_SALLOC srun ... (llm-env) python -m scripts.data.make_lsat_joint_eval`.

## 4. Aggregating
CSVs live in `evaluations/{lsat-re-eval,race-re-eval,bsweep2,final}/...`. Per run, read the curve and take:
- **cot metric** = `cot_accuracy`; **simpl metric** = `max(cot_accuracy, understanding_plus_cot_accuracy, u_and_a_accuracy)` (best eval mode).
- **tail-3** = mean of last 3 checkpoints (converged read); **peak** = best 3-step window (rich-rollout arms
  overtrain → peak ≠ tail; always report both). Seed-average across seeds; Δ = simpl − cot at matched read.
- Minimal pattern:
```python
import csv,glob; from statistics import mean
def tail(p,simpl,k=3):
    d={int(r['step'].split('_')[1]):(max(float(r['cot_accuracy']),float(r['understanding_plus_cot_accuracy']),float(r['u_and_a_accuracy'])) if simpl else float(r['cot_accuracy'])) for r in csv.DictReader(open(p))}
    ks=sorted(d); return mean(d[x] for x in ks[-k:])
print(mean(tail(f,True) for f in glob.glob("evaluations/bsweep2/lsat50-simpl-*_s*.csv")))
```
- Check completeness first: CSV step-count vs `ls <run_dir>/saved_models | grep -c step_`. Resubmit incompletes (§3).

## 5. Gotchas seen this project
- **RUN_PREFIX mismatch** → eval globs nothing, exits instantly ("no run dir"). Verify the run-dir name.
- **embers preemption** silently strands evals; some compute nodes (e.g. atl1-1-03-018-14-0) hang jobs — resubmit elsewhere.
- **beta** is overridden in code unless set in config (sentinel −1 → 0.04).
- **RACE-50 data** uses seed 92 (`train_92_50.jsonl`); on-the-fly race preprocessing fails in oat-env.
- Single H200 fits **8B LoRA** training+rollout fine (~7h, no OOM).
