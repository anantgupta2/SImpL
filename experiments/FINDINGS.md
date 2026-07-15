# SImpL — consolidated findings (updated 2026-07-05)

Self-contained handoff doc. **Canonical current results: `SWEEP_2026-07-02.md`
(FINAL RESULTS section) — that supersedes everything below.** Reproduce with
`scripts/eval/aggregate_finals.py`. Sections 1–5 here are older context (pre-final-sweep);
kept for the mechanism/history, but the numbers in §0.OLD are NOT the current story.

---

## 0. TL;DR (2026-07-05, read this first) — the final-sweep story

After rebuilding the protocol contamination-free (dev = unused train pool, test = old val+test
joined; HP swept clip×lr×beta on cot16+nbmarg, winner lr2e-5/clip0.5/β0.02; 3 seeds; selection
= rolling-3 peak of **seed-averaged** dev, report seed-avg test there), the picture is **no
longer uniformly null** — there is a real, control-beating understanding win when the
understanding is *well-formed*:

| dataset · 4B | best understanding arm | Δ vs compute-matched cot16 | verdict |
|---|---|---|---|
| LSAT | **flatsimpl** (flatten_cot, 1 prompt/Q) | **+1.0pp** (27.0 vs 26.0) | win |
| RACE | **flatsimplv3** (v3 no-summary prompt + zero-on-truncation) | **+0.5pp** (84.2 vs 83.7) | win |
| QUAIL | flatsimpl (no v3 fix yet) | −0.1pp | tie |

The lever that matters is **flatten_cot** (one understanding prompt per *question*, not the
per-passage marginal `nbmarg`, which stays flat) plus, on long-passage datasets, forcing the
understanding to be **inference-only and non-truncated** (RACE v3 prompt + `zero_understanding_
on_truncation`; without it RACE understanding is a lossy truncated summary and *hurts*, −0.4).
Where the understanding is malformed the gain vanishes → the win tracks understanding **content**,
not the extra compute (cot16 is the compute control and loses in both wins). Full table + method
+ caveats in `SWEEP_2026-07-02.md`. **Next:** replicate the sweep at 8B and 1.7B (scale of the
effect — memory `simpl-clip05-scaling` had earlier, pre-protocol hints it may flip with scale).

<details><summary>§0.OLD — pre-final-sweep TL;DR (2026-06-08, superseded)</summary>

The original thesis — *co-training an "understanding" RL objective extracts more learning
signal per datum than plain CoT* — **does not hold under proper controls.** The single
control that settled it was added 2026-06-06: **"more-cot"** — warm-start from a model and
just train CoT *longer* (matched compute). Once you compare against that, **every apparent
understanding gain disappears**, in-domain AND in transfer, on RACE AND LSAT:

| comparison (avg@8 cot_acc) | understanding arm | more-cot control | verdict |
|---|---|---|---|
| RACE in-domain (Δ vs base) | staged-simpl-letter +1.06pp | **+1.01pp** | tie |
| LSAT in-domain (Δ vs base) | staged-simpl-letter +0.91pp | **+1.01pp** | tie |
| LSAT→RACE transfer | staged-simpl-letter 0.8074 | **0.8070** | tie |

The understanding **reward** is redundant with simply doing more CoT RL.

**SPICE/curriculum — also closed (2026-06-08).** SPICE's earlier +0.84pp LSAT in-domain win was
confounded (it ran epochs=7 vs controls' 3). The deciding test was a **2×2 at matched step-rate**:
`selection {frontier-curriculum, rotate=round-robin no-curriculum} × understanding {on, off}`, all
K=1 question/passage/epoch, warm-start, 12 epochs, 3 seeds, full-curve avg@8 (in-domain + RACE
transfer). Result — **both levers are null**:

| 2×2 contrast | in-domain (final) | LSAT→RACE transfer (peak / final) |
|---|---|---|
| understanding (on − off), rotate row | +0.74pp (within noise) | +0.16pp / −0.06pp = tie |
| curriculum (frontier − rotate), cot col | +0.07pp = nil | n/a |
| curriculum w/ understanding (frontier − rotate) | **−0.89pp (hurts)** | n/a |

Once epochs are matched, SPICE's curriculum adds nothing over plain round-robin and *hurts* when
combined with understanding; understanding stays a tie in-domain AND in transfer. The +0.84pp was
**more epochs, not curriculum.** Both the understanding-reward and the curriculum lines are now
closed under proper controls. **No positive signal remains in this family.**

</details>

---

## 1. What SImpL is
RL (GRPO/Dr.GRPO via `oat`, Qwen3-4B-Base, LoRA r64/α256) co-training two behaviours on
shared weights: **cot** (answer directly from passage — the headline metric `cot_accuracy`)
and **understanding** (from the passage alone, emit a structured extraction, rewarded by how
well it lets the same model answer the passage's questions). Modes (entrypoint flags):
`--cot_only`, `--simpl`, `--understanding_only`, `--simpl_marginal`, `--simpl_spice`.
(`--combined` was removed — superseded by `--simpl`.)

## 2. The redundancy result (robust, the main finding)
Mechanistically (GRPO): the understanding only gets gradient when its N samples earn
*different* rewards. With the passage present at QA, the model answers correctly regardless of
understanding quality → rewards clump → ~no gradient. And the understanding reward = the same
answer-correctness signal CoT already optimizes → **redundant**. Every lever we tried to break
this returned in-domain parity or worse, and crucially **ties the more-cot control**:
- ratio up-weighting (`u_rows`), 1:1 (hurts — starves cot), full-output, passage-free (hurts
  transfer), difficulty-weighted "marginal" reward, reward-scale sweep {2,4,5}, direct/brief QA
  + answer-ready understanding ("letter-16"), staged (cot-then-understanding warm-start).
- letter-16 was the best *understanding* variant (RACE uonly +0.47pp, monotone letter>tiny>think)
  but **still < more-cot (+1.0pp)** — i.e. understanding did *less* than just training cot more.

## 3. Transfer (OOD) — also redundant under the control
Earlier "SImpL transfers +1.7pp to RACE" was a **greedy-decode artifact**; avg@8 shrank it to
+0.2pp (overlapping). With the more-cot control: staged-simpl-letter→RACE 0.8074 ≈ more-cot
0.8070 (both +1.4pp over the pre-stage cot base = "more training," not understanding). So the
OOD-transfer story doesn't survive the control either.

## 4. SPICE / curriculum — tested at matched steps, CLOSED (2026-06-08)
`src/algorithm/SImpL_spice_oat.py` (`--simpl_spice`). Per passage in one step: generate
understanding rollouts (co-trained, difficulty-weighted reward) → **reuse the already-computed
direct passage-only accuracy** to pick the **frontier question** (closest to 0.5) → train CoT
**only on that** (K=`num_frontier_questions`, default 1). Two knobs added for the 2×2:
`selection_mode` ∈ {`frontier`, `rotate`, `random`} and `train_understanding` ∈ {true,false}.
`rotate` = the no-curriculum control: K=1 deterministic round-robin through a per-passage shuffled
question order (question[i] on epoch i), so every question gets equal coverage at the **same
step-rate** as frontier (only state lives in the actor → guarded to a single actor process).

**The deciding 2×2 (matched step-rate, ep12, 3 seeds, full-curve avg@8):** earlier SPICE's +0.84pp
was at epochs=7 vs controls' 3. Matched, it vanishes (see §0 table). Means:
- LSAT in-domain final: rotate/cot 0.2587, rotate/simpl 0.2661, frontier/cot 0.2594, frontier/spice
  0.2572. Curriculum nil-to-negative; understanding +0.74pp (within ±1.5pp seed noise).
- LSAT→RACE transfer (corrected eval): rotate/cot mean peak 0.8080 / final 0.8039; rotate/simpl
  0.8096 / 0.8033 — tie, both on the more-cot baseline 0.807. Flat curves wandering 0.80–0.81.
- **Verdict:** the +0.84pp was *more epochs, not curriculum*. rotate/simpl ≈ rotate/cot everywhere;
  its only merit is convenience (scales to full dataset trivially), not performance. Line closed.
- Implementation verified correct: buffer = (K cot + [understanding]) groups of N/passage, contiguous
  `[Nu][Nc…]`; Dr.GRPO doesn't shuffle so `view(-1,num_samples)` recovers groups. cot-only arms ran
  reasoning_num_samples=16 (vs 8) to compute-match the dropped understanding rollouts.

## 5. Inference-mode finding (re: "use understand+answer as the metric")
Same models, three inference modes (avg@8):
- **RACE (strong cot):** u_and_a < und+cot < cot_acc for *every* model — the understanding route
  is **worse** than direct cot. SImpL's edge at u_and_a over more-cot = +0.68pp (noise).
- **LSAT (near floor):** flips — understand+answer modestly edges cot for understanding arms
  (staged u_and_a 0.259 > its cot 0.252). Consistent with "structure helps a weak model, hurts a
  strong one." But magnitudes ~0.5–0.8pp near the 0.20 random floor → not paper-grade.
- Caution: "understand+answer beats cot" is confounded by (a) circularity (trained on it),
  (b) extra test-time tokens, (c) weak-model-only. Needs same-inference + matched-token controls.

## 6. Reliable eval (use for ALL headline numbers)
LSAT 230-Q greedy is too noisy. Use **avg@8**: `--cot_samples 8` in `eval_saved_models`, `COT_SAMPLES=8`
in `run_eval_final_saved_model.sh` (cut LSAT seed-std 3–4x). `STEP=` env on that script evaluates a
specific checkpoint (for curves). `scripts/eval/auto_eval_chain.sh` resolves the run dir by glob at
eval-time (afterok the train job) so we don't hardcode timestamps.

## 7. Datasets / models
- **LSAT-AR**: MAIN. Analytical-reasoning, joint-constraint understanding, base near floor (~0.24,
  random 0.20). avg@8 on 230 test Q. Where SPICE's signal appears.
- **RACE-C**: easy (~0.85), independent Qs. In-domain tie, strong-cot regime.
- ProofWriter-d5: rule-deduction, good headroom; training on it doesn't transfer (archived focus).
- Use **base** models (instruct cratered on LSAT). 32B only ~0.30 on LSAT → not a capacity issue.

## 8. Key infra / lessons
- **fail_fast guard** (`src/utils/fail_fast.py`): any training exception → scancel the job + exit,
  so a crash frees the GPU instead of hanging (it once burned 4h). Wraps learner.run + actor.step.
- **Staged warm-start**: merge a cot LoRA adapter into the base (`scripts/run/merge_staged.sh` →
  `oat-output/staged-bases/{lsat,race}-cot-merged_seed{42,24,36}`), then train a fresh adapter.
  Merged-base path has no "qwen" → MUST set `use_fused_lm_head:false` in the config (else Qwen3
  lm-head crash). Launchers `simpl_marginal_oat.sh` / `cot_oat.sh` / `simpl_spice_oat.sh` take an
  optional 4th arg = pretrain override.
- Controls that earn their keep: tune the baseline LR; **more-epochs/more-compute control**;
  avg@8 (not greedy). Every premature claim died to one of these.
- afterok fails on already-COMPLETED jobs (submit direct); salloc needs `srun`/sbatch to bind a GPU
  (plain salloc bash fell to CPU); embers QoS is preemptible (use inferno for must-finish jobs).
- **Curve-eval pipeline (`scripts/eval/auto_curve_eval.sh` + `src/eval_saved_models.py`), fixed
  2026-06-08:** (a) the all-steps job MUST pass `--reasoning_max_tokens 1024 --answer_max_tokens
  1024`; the parser defaults (384/256) truncate RACE CoT before the boxed answer → fake-low ~0.58
  (looked like a transfer collapse, was an eval bug). (b) `eval_saved_models` now **flushes the CSV
  after every checkpoint** (`write_summary(..., merge=False)` / `_write_curve`), and (c) **resumes**
  — skips steps already in `--output_csv` unless `--redo_all`. So the curve job runs on embers/2h
  (sub-2h = preempt-safe; killed mid-way keeps done rows; resubmit fills the gaps).

## 9. OPEN / NEXT
The understanding-reward AND curriculum lines are both closed under matched controls (§0, §2–§4).
What remains:
1. **DECISION PENDING (user, 2026-06-08): pick a new direction.** The understanding/SPICE family is
   exhausted; chose "write up, decide later." Candidate pivots discussed:
   - **Verifiable sub-task / process rewards on ProofWriter-d5** (top pick): reward *checkable*
     intermediate derivation steps so the extra signal is genuinely new, not a re-encoding of
     final-answer correctness (which is exactly why understanding kept collapsing into the CoT
     gradient). ProofWriter-d5 has real headroom, unlike near-saturated LSAT/RACE.
   - (rejected-for-now) scale rotate/simpl on the full dataset — only a convenience win, not perf.
2. **The honest fallback paper** is the matched-compute negative result: redundancy mechanism +
   the control tables (§0/§2/§3/§4). This is now well-supported and reproducible.
3. (cleanup) cot-all seed24 transfer curve was incomplete (5/13) when last aggregated — resubmit
   `auto_curve_eval.sh` (it resumes) if the exact mean is wanted, but it won't change the tie.

---

## Update 2026-06-22 — clip05 SImpL recipe is scale-dependent (8B kills the win)

The current live story (post-06-08) is the **clip05 "clean recipe"** where, at a shared
regularization-heavy HP (β=0.02, max_norm=0.5, rs=1, us=2), SImpL beats CoT-only on the cot head.
Full mechanism + generality table now in **`RESULTS_joint.md`** (the live results doc; this section
is just the headline). Scaling result:

| clip05, tail-3 cot_acc, 3 seeds | CoT | SImpL | Δ |
|---|---|---|---|
| 4B · LSAT-50 | 24.80 | 26.12 | **+1.33** |
| 4B · RACE-50 | 82.64 | 83.2–83.4 | **+0.5…0.8** |
| 8B · LSAT-50 | 28.61 | 28.63 | **+0.03 (tie)** |

**The advantage vanishes at 8B** — both arms converge to ~28.6% and SImpL's seed variance widens.
Consistent with the whole 06-08 picture: the understanding objective behaves like an auxiliary
*regularizer* that helps a **weaker / unsaturated** base, not a universal source of extra signal.
Honest framing for the paper: "more signal per datum **for smaller models**," scale-dependent.

**In flight (06-22):** `simpl_no_bias` ablation (flattened data like cot, plain single-question
understanding reward — no difficulty weighting / baseline / rotate) running at clip0.5 & clip1.0,
LSAT-50, 3 seeds → `evaluations/nobias/`; RACE-50 on 8B (old simpl/cot, clip05) → `evaluations/race8b/`.
Both isolate whether the 4B win is the understanding *objective* vs. the difficulty-weighting confound,
and whether RACE shows the same 8B collapse as LSAT.

---

## Update 2026-06-27 — 4B understanding-win confirmed; 8B = compute-win (HP not settled)

Full snapshot in **`SUMMARY_2026-06-27.md`**. Headline (cot_acc @216, Δ vs cot, nbmarg = canonical recipe):

| | cot16 Δ | nbmarg Δ |
|---|---|---|
| 4B LSAT-50 (n=3) | −0.06 | **+0.82** |
| 4B RACE-50 (n=3) | −0.03 | **+1.24** |
| 4B QuAIL-50 (n=3) | +0.25 | **+0.50** |
| 8B LSAT-50 clip1/lr32 (n=4) | **+1.97** | +0.57 |
| 8B RACE-50 clip1/lr32 (n=3) | +0.08 | −0.55 |

**4B:** nbmarg beats cot AND the compute control cot16 on all 3 datasets → the gain is the understanding
objective, not extra samples. **8B (retuned clip1/lr32):** cot16 overtakes nbmarg → at 8B the edge is
raw compute, not understanding (cot16 win is real, 3/4 seeds ≥30.8, not a hot seed). clip1/lr32 raised
the 8B floor (cot 28.99→29.46) but produced no SImpL win. **Next:** proper 8B HP sweep (clip×lr); the
8B optimum is not yet found. 4B LSAT-25 low-data is n=1, inconclusive.
