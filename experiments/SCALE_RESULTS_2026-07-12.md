# SImpL — scale + convention + ablation results (2026-07-12)

Consolidated record of the 8B/1.7B scale study, the convention analysis, the α=128 adapter
experiment, and the rollout-split ablations. Companion to `SWEEP_2026-07-02.md` (original 4B
finals) and `FINDINGS.md`. Reproduce every number with `scripts/eval/aggregate_finals.py`
(env `AGG_DEV_DIR`/`AGG_TEST_DIR` select the size's folder: `evaluations/final_{8b,1p7b}/`;
4B uses the default `finals_{dev,test}`).

## Methods
- **cot16 (cotn16)** = compute-matched CoT control (reasoning_num_samples=16).
- **flatsimpl** = simpl_no_bias + flatten_cot (1 understanding prompt/question), u=c=8.
  RACE uses **flatsimplv3** = flatsimpl + v3 no-summarization prompt + zero_understanding_on_truncation.
- **u4c12 (flatsplit-u4c12)** = the split variant: 4 understanding + 12 cot rollouts/passage
  (`simpl_split_oat.py`, correct per-group GRPO advantages via grp_start). RACE = flatsplitv3-u4c12.
- HP (all): lr2e-5, clip(max_norm)=0.5, β=0.02, temp1.0, tbs128, constant LR, LoRA r64/α256.
  8B uses clip0.5; 1.7B uses clip1.0/lr8e-5 (dev-best per size). 3 seeds (123/234/345).

## CANONICAL CONVENTION (decided 2026-07-12)
**Single-step dev-argmax, PER-SEED**: each run picks its own best-dev checkpoint and we read test
there (the checkpoint you'd actually deploy), averaged over seeds. This replaced the rolling-3
seed-avg-center convention, which *undersells* the understanding methods because 8B cot16's dev
peaks early then declines while simpl holds/rises — a single averaged-dev step near cot16's peak
flatters cot16. We validated across ~8 conventions (center/last/tailavg/targmax/seed-tmax/
dev-argmax-savg/per-seed/window-max); single-step dev-argmax was chosen as it matches practice.

## HEADLINE: DEVARGMAX(per-seed) Δ vs cot16

| Dataset | Method | 1.7B | 4B | 8B |
|---|---|---|---|---|
| **LSAT** | flatsimpl | −0.3 (floored) | **+1.1** | +0.5 / +0.4 (long) |
| | **u4c12** | −0.6 (floored) | **+1.0** | **+1.3** / +0.9 (long) |
| **RACE** | flatsimplv3 | −0.5 | **+0.7** | +0.2 |
| | **u4c12** | −1.3 | **+0.8** | +0.4 |

(4B QUAIL flatsimpl −1.0. 4B failed variants: nbmarg −0.8, nbmargurs1 −1.0, uevery4 −2.4.)

**Story:** the understanding objective — best expressed as **u4c12** — gives a real **+0.4 to
+1.3pp** gain on LSAT and RACE at **4B and 8B**, peaking at 4B and 8B-LSAT (+1.3). It **washes to
~0 at 8B-RACE** and **reverses at 1.7B** (real loss on RACE; LSAT floored at ~20% random). Needs
model capacity to pay off.

**IMPORTANT correction:** an earlier 8B-RACE "understanding +1.2" was an artifact of an
**undertrained cot16 baseline** (ns16 is slow + OOM-prone). Once cot16-long trained fully the gap
dropped to +0.2/+0.4.

## MULTI-CONVENTION ROBUSTNESS (Δ vs cot16, key cells)
Every convention agrees on sign except single-step *seed-avg* dev-select at 8B-LSAT (the artifact).
- **4B-RACE flatsimplv3 = the single cell positive under EVERY convention (+0.3..+0.8)** — most defensible result.
- **u4c12 positive on LSAT under every convention** (+0.5..+1.3), and 4B-RACE.
- 8B-RACE ≈ 0; 1.7B-RACE negative under all.
- Tail-averaging is most favorable to simpl at 8B-LSAT (cot16 declines late); symmetric
  best-of-dev-window is least favorable (cot16 recovers via its own best-of-3).

## α=128 ADAPTER EXPERIMENT (4B LSAT)
Halving lora_alpha 256→128 (α/r 4→2) **also halves the effective LR** (confound). Within-α Δ vs cot16:
- **u4c12 +1.4** (held/grew), **flatsimpl −0.4** (collapsed), cot16 & flatsimpl absolute both dropped.
- Stability (cross-seed deploy-std): flatsimpl 1.23→**0.51** (big), u4c12 0.38→0.38 (no change), cot16 0.53→0.66.
- **Decision: stick with α=256.** For u4c12 (our method) there's no deploy-stability gain, and α=256
  wins on absolute accuracy. The stabilization only helped the deprioritized flatsimpl.

## ROLLOUT-SPLIT ABLATION (4B LSAT + RACE, total=16 rollouts, canonical Δ)
Understanding fraction sweep at fixed total=16 (`simpl_split_oat.py` now accepts ANY split incl. 0):

| Split | u : c | frac | 4B-LSAT Δ | status |
|---|---|---|---|---|
| u4c12 | 4:12 | 0.25 | +1.0 | done |
| u8c8 (flatsimpl) | 8:8 | 0.50 | +1.1 | done |
| u12c4 | 12:4 | 0.75 | — | running |
| u16c0 | 16:0 | 1.00 | — | running (understanding-ONLY: no cot training; tests pure understanding→cot transfer) |

Same 4 splits launched on RACE (v3 recipe). u16c0 is the acid test: does understanding-only
training transfer to cot at all when cot rollouts are removed entirely?

## PROTOCOL NOTES / GOTCHAS (scale)
- **8B simpl needs `-long`** (num_prompt_epoch=15, ~540 steps): cot16 converges early & declines,
  but simpl keeps rising past 325. Set **max_save_num≥150** or early checkpoints get auto-deleted
  (LSAT -long CSVs start at step 148 because eval ran post-hoc after deletion).
- **8B-RACE understanding OOMs** (fail_fast catches → scancel → looks like "SIGNAL Terminated"; real
  cause in `fail_fast_<jid>.log`). Guard: **tbspd=4 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True**.
  cot16-long (ns16) also needs it.
- **1.7B-LSAT is uninformative** (all methods ~20% = 5-choice random floor). Use RACE for 1.7B.
- Eval: watchdog `finals_eval_watchdog.sh` now runs on **inferno** (was embers). Manifests:
  `evaluations/{scale_finals,finals}_eval_manifest.tsv`. Jobids: `experiments/*_jobids.txt`.
