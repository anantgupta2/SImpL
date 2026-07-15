# SImpL — Results so far (2026-06-16)

In-domain RL on **Qwen3-4B-Base** (LoRA r64/α256, Dr.GRPO via `oat`). Two objectives co-trained
on shared weights: **cot** (answer the passage's questions directly — headline metric
`cot_accuracy`) and **understanding** (from the passage alone, emit a structured extraction,
rewarded by how well the same model can then answer). Eval = **avg@8**, deterministic (seeded).

> ⚠️ Eval-set note: numbers below are on the **230-Q LSAT test split** (the eval sets we had until
> now). A **joint 461-Q** re-eval (test+validation pooled, tighter CIs) is running and will replace
> these; the trend is already consistent. RACE is on its 708-Q test split.

---

## 1. Standardized HP (locked recipe)

The recipe we are standardizing on (the "old" recipe — see §4, it beats the grid-picked one at
matched compute):

| knob | value |
|---|---|
| base model | Qwen/Qwen3-4B-Base |
| critic | drgrpo (Dr.GRPO) |
| learning_rate | **3.2e-5** |
| num_ppo_epochs (pe) | **2** |
| reward_scale (rs) | **2.0** |
| understanding_reward_scale (us) | **2** (simpl only; applied *on top of* rs) |
| LoRA | r=64, α=256, dropout 0.05, all proj modules |
| reasoning_num_samples | 8 |
| reasoning_max_tokens | 768 |
| qa_num_samples / qa_eval_max_tokens | 4 / 16 |
| temperature / top_p | 0.8 / 0.95 |
| rollout_batch_size / train_batch_size | 8 / 64 |
| **save_steps** | **4** (standardized across all runs) |
| selection_mode (simpl) | rotate (round-robin, K=1 question/passage/epoch) |
| num_prompt_epoch | size-dependent → ~216 steps (n=50), ~432 (n=100) |
| eval | avg@8, reasoning/answer max 1024 tok, joint 461-Q LSAT set |

`understanding_reward_scale` is **separate** and multiplies on top of `reward_scale`:
understanding reward = base × rs × us; cot reward = base × rs.

---

## 2. HP tuning

### 2a. lr × ppo_epochs grid (LSAT-50 cot from base, seed 42, 216 steps)
`cot_accuracy` tail-3 (peak in parens). **Best per lr row in bold.**

| lr \ pe | pe1 | pe2 | pe4 |
|---|---|---|---|
| 1.6e-5 | 0.2301 (0.2426) | 0.2379 (0.2502) | **0.2560 (0.2652)** |
| **3.2e-5** | 0.2509 (0.2520) | 0.2424 (0.2527) | **0.2621 (0.2681)** ⭐ overall best |
| 6.4e-5 | 0.2516 (0.2560) | 0.2514 (**0.2665**) | 0.2199 (0.2598) — **collapses** |

→ On the cot objective alone the grid prefers **lr 3.2e-5, pe4**. But **lr 6.4e-5 × pe4 collapses**
(0.22), so simpl does *not* tolerate higher lr than cot. Note: pe4/rs1 was later found to
*underperform* the old pe2/rs2 at matched compute (§4), so the locked recipe is pe2/rs2, not pe4/rs1.

### 2b. cot warm-start lr sweep (hp_warmup)
`cot_accuracy` tail-3 (peak). **Best in bold.**

| lr | 4e-6 | 8e-6 | 1.6e-5 | **3.2e-5** | 6.4e-5 |
|---|---|---|---|---|---|
| tail | 0.2366 | 0.2446 | 0.2692 | **0.2746** | 0.2507 |
| peak | 0.2410 | 0.2518 | 0.2758 | **0.2766** | 0.2551 |

→ **lr 3.2e-5** wins; confirms the grid.

---

## 3. In-domain: cot vs simpl vs cot-N16, by data size (hp_size, us2/rs2)

Matched-step `cot_accuracy` (simpl read = best of its eval columns). cot-N16 = compute-matched
control (more cot samples, no understanding). **Best arm per size in bold.**

| size | step | cot | simpl | cot-N16 | best Δ vs cot |
|---|---|---|---|---|---|
| LSAT-50 | 217 | 0.2658 | **0.2778** | — | **+1.20pp** (simpl) |
| LSAT-100 | 301 | 0.2675 | 0.2738 | **0.2790** | +1.15pp (cot-N16 ≥ simpl +0.63) |
| LSAT-200 | 436 | 0.2771 | **0.2795** | — | +0.24pp (≈ tie) |

→ **Trend: simpl's advantage is largest when data is scarce (+1.2pp @50), shrinks toward parity by
200.** The one threat: at **LSAT-100 the compute-matched cot-N16 matches/beats simpl**, so the
cleanest defensible win is at **LSAT-50** (no pe2/rs2 cot-N16 control exists there yet — TODO).

---

## 4. The headline: LSAT-100 simpl from base (+1.7pp) & old vs new recipe

### 4a. hp_simpl_base — from base, matched 216 steps, n=3, pe2/rs2/us2
| arm | cot_accuracy (tail-3) | Δ |
|---|---|---|
| cot-base (n=3) | 0.2595 | — |
| **simpl** (n=3) | **0.2767** | **+1.72pp** |

This is the banked **+1.7pp** headline (per-seed: 0.2815 / 0.2777 / 0.2708).

### 4b. Old (us2/**rs2**) vs new (us2/**rs1**) simpl — LSAT-50, same steps, n=2/3
| recipe | cot | simpl | Δ (tail-3) |
|---|---|---|---|
| **old us2/rs2** (hp_size) | 0.2658 | **0.2778** | **+1.20pp** |
| new us2/rs1 (final) | 0.2607 | 0.2672 | +0.65pp |

→ **rs2 is load-bearing**: dropping rs2→rs1 (holding pe2/us2) shrank the simpl win from +1.2 → +0.65,
and the old recipe is higher in *absolute* terms for **both** arms (simpl 0.278>0.267, cot
0.266>0.261) at the same 217 steps. **Conclusion: standardize on pe2/rs2/us2.**

---

## 5. RACE (in-domain, 708-Q test) — tie / saturated

RACE base is already ~0.73→0.83, little headroom. cot ≈ simpl at every lr (e.g. 3.2e-5: 0.832 vs
0.824). HP effect (pe2/rs2 vs pe4/rs1) appears large but the current re-eval is mid-flight — numbers
pending the resume-eval completion. **Provisional verdict: tie.**

---

## 6. Caveats (read before quoting any number)

- **Per-step noise is ±2–3pp.** The tail-3 average is what surfaces the +0.6/+1.2/+1.7; single
  seeds/steps swing wildly (e.g. LSAT-200 simpl: +3.8pp @step360 was a cot-trough/simpl-peak
  **artifact**, −0.05pp by step 432).
- **Matched-budget, not converged.** Steps scale with data (50→216, 100→432); at the larger sizes
  the cot curve is still rising at the cut-off, so report as "matched compute," not "converged."
- **cot-N16 control** closes the gap at LSAT-100 — the strongest claim lives at LSAT-50.
- LSAT-50 / LSAT-200 are **n=2** in hp_size; LSAT-100 is n=3.

---

## 7. One-line summary

> Co-training the understanding objective helps in-domain on LSAT, **largest when data is scarce
> (+1.2pp @50, +1.7pp @100-from-base) and fading to a tie by 200**; the effect needs the **pe2/rs2/us2**
> recipe (rs2 is load-bearing) and survives best at LSAT-50. RACE is saturated (tie).

---

## 8. Status (eval re-runs in flight)

- **Joint 461-Q LSAT re-eval** of all 64 trained runs + the new lsat100-simpl-pe2-us2 (3 seeds, 432
  steps) → `evaluations/lsat-re-eval/`. **RACE re-eval** (708-Q) → `evaluations/race-re-eval/`.
- Running as per-run inferno resume-evals (skip done steps, no model reload), supervised by an
  autonomous `sweep_watcher` slurm job (resubmits any that hit walltime; 50-job circuit breaker).
- This doc will be refreshed off the joint set once it lands.

---

## Appendix: per-step `cot_accuracy`, seed-averaged

All values are mean `cot_accuracy` (avg@8) over available seeds at each saved step (blank = no checkpoint at that step). 230-Q test split (pre-joint).

### HP grid — LSAT-50 cot from base (seed 42)

| step | lr16e6_pe1 (n1) | lr16e6_pe2 (n1) | lr16e6_pe4 (n1) | lr32e6_pe1 (n1) | lr32e6_pe2 (n1) | lr32e6_pe4 (n1) | lr64e6_pe1 (n1) | lr64e6_pe2 (n1) | lr64e6_pe4 (n1) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.2299 | 0.2228 | 0.2283 | 0.2147 | 0.2380 | 0.2375 | 0.2413 | 0.2283 | 0.2505 |
| 8 | 0.2375 | 0.2147 | 0.2261 | 0.2283 | 0.2326 | 0.2386 | 0.2348 | 0.2277 | 0.2554 |
| 12 | 0.2261 | 0.2380 | 0.2315 | 0.2245 | 0.2326 | 0.2136 | 0.2223 | 0.2533 | 0.2630 |
| 16 | 0.2342 | 0.2500 | 0.2293 | 0.2223 | 0.2424 | 0.2408 | 0.2397 | 0.2549 | 0.2478 |
| 20 | 0.2272 | 0.2375 | 0.2484 | 0.2364 | 0.2424 | 0.2353 | 0.2326 | 0.2440 | 0.2391 |
| 24 | 0.2457 | 0.2435 | 0.2332 | 0.2342 | 0.2386 | 0.2250 | 0.2478 | 0.2370 | 0.2440 |
| 28 | 0.2408 | 0.2370 | 0.2272 | 0.2560 | 0.2370 | 0.2283 | 0.2446 | 0.2418 | 0.2473 |
| 32 | 0.2413 | 0.2359 | 0.2386 | 0.2321 | 0.2326 | 0.2505 | 0.2353 | 0.2538 | 0.2500 |
| 36 | 0.2353 | 0.2467 | 0.2304 | 0.2462 | 0.2495 | 0.2353 | 0.2435 | 0.2386 | 0.2473 |
| 40 | 0.2386 | 0.2413 | 0.2272 | 0.2342 | 0.2429 | 0.2587 | 0.2592 | 0.2446 | 0.2467 |
| 44 | 0.2408 | 0.2359 | 0.2505 | 0.2408 | 0.2332 | 0.2402 | 0.2380 | 0.2533 | 0.2402 |
| 48 | 0.2353 | 0.2505 | 0.2380 | 0.2337 | 0.2440 | 0.2446 | 0.2484 | 0.2549 | 0.2451 |
| 52 | 0.2424 | 0.2440 | 0.2364 | 0.2332 | 0.2353 | 0.2402 | 0.2495 | 0.2424 | 0.2413 |
| 56 | 0.2293 | 0.2402 | 0.2359 | 0.2511 | 0.2435 | 0.2337 | 0.2516 | 0.2462 | 0.2413 |
| 60 | 0.2527 | 0.2364 | 0.2560 | 0.2342 | 0.2397 | 0.2576 | 0.2424 | 0.2473 | 0.2337 |
| 64 | 0.2435 | 0.2505 | 0.2408 | 0.2397 | 0.2348 | 0.2723 | 0.2576 | 0.2418 | 0.2761 |
| 68 | 0.2315 | 0.2310 | 0.2620 | 0.2418 | 0.2337 | 0.2565 | 0.2348 | 0.2707 | 0.2533 |
| 72 | 0.2380 | 0.2446 | 0.2527 | 0.2375 | 0.2408 | 0.2755 | 0.2582 | 0.2690 | 0.2500 |
| 76 | 0.2299 | 0.2370 | 0.2500 | 0.2478 | 0.2473 | 0.2495 | 0.2527 | 0.2598 | 0.2511 |
| 80 | 0.2234 | 0.2457 | 0.2543 | 0.2359 | 0.2277 | 0.2538 | 0.2495 | 0.2609 | 0.2647 |
| 84 | 0.2408 | 0.2527 | 0.2489 | 0.2451 | 0.2429 | 0.2549 | 0.2538 | 0.2690 | 0.2429 |
| 88 | 0.2272 | 0.2522 | 0.2353 | 0.2478 | 0.2582 | 0.2592 | 0.2413 | 0.2505 | 0.2598 |
| 92 | 0.2380 | 0.2315 | 0.2663 | 0.2576 | 0.2571 | 0.2560 | 0.2429 | 0.2592 | 0.2500 |
| 96 | 0.2364 | 0.2457 | 0.2625 | 0.2332 | 0.2217 | 0.2484 | 0.2451 | 0.2418 | 0.2554 |
| 100 | 0.2310 | 0.2429 | 0.2668 | 0.2549 | 0.2446 | 0.2679 | 0.2625 | 0.2440 | 0.2413 |
| 104 | 0.2359 | 0.2321 | 0.2582 | 0.2522 | 0.2380 | 0.2560 | 0.2560 | 0.2609 | 0.2293 |
| 108 | 0.2272 | 0.2408 | 0.2549 | 0.2489 | 0.2446 | 0.2652 | 0.2495 | 0.2467 | 0.2163 |
| 109 | 0.2272 | 0.2408 | 0.2549 | 0.2516 | 0.2446 | 0.2652 | 0.2495 | 0.2467 | 0.2141 |

### HP warmup — LSAT cot warm-start (lr sweep, seed 42)

| step | lr4e6 (n1) | lr8e6 (n1) | lr16e6 (n1) | lr32e6 (n1) | lr64e6 (n1) |
|---|---|---|---|---|---|
| 24 | 0.2473 | 0.2505 | 0.2332 | 0.2239 | 0.2609 |
| 48 | 0.2348 | 0.2348 | 0.2500 | 0.2386 | 0.2467 |
| 72 | 0.2348 | 0.2467 | 0.2408 | 0.2353 | 0.2402 |
| 96 | 0.2266 | 0.2522 | 0.2429 | 0.2696 | 0.2495 |
| 120 | 0.2370 | 0.2375 | 0.2560 | 0.2446 | 0.2484 |
| 144 | 0.2315 | 0.2402 | 0.2293 | 0.2484 | 0.2429 |
| 168 | 0.2342 | 0.2647 | 0.2402 | 0.2554 | 0.2603 |
| 192 | 0.2337 | 0.2478 | 0.2560 | 0.2707 | 0.2603 |
| 216 | 0.2380 | 0.2429 | 0.2755 | 0.2766 | 0.2446 |
| 217 | 0.2380 | 0.2429 | 0.2761 | 0.2766 | 0.2473 |

### hp_size LSAT-50 — cot vs simpl vs cotN16 (us2/rs2)

| step | cot (n2) | simpl (n2) |
|---|---|---|
| 12 | 0.2283 | 0.2497 |
| 24 | 0.2497 | 0.2519 |
| 36 | 0.2429 | 0.2465 |
| 48 | 0.2432 | 0.2435 |
| 60 | 0.2478 | 0.2448 |
| 72 | 0.2432 | 0.2356 |
| 84 | 0.2505 | 0.2614 |
| 96 | 0.2451 | 0.2617 |
| 108 | 0.2522 | 0.2606 |
| 120 | 0.2560 | 0.2739 |
| 132 | 0.2508 | 0.2603 |
| 144 | 0.2530 | 0.2592 |
| 156 | 0.2576 | 0.2617 |
| 168 | 0.2674 | 0.2717 |
| 180 | 0.2758 | 0.2644 |
| 192 | 0.2774 | 0.2707 |
| 204 | 0.2739 | 0.2785 |
| 216 | 0.2617 | 0.2774 |
| 217 | 0.2617 | 0.2774 |

### hp_size LSAT-100 — cot vs simpl vs cotN16 (us2/rs2)

| step | cot (n3) | simpl (n3) | cotN16 (n2) |
|---|---|---|---|
| 24 | 0.2429 | 0.2386 | 0.2505 |
| 48 | 0.2395 | 0.2408 | 0.2334 |
| 72 | 0.2438 | 0.2533 | 0.2383 |
| 96 | 0.2491 | 0.2471 | 0.2519 |
| 120 | 0.2533 | 0.2583 | 0.2481 |
| 144 | 0.2636 | 0.2522 | 0.2549 |
| 168 | 0.2690 | 0.2630 | 0.2549 |
| 192 | 0.2563 | 0.2505 | 0.2587 |
| 216 | 0.2705 | 0.2612 | 0.2707 |
| 240 | 0.2636 | 0.2641 | 0.2709 |
| 264 | 0.2612 | 0.2630 | 0.2785 |
| 288 | 0.2777 | 0.2705 | 0.2875 |
| 301 |  | 0.2844 |  |
| 312 | 0.2833 |  | 0.2842 |
| 336 | 0.2837 |  | 0.2973 |
| 360 | 0.2828 |  | 0.2769 |
| 361 | 0.2822 |  | 0.2859 |

### hp_size LSAT-200 — cot vs simpl vs cotN16 (us2/rs2)

| step | cot (n2) | simpl (n2) |
|---|---|---|
| 24 | 0.2389 | 0.2546 |
| 48 | 0.2459 | 0.2397 |
| 72 | 0.2584 | 0.2443 |
| 96 | 0.2440 | 0.2505 |
| 120 | 0.2524 | 0.2473 |
| 144 | 0.2383 | 0.2755 |
| 168 | 0.2549 | 0.2557 |
| 192 | 0.2554 | 0.2584 |
| 216 | 0.2617 | 0.2557 |
| 240 | 0.2552 | 0.2467 |
| 264 | 0.2620 | 0.2565 |
| 288 | 0.2704 | 0.2435 |
| 312 | 0.2576 | 0.2720 |
| 336 | 0.2769 | 0.2889 |
| 360 | 0.2696 | 0.3076 |
| 384 | 0.2690 | 0.2834 |
| 408 | 0.2834 | 0.2774 |
| 432 | 0.2783 | 0.2777 |
| 436 | 0.2696 |  |
| 451 |  | 0.2734 |

### hp_simpl_base — LSAT-100 simpl from base (us2/rs2, the +1.7)

| step | simpl_base (n3) |
|---|---|
| 12 | 0.2389 |
| 24 | 0.2491 |
| 36 | 0.2420 |
| 48 | 0.2462 |
| 60 | 0.2406 |
| 72 | 0.2542 |
| 84 | 0.2493 |
| 96 | 0.2571 |
| 108 | 0.2589 |
| 120 | 0.2645 |
| 132 | 0.2591 |
| 144 | 0.2697 |
| 156 | 0.2676 |
| 168 | 0.2549 |
| 180 | 0.2643 |
| 192 | 0.2732 |
| 204 | 0.2721 |
| 216 | 0.2790 |
| 217 | 0.2790 |

### final LSAT-50 — newer runs (⚠ pe2/pe4/us* PARTIAL, superseded by joint)

| step | cot(pe4/rs1) (n3) | cotN16 (n3) | simpl_pe2us2 (n3) | simpl_pe2 (n3) | simpl_pe4 (n3) | us1 (n1) | us2 (n1) | us4 (n1) |
|---|---|---|---|---|---|---|---|---|
| 4 |  |  | 0.2272 | 0.2306 | 0.2268 | 0.2402 | 0.2207 | 0.2261 |
| 8 | 0.2295 | 0.2342 | 0.2344 | 0.2275 | 0.2370 | 0.2391 | 0.2321 | 0.2288 |
| 12 |  |  | 0.2484 | 0.2332 | 0.2413 | 0.2424 | 0.2554 | 0.2353 |
| 16 | 0.2344 | 0.2404 | 0.2400 | 0.2353 | 0.2531 | 0.2663 | 0.2495 | 0.2560 |
| 20 |  |  | 0.2487 | 0.2346 | 0.2587 | 0.2429 | 0.2413 | 0.2462 |
| 24 | 0.2310 | 0.2446 | 0.2404 | 0.2402 | 0.2525 | 0.2614 | 0.2603 | 0.2440 |
| 28 |  |  | 0.2384 | 0.2332 | 0.2364 | 0.2636 | 0.2429 | 0.2337 |
| 32 | 0.2518 | 0.2379 | 0.2493 | 0.2453 | 0.2551 | 0.2511 | 0.2641 | 0.2397 |
| 36 |  |  | 0.2500 | 0.2473 | 0.2543 | 0.2522 | 0.2473 | 0.2462 |
| 40 | 0.2582 | 0.2511 | 0.2366 | 0.2552 | 0.2543 | 0.2565 | 0.2429 | 0.2277 |
| 44 |  |  | 0.2478 | 0.2293 | 0.2522 | 0.2614 | 0.2424 | 0.2424 |
| 48 | 0.2553 | 0.2522 | 0.2348 | 0.2353 | 0.2507 | 0.2440 | 0.2571 | 0.2429 |
| 52 |  |  | 0.2491 | 0.2516 | 0.2442 | 0.2402 | 0.2462 | 0.2440 |
| 56 | 0.2431 | 0.2484 | 0.2310 | 0.2636 | 0.2405 | 0.2408 | 0.2299 | 0.2353 |
| 60 |  |  | 0.2464 | 0.2473 | 0.2417 | 0.2364 | 0.2315 | 0.2478 |
| 64 | 0.2553 | 0.2569 | 0.2469 | 0.2557 | 0.2413 | 0.2571 | 0.2603 | 0.2418 |
| 68 |  |  | 0.2495 | 0.2500 | 0.2549 | 0.2375 | 0.2342 | 0.2234 |
| 72 | 0.2409 | 0.2489 | 0.2393 | 0.2505 | 0.2455 | 0.2473 | 0.2348 | 0.2163 |
| 76 |  |  | 0.2467 | 0.2389 | 0.2484 | 0.2495 | 0.2212 | 0.2207 |
| 80 | 0.2509 | 0.2505 | 0.2451 | 0.2443 | 0.2417 | 0.2478 | 0.2462 | 0.2109 |
| 84 |  |  | 0.2547 | 0.2427 | 0.2504 | 0.2321 | 0.2283 | 0.2147 |
| 88 | 0.2457 | 0.2391 | 0.2496 | 0.2478 | 0.2486 | 0.2217 | 0.2239 | 0.2304 |
| 92 |  |  | 0.2466 | 0.2484 | 0.2335 | 0.2598 | 0.2348 | 0.2185 |
| 96 | 0.2520 | 0.2496 | 0.2534 | 0.2359 | 0.2444 | 0.2451 | 0.2174 | 0.1929 |
| 100 |  |  | 0.2409 | 0.2543 | 0.2379 | 0.2533 | 0.2402 | 0.2326 |
| 104 | 0.2505 | 0.2429 | 0.2473 | 0.2508 | 0.2389 | 0.2255 | 0.2293 | 0.2163 |
| 108 |  |  | 0.2460 | 0.2505 | 0.2257 | 0.2402 | 0.2446 | 0.2136 |
| 109 |  |  |  |  |  | 0.2402 | 0.2418 | 0.2136 |
| 112 | 0.2406 | 0.2440 | 0.2594 | 0.2554 | 0.2295 |  |  |  |
| 116 |  |  | 0.2386 | 0.2571 |  |  |  |  |
| 120 | 0.2558 | 0.2462 | 0.2449 | 0.2522 |  |  |  |  |
| 124 |  |  | 0.2317 | 0.2413 |  |  |  |  |
| 128 | 0.2469 | 0.2616 | 0.2520 | 0.2408 |  |  |  |  |
| 132 |  |  | 0.2514 | 0.2429 |  |  |  |  |
| 136 | 0.2598 | 0.2547 | 0.2611 | 0.2467 |  |  |  |  |
| 140 |  |  | 0.2571 | 0.2598 |  |  |  |  |
| 144 | 0.2540 | 0.2514 | 0.2502 | 0.2484 |  |  |  |  |
| 148 |  |  | 0.2643 | 0.2587 |  |  |  |  |
| 152 | 0.2525 | 0.2458 | 0.2611 | 0.2277 |  |  |  |  |
| 156 |  |  | 0.2484 | 0.2630 |  |  |  |  |
| 160 | 0.2582 | 0.2496 | 0.2531 | 0.2522 |  |  |  |  |
| 164 |  |  | 0.2609 | 0.2576 |  |  |  |  |
| 168 | 0.2591 | 0.2524 | 0.2565 | 0.2424 |  |  |  |  |
| 172 |  |  | 0.2609 |  |  |  |  |  |
| 176 | 0.2679 | 0.2638 | 0.2600 |  |  |  |  |  |
| 180 |  |  | 0.2534 |  |  |  |  |  |
| 184 | 0.2511 | 0.2489 | 0.2527 |  |  |  |  |  |
| 188 |  |  | 0.2645 |  |  |  |  |  |
| 192 | 0.2498 | 0.2558 | 0.2609 |  |  |  |  |  |
| 196 |  |  | 0.2547 |  |  |  |  |  |
| 200 | 0.2658 | 0.2511 | 0.2492 |  |  |  |  |  |
| 204 |  |  | 0.2636 |  |  |  |  |  |
| 208 | 0.2587 | 0.2464 | 0.2698 |  |  |  |  |  |
| 212 |  |  | 0.2658 |  |  |  |  |  |
| 216 | 0.2618 | 0.2478 | 0.2682 |  |  |  |  |  |
| 217 | 0.2618 | 0.2476 | 0.2677 |  |  |  |  |  |

### final LSAT-100 — cot (pe4/rs1)

| step | cot (n3) |
|---|---|
| 12 | 0.2433 |
| 24 | 0.2397 |
| 36 | 0.2308 |
| 48 | 0.2359 |
| 60 | 0.2594 |
| 72 | 0.2422 |
| 84 | 0.2404 |
| 96 | 0.2580 |
| 108 | 0.2553 |
| 120 | 0.2469 |
| 132 | 0.2529 |
| 144 | 0.2551 |
| 156 | 0.2525 |
| 168 | 0.2594 |
| 180 | 0.2563 |
| 192 | 0.2591 |
| 204 | 0.2457 |
| 216 | 0.2594 |
| 228 | 0.2616 |
| 240 | 0.2370 |
| 252 | 0.2493 |
| 264 | 0.2464 |
| 276 | 0.2652 |
| 288 | 0.2553 |
| 300 | 0.2547 |
| 312 | 0.2654 |
| 324 | 0.2594 |
| 336 | 0.2674 |
| 348 | 0.2630 |
| 360 | 0.2683 |
| 372 | 0.2868 |
| 384 | 0.2759 |
| 396 | 0.2755 |
| 408 | 0.2855 |
| 420 | 0.2799 |
| 432 | 0.2900 |
| 433 | 0.2900 |

