# SImpL Paper Plan

**Thesis:** There is more RL training signal in a fixed dataset than plain CoT GRPO extracts.
Co-training an auxiliary **understanding** objective (reward = how well the understanding lets the
same model answer the passage's questions) lifts the *primary* CoT task through shared weights —
i.e. **more signal per example from the same data**, not more knowledge and not just more compute.

**Primary metric:** `cot_accuracy` (direct MCQ, no understanding at inference). Secondary:
`understanding_plus_cot`, `u_and_a`.

**Current status:** RACE-C shows `--simpl` cot_acc 0.8475 = +2.0pp over CoT-only mean (≈1.5σ),
**n=1**. LSAT-AR inconclusive (near random floor). Need seeds, a clean second dataset, controls.

---

## 0. Design principles (decide once, apply everywhere)
- [ ] **Held-out dev split** for ALL hyperparameter / checkpoint selection. Never touch `test*.jsonl`
      until final numbers. Carve dev from train (or from RACE-C dev) per dataset.
- [ ] **Tune HPs on the baseline (CoT-only), reuse for SImpL.** Conservative direction — state it
      explicitly in the paper as a fairness guarantee.
- [ ] **Fix a checkpoint-selection rule** up front (e.g. best dev cot_acc, or final step) and apply
      identically to every method. No per-method cherry-picking.
- [ ] **Compute-accounting table**: for each method log #passages, #rollouts, #gradient steps,
      GPU-hours. Needed for the matched-compute claim (Workstream 2) and the SEAL claim (6).
- [ ] **Significance protocol**: report mean ± std over seeds AND a per-question test (paired
      bootstrap or McNemar over the ~708 RACE questions, pooled across seeds). Define it before
      looking at results.

## 1. Hyperparameter search (one seed, CoT-only, on DEV)
- [ ] Search space, prioritized: `learning_rate` (4e-6 now), `num_prompt_epoch`, group size
      (`reasoning_num_samples`/`qa_num_samples`), KL coeff, then `lora_rank`/`alpha`,
      `temperature`, `reward_scale`, `conciseness_penalty_k`.
- [ ] Coarse grid first (handful of runs), then refine the 1-2 sensitive axes. One seed only.
- [ ] Output: a single frozen HP config used for every subsequent run, both baselines and SImpL.
- [ ] Sanity: confirm the tuned CoT-only beats the pre-tuning CoT-only (so the baseline is strong).

## 2. Matched-budget controls (the "epochs won't catch up" claim)
- [ ] **Data-matched, more epochs:** sweep `num_prompt_epoch` for CoT-only on the same passages.
      Plot cot_acc vs epochs. Expect saturation / overfit (small-data regime, 200 passages).
- [ ] **Compute-matched:** CoT-only with extra rollouts / larger groups to match SImpL's total
      rollout+gradient budget. Does CoT close the gap with equal compute? (If yes, the effect is
      compute, not understanding — be ready to report honestly.)
- [ ] **Placebo auxiliary (critical):** co-train CoT with a meaningless auxiliary (understanding
      with shuffled/random reward, OR extra non-understanding rollouts). If placebo ≈ SImpL, the
      lift is regularization, not understanding signal. This defends novelty directly.
- [ ] Deliverable: one figure — cot_acc vs {epochs, compute} for CoT-only vs SImpL on matched data.

## 3. Main result: seed-replicated SImpL vs CoT-only (RACE-C)
- [ ] **≥3 seeds each** for CoT-only and SImpL (current: simpl n=1, cot n=3). Add simpl seeds 36, +2 more.
- [ ] Report all 3 eval modes; primary = cot_accuracy. Mean ± std + significance test.
- [ ] **Ablation ladder** (same seeds where feasible): base → CoT-only → understanding-only →
      combined(naive `cot_and_understanding_oat`) → `--simpl`. Shows each design choice's value.
- [ ] Success bar (pre-registered): SImpL cot_acc beats CoT-only by >1.5pp with p<0.05 across seeds.

## 4. Generalization: rescue LSAT-AR as the 2nd dataset (DECIDED; highest paper risk)
- [ ] **Decision:** LSAT-AR is the second dataset. It's currently inconclusive (near 0.20 random
      floor, only 100 train passages, earlier runs used the wrong generic prompt).
- [ ] Use the existing dataset-specific **logic-constraint understanding prompt** (already in the
      registry), not the RACE summary prompt.
- [ ] **Get the base model off the floor first** — this is the precondition. Options, cheapest first:
      (a) more train passages (scale past 100), (b) more training steps / `num_prompt_epoch`,
      (c) verify CoT-only itself clears random by a clear margin before judging SImpL. If CoT-only
      can't beat the floor, neither methods nor the dataset will show anything.
- [ ] **Pre-registered fallback:** if after the rescue attempt CoT-only LSAT-AR still sits within
      noise of 0.20, fall back to RACE-high (prompt exists, guaranteed headroom) for the main 2nd
      dataset and relegate LSAT-AR to an honest "harder regime" appendix. Decide this on DEV before
      committing seeds.
- [ ] Once base is off the floor: run the full ablation ladder + ≥3 seeds (mirror WS3).

## 5. Mechanism / ablations (why does it work?)
- [ ] **Which fix drives the +2pp?** Ablate SImpL design choices: train/eval parity, dataset-specific
      prompt, fixed metrics, `use_understanding_passage` on/off. (Notes say lift came from
      parity+prompt+metrics, NOT advantage scaling — verify.)
- [ ] **Understanding quality → CoT lift:** correlate understanding reward with cot_acc gain across
      checkpoints/seeds. Show understanding isn't degenerate (not copying passage) — qualitative
      appendix examples.
- [ ] **In-weights vs in-context:** does benefit appear in `cot_only` path (weights) and/or
      `understanding_plus_cot` (inference use)? Report both; current data says both moved.
- [ ] (Optional) the per-task batch-constant advantage scaling (still unimplemented; per-group-std
      version FAILED). Only pursue if it adds a clean ablation point — not required for the main claim.

## 6. SEAL comparison (DECIDED: appendix)
- [ ] **Scaling pathology plot:** accuracy vs #examples/round, SEAL/ReST^EM vs SImpL/GRPO, same
      dataset + budget. SEAL collapses (already observed empirically; zero sequential edits), GRPO
      flat-or-rising. Appendix figure — but write it up cleanly; it's the strongest differentiator and
      can be promoted later if a reviewer pushes on novelty.
- [ ] **Head-to-head on SEAL's own SQuAD** knowledge-incorporation task, matched data budget, primary
      metric per each method's own framing + a common one.
- [ ] Related-work paragraph: honestly state conceptual overlap (generate artifact → reward by
      downstream QA → RL), then carry differentiation on (1) data-efficiency/transfer and (2)
      optimizer scalability — not on the artifact.

## 7. Writing & reproducibility
- [ ] One-sentence thesis (above) + positioning vs SEAL, multitask/auxiliary-task RL, scratchpad/CoT
      RL, self-distillation, RLAIF.
- [ ] Tables: main result (per dataset), ablation ladder, compute accounting. Figures: epochs/compute
      control (WS2), SEAL scaling (WS6), understanding-quality correlation (WS5).
- [ ] Reproducibility: dev-split creation script, seed plumbing, aggregate-seeds + significance eval
      script, frozen configs in `configs/`, GPU-hour log.

---

## Dependency order
0 (design) → 1 (HP) → 3 (main seeds) ∥ 2 (controls) → 4 (2nd dataset) → 5 (ablations) → 6 (SEAL) → 7 (write).
WS2 and WS3 can overlap once the HP config is frozen.

## Biggest risks (in priority order)
1. Only one dataset with a clear effect (WS4).
2. Effect turns out compute/regularization, not understanding (WS2 placebo).
3. n=1 → +2pp not significant after seeds (WS3).
4. SEAL overlap framing (WS6 related work).
