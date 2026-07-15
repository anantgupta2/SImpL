# STATUS — live tracker (2026-06-08)

Canonical findings: **`FINDINGS.md`** (read that first). Every job: `LEDGER.md`.
Branch `simpl-paper-experiments`.

## Where we are
- In-domain + transfer: understanding objective is **redundant with more-cot** (the control). Tie everywhere.
- **SPICE/curriculum CLOSED (2026-06-08):** the +0.84pp was epochs (7 vs 3), not curriculum. The matched-step
  2×2 (`selection {frontier,rotate} × understanding {on,off}`, K=1, ep12, 3 seeds, full-curve avg@8) shows
  curriculum nil-to-negative and understanding a tie in-domain (+0.74pp, noise) AND in RACE transfer
  (rotate/simpl 0.8096 vs rotate/cot 0.8080 peak — both on baseline 0.807). No positive signal left.
- **DECISION PENDING:** user chose "write up, decide later." Top pivot candidate = verifiable process/sub-task
  rewards on ProofWriter-d5 (real headroom; signal not redundant with final-answer correctness). See FINDINGS §9.

## Active configs (configs/qwen/main/) — superseded ones moved to main/archive/
- baselines: `cot-lr8e6{,-n50,-n100}`, `simpl-lr8e6{,-n50,-n100}`
- warm-start bases on disk: `oat-output/staged-bases/{lsat,race}-cot-merged_seed{42,24,36}`
- controls: `{lsat,race}-cottrained-morecot` (more-cot), `{lsat,race}-staged-cot`
- understanding arm: `{lsat,race}-staged-simpl-letter`
- SPICE: `{lsat,race}-spice-letter`

## In flight / next
- Nothing training. **Awaiting direction on the pivot** (see FINDINGS §9). The 2×2 (configs
  `lsat-2x2-{cot-all,simpl-all,cot-front,spice}`) + its curves are done; cot-all seed24 transfer
  curve is 5/13 (resubmit `auto_curve_eval.sh` — it resumes — only if the exact mean is wanted).

## Eval recipe
avg@8 (`COT_SAMPLES=8`). `auto_eval_chain.sh` (single-ckpt, glob-resolve dir, afterok train).
**Curves:** `auto_curve_eval.sh` (embers/2h, all checkpoints → one CSV, flushes per step, resumes;
`REDO_ALL=1` to re-eval). MUST keep `--reasoning/answer_max_tokens 1024` (defaults 384/256 truncate
RACE → fake-low). Always compare against the **more-cot control**, never just the base.
