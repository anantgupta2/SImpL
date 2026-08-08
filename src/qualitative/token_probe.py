"""Token-usage + qualitative probe for the "why does it work" analysis.

Deploys ONE checkpoint (a Reasoner/cot16 or an Understander/flatsimpl run's dev-argmax step) and,
for every question in a test set, samples N chain-of-thought answers with the paper's exact
convention (avg@8, temp 0.6, top_p 0.95, max_tokens 1024, seed 42) via the plain-CoT path -- the
same path the deployed model is scored on in the paper. For each of the N samples it records the
generated text, its TOKEN COUNT (via the model tokenizer), the extracted \\boxed{} letter, and
correctness.

Reuses eval_saved_models.Evaluator verbatim (model load, LoRA, prompt, extraction) so the numbers
are identical to the reported ones -- we only add token counting and full-trace dumping, which the
main harness does not do (it samples a subset and counts no tokens).

Output: one JSONL per (model, seed, dataset) at evaluations/qualitative/<tag>.jsonl, one row per
question:
  {example_id, question_index, gold, options_n, n_correct, frac_correct,
   samples: [{text, n_tokens, pred, correct}, ...]}

Usage (invoked by run_token_probe.sh on a GPU node, never the login node):
  python -m src.qualitative.token_probe --checkpoint <ckpt_dir> --base_model Qwen/Qwen3-8B-Base \
      --dataset_name <race-c|quail> --data_path <test.jsonl> --out <path.jsonl> \
      [--extra_instruction "Think step by step and show all your work in detail."]
"""
import argparse
import json
import os
from pathlib import Path

# import the harness we mirror; sys.path is the repo root when run as -m src.qualitative.token_probe
from src.eval_saved_models import (
    CheckpointEvaluator,
    load_lora_adapter_info,
    read_adapter_rank,
    infer_instruct_flag,
    load_eval_examples,
    extract_boxed_letter,
)
from src.utils.oat_prompt_templates import qa_cot_prompt


# The line qa_cot_prompt ends every prompt with. We rewrite it for the interventions rather than
# appending a contradiction (the default already says "think step-by-step", which the Understander
# ignores and the Reasoner obeys).
_STEP_LINE = ("Think step-by-step and return your final answer as exactly one boxed letter "
              "(e.g., \\boxed{A}, \\boxed{B}, etc.).")

_REASON_LINE = (
    "Do not answer immediately. First reason through the passage step by step: state the evidence "
    "that bears on the question, weigh each option against it, and explain your thinking in full. "
    "Only after you have reasoned, give your final answer as exactly one boxed letter "
    "(e.g., \\boxed{A}, \\boxed{B}, etc.).")

_DIRECT_LINE = (
    "Do not explain and do not show any reasoning. Output only your final answer as exactly one "
    "boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.) and nothing else.")

# reason_first: the Understander ignores a plain 'reason first' instruction -- it emits \boxed{}
# at token 0 and only then writes post-hoc reasoning (~50% of the time). So we (a) forbid the box
# until the end AND (b) PREFILL the response with a reasoning lead-in, so the first generated token
# is forced to be reasoning, not the answer. On a base model the prompt is raw text the model
# continues, so appending the prefill seeds the generation mid-reasoning.
_REASON_FIRST_LINE = (
    "Reason through the passage step by step BEFORE committing to an answer. Do NOT write "
    "\\boxed{} anywhere until you have finished reasoning; the boxed letter must be the very LAST "
    "thing in your response.")
_REASON_PREFILL = "Step-by-step reasoning:\n1."

# reason_after: the complement of reason_first. The Understander answers directly; here we let it
# keep answering first but REQUIRE it to then articulate its reasoning. Tests whether eliciting
# (post-hoc) reasoning from the direct-answerer changes the answer at all, and lets us READ the
# reasoning it produces after committing. The boxed answer still comes first, so extraction is
# unaffected (extract_boxed_letter tolerates a trailing re-box that agrees with the first).
_REASON_AFTER_LINE = (
    "First give your final answer as exactly one boxed letter (e.g., \\boxed{A}, \\boxed{B}, etc.). "
    "Then, AFTER the boxed answer, explain your reasoning in full: state the evidence in the passage "
    "that bears on the question and weigh each option against it.")


def build_cot_prompts(examples, evaluator, dataset_name, prompt_mode="default"):
    """One CoT prompt per question. prompt_mode:
      default      -> qa_cot_prompt unchanged (the paper's deployed-eval prompt)
      reason       -> replace closing instruction with a 'reason first' directive (weak: obeyed ~50%)
      reason_first -> reason directive + response prefill so the answer CANNOT come first (strong)
      direct       -> replace it with an explicit 'answer only, no reasoning' directive
    """
    prompts, meta = [], []
    for ex in examples:
        for q_idx, q in enumerate(ex.questions):
            base = qa_cot_prompt(ex.article, q.question, q.options, dataset_name)
            if prompt_mode == "reason":
                base = base.replace(_STEP_LINE, _REASON_LINE)
            elif prompt_mode == "reason_first":
                base = base.replace(_STEP_LINE, _REASON_FIRST_LINE) + _REASON_PREFILL
            elif prompt_mode == "reason_after":
                base = base.replace(_STEP_LINE, _REASON_AFTER_LINE)
            elif prompt_mode == "direct":
                base = base.replace(_STEP_LINE, _DIRECT_LINE)
            prompts.append(evaluator._apply_template(base))
            meta.append((ex.example_id, q_idx, q.answer, len(q.options)))
    return prompts, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cot_samples", type=int, default=8)
    ap.add_argument("--cot_temperature", type=float, default=0.6)
    ap.add_argument("--answer_max_tokens", type=int, default=1024)
    ap.add_argument("--eval_seed", type=int, default=42)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--prompt_mode", default="default",
                    choices=["default", "reason", "reason_first", "reason_after", "direct"],
                    help="default=paper prompt; reason=ask to reason (weak); "
                         "reason_first=ask + prefill so answer can't come first; "
                         "reason_after=answer first, then require reasoning; direct=answer-only.")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    lora_info = load_lora_adapter_info(ckpt)
    is_instruct = infer_instruct_flag(args.base_model, None)

    ev = CheckpointEvaluator(
        checkpoint_dir=ckpt,
        lora_info=lora_info,
        base_model=args.base_model,
        is_instruct=is_instruct,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        batch_size=1,
        reasoning_max_items=1,
        reasoning_max_tokens=768,
        answer_max_tokens=args.answer_max_tokens,
        trust_remote_code=True,
        dataset_name=args.dataset_name,
        cot_samples=args.cot_samples,
        cot_sample_temperature=args.cot_temperature,
        eval_seed=args.eval_seed,
        max_lora_rank=read_adapter_rank(ckpt) if lora_info.has_adapter else None,
        cot_eval_only=True,
    )

    examples = load_eval_examples(args.data_path, "test", "article", "questions", None, None)
    prompts, meta = build_cot_prompts(examples, ev, args.dataset_name, args.prompt_mode)
    print(f"[probe] {len(examples)} passages, {len(prompts)} questions, "
          f"n={args.cot_samples} temp={args.cot_temperature} prompt_mode={args.prompt_mode}",
          flush=True)

    # _generate_n returns per-prompt lists of N texts, exactly as the harness scores avg@N.
    gen = ev._generate_n(prompts, max_tokens=args.answer_max_tokens,
                         n=args.cot_samples, temperature=args.cot_temperature)
    tok = ev.tokenizer

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_q = n_correct_sum = tok_sum = tok_n = 0
    with open(args.out, "w") as f:
        for (example_id, q_idx, gold, n_opt), samples in zip(meta, gen):
            rows = []
            nc = 0
            for text in samples:
                n_tokens = len(tok.encode(text, add_special_tokens=False))
                pred = extract_boxed_letter(text, n_opt)
                correct = int(pred == gold)
                nc += correct
                tok_sum += n_tokens
                tok_n += 1
                rows.append({"text": text, "n_tokens": n_tokens, "pred": pred, "correct": correct})
            f.write(json.dumps({
                "example_id": example_id, "question_index": q_idx, "gold": gold,
                "options_n": n_opt, "n_samples": len(samples), "n_correct": nc,
                "frac_correct": nc / max(len(samples), 1), "samples": rows,
            }) + "\n")
            n_q += 1
            n_correct_sum += nc / max(len(samples), 1)
    print(f"[probe] wrote {n_q} questions to {args.out} | "
          f"avg@{args.cot_samples} acc={n_correct_sum / max(n_q,1):.4f} | "
          f"mean_tokens/sample={tok_sum / max(tok_n,1):.1f}", flush=True)


if __name__ == "__main__":
    main()
