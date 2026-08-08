"""Run the FULL deployed two-step pipeline and dump both halves, readably.

understanding_probe.py samples only the understanding. This runs the actual deployed
SImpL path end to end -- generate the understanding, then answer each question CONDITIONED
on that understanding via qa_eval_understanding_only_prompt -- and records both the
understanding and the answer it produces (plus prediction / gold / correctness).

Faithful to eval_saved_models.evaluate()'s understanding_plus_cot mode:
  * understanding is generated GREEDILY (temp 0) with reasoning_max_tokens   (eval line 625)
  * the answer is scored on qa_eval_understanding_only_prompt                 (eval line 646)
  * understanding_with_passage defaults False -> the answer sees ONLY the understanding,
    matching the self-contained (use_baseline_reward=False) understanders.

  python -m src.qualitative.understanding_answer_probe --checkpoint <ckpt> \
      --base_model Qwen/Qwen3-8B-Base --dataset_name race-c \
      --data_path data/cosmosqa/test_42_all.jsonl --out <path.jsonl> --max_passages 50
"""
import argparse
import json
import os
from pathlib import Path

from src.eval_saved_models import (
    CheckpointEvaluator, load_lora_adapter_info, read_adapter_rank,
    infer_instruct_flag, load_eval_examples, _extract_understanding_from_tags,
    extract_boxed_letter,
)
from src.utils.oat_prompt_templates import understanding_prompt, qa_eval_understanding_only_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--understanding_max_tokens", type=int, default=768)
    ap.add_argument("--answer_max_tokens", type=int, default=1024)
    ap.add_argument("--understanding_with_passage", action="store_true",
                    help="pass the passage alongside the understanding at answer time "
                         "(only if training used use_baseline_reward=True)")
    ap.add_argument("--max_passages", type=int, default=None)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=4096)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    lora = load_lora_adapter_info(ckpt)
    ev = CheckpointEvaluator(
        checkpoint_dir=ckpt, lora_info=lora, base_model=args.base_model,
        is_instruct=infer_instruct_flag(args.base_model, None),
        tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
        max_model_len=args.max_model_len, batch_size=1, reasoning_max_items=1,
        reasoning_max_tokens=args.understanding_max_tokens, answer_max_tokens=args.answer_max_tokens,
        trust_remote_code=True, dataset_name=args.dataset_name, cot_samples=1,
        eval_seed=42, understanding_with_passage=args.understanding_with_passage,
        max_lora_rank=read_adapter_rank(ckpt) if lora.has_adapter else None,
    )
    tok = ev.tokenizer

    examples = load_eval_examples(args.data_path, "test", "article", "questions", None, None)
    if args.max_passages:
        examples = examples[:args.max_passages]
    print(f"[ua-probe] {len(examples)} passages, dataset={args.dataset_name}, "
          f"with_passage={args.understanding_with_passage}", flush=True)

    # Step 1: understanding (greedy, exactly as the deployed eval).
    u_prompts = [ev._apply_template(understanding_prompt(ex.article, args.dataset_name)) for ex in examples]
    understandings = ev._generate(u_prompts, max_tokens=args.understanding_max_tokens)
    if len(understandings) < len(examples):
        understandings += [""] * (len(examples) - len(understandings))

    # Step 2: answer each question conditioned ONLY on the understanding (greedy).
    a_prompts, flat_meta = [], []
    bodies = []
    for ex, u_raw in zip(examples, understandings):
        body = _extract_understanding_from_tags(u_raw)
        bodies.append(body)
        imp_article = ex.article if args.understanding_with_passage else ""
        for qi, q in enumerate(ex.questions):
            a_prompts.append(ev._apply_template(
                qa_eval_understanding_only_prompt(imp_article, body, q.question, q.options, args.dataset_name)))
            flat_meta.append((len(bodies) - 1, qi))
    answers = ev._generate(a_prompts, max_tokens=args.answer_max_tokens)

    # regroup answers per passage
    per_passage = [[] for _ in examples]
    n_q = n_correct = 0
    for (ex_i, qi), ans in zip(flat_meta, answers):
        q = examples[ex_i].questions[qi]
        pred = extract_boxed_letter(ans, len(q.options))
        correct = int(pred == q.answer)
        n_q += 1; n_correct += correct
        per_passage[ex_i].append({
            "question": q.question, "options": q.options, "gold": q.answer,
            "answer_text": ans, "pred": pred, "correct": correct,
            "n_tokens_answer": len(tok.encode(ans, add_special_tokens=False)),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ubody_toks = []
    with open(args.out, "w") as f:
        for ex, u_raw, body, qs in zip(examples, understandings, bodies, per_passage):
            nt_body = len(tok.encode(body, add_special_tokens=False)) if body else 0
            ubody_toks.append(nt_body)
            f.write(json.dumps({
                "example_id": ex.example_id, "n_questions": len(ex.questions),
                "understanding_raw": u_raw, "understanding": body,
                "n_tokens_understanding": nt_body,
                "closed_tag": "</understanding>" in u_raw.lower(),
                "questions": qs,
            }) + "\n")
    acc = 100.0 * n_correct / max(n_q, 1)
    ub = sum(ubody_toks) / max(len(ubody_toks), 1)
    print(f"[ua-probe] wrote {len(examples)} passages -> {args.out} | "
          f"understanding-body={ub:.0f}tok | answer acc={acc:.1f}% ({n_correct}/{n_q})", flush=True)


if __name__ == "__main__":
    main()
