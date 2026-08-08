"""Sample the model's actual UNDERSTANDINGS (not the deployed CoT).

Everything so far probed the plain-CoT path (where the RACE model answers directly). This instead
prompts the model with its training-time understanding_prompt and captures what it writes inside
<understanding>...</understanding> -- the intermediate representation itself. Lets us read what a
"good understanding" looks like after training, and test whether the RACE understander's terseness
is because its understandings pre-compute the answers.

Generates one understanding per passage (x N samples), same sampling convention as everything else
(temp 0.6, top_p 0.95, seed 42). Records the raw text, extracted <understanding> body, and token
counts of both.

  python -m src.qualitative.understanding_probe --checkpoint <ckpt> --base_model Qwen/Qwen3-8B-Base \
      --dataset_name race-c --data_path data/race-c/final_test.jsonl --out <path.jsonl> --n 2
"""
import argparse
import json
import os
from pathlib import Path

from src.eval_saved_models import (
    CheckpointEvaluator, load_lora_adapter_info, read_adapter_rank,
    infer_instruct_flag, load_eval_examples, _extract_understanding_from_tags,
)
from src.utils.oat_prompt_templates import understanding_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2, help="understandings sampled per passage")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_tokens", type=int, default=768, help="reasoning_max_tokens at train time")
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
        reasoning_max_tokens=args.max_tokens, answer_max_tokens=args.max_tokens,
        trust_remote_code=True, dataset_name=args.dataset_name, cot_samples=args.n,
        cot_sample_temperature=args.temperature, eval_seed=42,
        max_lora_rank=read_adapter_rank(ckpt) if lora.has_adapter else None, cot_eval_only=True,
    )

    examples = load_eval_examples(args.data_path, "test", "article", "questions", None, None)
    if args.max_passages:
        examples = examples[:args.max_passages]
    prompts = [ev._apply_template(understanding_prompt(ex.article, args.dataset_name))
               for ex in examples]
    print(f"[u-probe] {len(examples)} passages, n={args.n}, dataset={args.dataset_name}", flush=True)

    gen = ev._generate_n(prompts, max_tokens=args.max_tokens, n=args.n, temperature=args.temperature)
    tok = ev.tokenizer

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    full_toks, body_toks = [], []
    with open(args.out, "w") as f:
        for ex, samples in zip(examples, gen):
            rows = []
            for text in samples:
                body = _extract_understanding_from_tags(text)
                nt_full = len(tok.encode(text, add_special_tokens=False))
                nt_body = len(tok.encode(body, add_special_tokens=False)) if body else 0
                full_toks.append(nt_full); body_toks.append(nt_body)
                rows.append({"raw": text, "understanding": body,
                             "n_tokens_full": nt_full, "n_tokens_understanding": nt_body,
                             "closed_tag": "</understanding>" in text.lower()})
            f.write(json.dumps({"example_id": ex.example_id, "n_questions": len(ex.questions),
                                "samples": rows}) + "\n")
    n = max(len(full_toks), 1)
    print(f"[u-probe] wrote {len(examples)} passages -> {args.out} | "
          f"mean full={sum(full_toks)/n:.0f}tok  understanding-body={sum(body_toks)/n:.0f}tok", flush=True)


if __name__ == "__main__":
    main()
