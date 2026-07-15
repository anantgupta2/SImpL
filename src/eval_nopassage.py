"""Evaluate a checkpoint on PASSAGE-LESS transfer targets (ARC-Challenge, GSM8K).

Why a separate file: `eval_saved_models.py` is built around (passage -> understanding -> answer)
and hard-drops any record with an empty `article` (see load_eval_examples), so ARC's passage-less
science QA loads as zero examples. GSM8K is not multiple-choice at all -- the answer is a number.
Neither fits that harness, but both are useful "did general reasoning transfer at all?" probes
where there is no passage for the understanding role to operate on. We just report raw accuracy.

Tasks:
  * ``mcq_nopassage`` -- a SImpL-format jsonl whose records have article == "" (ARC). Question +
    options -> one boxed letter.
  * ``gsm8k``         -- GSM8K test, CoT -> one boxed number (loader/prompt/parser mirror the
    Roles project's eval_gsm.py, kept independent so the two repos do not couple).

Writes the SAME csv schema as eval_saved_models.py, so scripts/eval/aggregate_transfer.py and the
baseline aggregation read it with no changes. Understanding columns are always 0 (no passage).

  python -m src.eval_nopassage --task mcq_nopassage --data_path data/arc-challenge/test_42_all.jsonl \
      --base_model Qwen/Qwen3-4B-Base --checkpoint_dir <run>/saved_models/step_00208 \
      --cot_samples 8 --output_csv evaluations/transfer/4B_lsat_u4c12_to_arc_s123.csv
  python -m src.eval_nopassage --task gsm8k --base_model Qwen/Qwen3-4B-Base --max_n 500 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import vllm

try:
    from vllm.lora.request import LoRARequest
except ImportError:  # pragma: no cover
    LoRARequest = None

from src.utils.oat_prompt_templates import _mcq_block
from src.utils.parsing_utils import extract_boxed_letter, normalize_gold_letter


# ------------------------------------------------------------------ prompts
def arc_cot_prompt(question_text: str, options: List[str]) -> str:
    """Same shape as qa_cot_prompt but with no Passage block."""
    return (
        "Solve the multiple-choice question.\n"
        "Think step by step, then output exactly one final boxed letter.\n\n"
        f"{_mcq_block(question_text, options)}\n\n"
        "Output format requirement: the final line must contain \\boxed{<letter>}.\n"
    )


def gsm_cot_prompt(question: str) -> str:
    return (
        "Solve the math word problem. Think step-by-step, then give the final answer as "
        "exactly one boxed number, e.g. \\boxed{42}.\n\n"
        f"Problem:\n{question}\n\n"
        "Reason step by step, then output \\boxed{<final number>}.\n"
    )


# ------------------------------------------------------------------ numeric parsing (GSM8K)
def _to_float(s: Any) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").replace("$", "").strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def extract_boxed_number(text: str) -> Optional[float]:
    """Prefer \\boxed{...}; else fall back to the last number anywhere."""
    if not text:
        return None
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        v = _to_float(boxed[-1])
        if v is not None:
            return v
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _to_float(nums[-1]) if nums else None


def numbers_match(a: Optional[float], b: Optional[float], tol: float = 1e-4) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


def parse_gsm_gold(ans: str) -> Optional[float]:
    """GSM8K gold answers end with '#### <number>'."""
    if "####" in ans:
        return _to_float(ans.split("####")[-1])
    return None


# ------------------------------------------------------------------ data
def load_mcq_nopassage(data_path: str) -> List[Dict[str, Any]]:
    """SImpL-format jsonl; keeps records regardless of whether `article` is empty."""
    out: List[Dict[str, Any]] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for q in rec.get("questions", []):
                opts = q.get("options") or []
                gold = normalize_gold_letter(q.get("answer", ""), len(opts))
                if len(opts) < 2 or not gold or not str(q.get("question", "")).strip():
                    continue
                out.append({"question": q["question"], "options": [str(o) for o in opts],
                            "gold": gold})
    return out


def load_gsm8k(max_n: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = [{"question": r["question"], "gold": parse_gsm_gold(r["answer"])} for r in ds]
    rows = [r for r in rows if r["gold"] is not None]
    return rows[:max_n] if max_n else rows


# ------------------------------------------------------------------ eval
def adapter_rank(ckpt: Path, default: int = 64) -> int:
    cfg = ckpt / "adapter_config.json"
    if not cfg.exists():
        return default
    try:
        r = int(json.loads(cfg.read_text()).get("r", default))
    except (json.JSONDecodeError, ValueError):
        return default
    for allowed in (8, 16, 32, 64, 128, 256):  # vLLM only accepts a fixed set; round up.
        if r <= allowed:
            return allowed
    return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["mcq_nopassage", "gsm8k"])
    ap.add_argument("--data_path", default=None, help="jsonl for mcq_nopassage")
    ap.add_argument("--max_n", type=int, default=None, help="cap eval size (gsm8k)")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--checkpoint_dir", default=None,
                    help="LoRA adapter dir; omit to evaluate the untrained base model")
    # Defaults MUST match eval_saved_models.py so these numbers are comparable to every other
    # eval: avg@8, --cot_temperature default 0.6 (the runner scripts never override it), top_p 0.95
    # (eval_saved_models._generate_n), and answer_max_tokens 1024 (what the runners pass).
    ap.add_argument("--cot_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--eval_seed", type=int, default=42)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--run_name", default="nopassage")
    ap.add_argument("--step", default=None)
    args = ap.parse_args()

    if args.task == "mcq_nopassage":
        if not args.data_path:
            raise SystemExit("--data_path is required for mcq_nopassage")
        rows = load_mcq_nopassage(args.data_path)
        prompts = [arc_cot_prompt(r["question"], r["options"]) for r in rows]
    else:
        rows = load_gsm8k(args.max_n)
        prompts = [gsm_cot_prompt(r["question"]) for r in rows]
    if not rows:
        raise SystemExit(f"No examples loaded for task={args.task}")
    print(f"[eval-nopassage] task={args.task} n={len(rows)} ckpt={args.checkpoint_dir or 'BASE'}")

    ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    use_lora = bool(ckpt and (ckpt / "adapter_config.json").exists())
    llm = vllm.LLM(model=args.base_model, dtype="bfloat16", seed=args.eval_seed,
                   gpu_memory_utilization=args.gpu_memory_utilization,
                   enable_lora=use_lora,
                   max_lora_rank=adapter_rank(ckpt) if use_lora else 16,
                   disable_log_stats=True)
    sp = vllm.SamplingParams(n=args.cot_samples, temperature=args.temperature,
                             top_p=args.top_p, max_tokens=args.max_tokens, seed=args.eval_seed)
    print(f"[eval-nopassage] avg@{args.cot_samples} temp={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens} seed={args.eval_seed}")
    lora_req = LoRARequest("adapter", 1, str(ckpt)) if (use_lora and LoRARequest) else None
    outs = llm.generate(prompts, sp, lora_request=lora_req) if lora_req else llm.generate(prompts, sp)

    # avg@N: each question contributes the fraction of its samples that are correct.
    total = 0.0
    for r, o in zip(rows, outs):
        gens = [c.text for c in o.outputs]
        if args.task == "mcq_nopassage":
            hits = sum(1 for g in gens
                       if extract_boxed_letter(g, len(r["options"])) == r["gold"])
        else:
            hits = sum(1 for g in gens if numbers_match(extract_boxed_number(g), r["gold"]))
        total += hits / max(1, len(gens))
    acc = total / len(rows)
    print(f"[eval-nopassage] {args.task} acc={acc:.4f} ({total:.2f}/{len(rows)})")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    # Same schema as eval_saved_models.py so aggregate_transfer.py needs no changes.
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cot_accuracy", "cot_correct", "understanding_plus_cot_accuracy",
                    "understanding_plus_cot_correct", "u_and_a_accuracy", "u_and_a_correct",
                    "total_questions", "is_instruct", "run_name", "step", "checkpoint_path",
                    "base_model", "num_articles"])
        w.writerow([f"{acc:.6f}", f"{total:.2f}", "0.000000", "0", "0.000000", "0",
                    len(rows), 0, args.run_name,
                    args.step or (ckpt.name if ckpt else "base"),
                    str(ckpt) if ckpt else "base", args.base_model, 0])
    print(f"[eval-nopassage] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
