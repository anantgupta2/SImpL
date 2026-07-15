from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import random
import re
import shutil
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import vllm
from datasets import load_dataset

try:
    _peft = importlib.import_module("peft")
    PeftConfig = getattr(_peft, "PeftConfig", None)
except Exception:
    PeftConfig = None

try:
    from vllm.lora.request import LoRARequest
except ImportError:
    LoRARequest = None

from src.utils.oat_prompt_templates import (
    understanding_prompt,
    qa_cot_prompt,
    qa_eval_understanding_only_prompt,
    understand_and_answer_prompt,
)
from src.utils.parsing_utils import extract_boxed_letter, normalize_gold_letter, parse_questions


@dataclass
class EvalQuestion:
    question: str
    options: List[str]
    answer: str


@dataclass
class EvalExample:
    example_id: str
    article: str
    questions: List[EvalQuestion]


@dataclass
class EvalResult:
    mode: str
    total_questions: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return float(self.correct) / float(self.total_questions)


@dataclass
class CheckpointEvalResult:
    checkpoint_path: str
    run_name: str
    step: str
    base_model: str
    is_instruct: bool
    num_articles: int
    cot_only: EvalResult
    understanding_plus_cot: EvalResult
    u_and_a: EvalResult


@dataclass
class QuestionEvalTrace:
    example_id: str
    question_index: int
    question: str
    options: List[str]
    gold_answer: str
    understanding_output: str
    cot_output: str
    understanding_qa_output: str
    u_and_a_output: str
    cot_prediction: str
    understanding_qa_prediction: str
    u_and_a_prediction: str


@dataclass(frozen=True)
class LoRAAdapterInfo:
    has_adapter: bool
    adapter_path: Optional[Path]
    base_model_name_or_path: Optional[str]


def _extract_understanding_from_tags(text: str) -> str:
    # Mirror the training-time extraction (cot_and_understanding_oat /
    # understanding_only_oat) exactly so eval scores what was trained.
    match = re.search(r"<understanding>\s*(.*?)\s*</understanding>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # No closing tag: extract from the start tag to the end of the text.
    match2 = re.search(r"<understanding>\s*(.*?)\s*$", text, re.DOTALL | re.IGNORECASE)
    if match2:
        return match2.group(1).strip()
    return ""


def load_lora_adapter_info(checkpoint_dir: Path) -> LoRAAdapterInfo:
    adapter_cfg = checkpoint_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        return LoRAAdapterInfo(
            has_adapter=False,
            adapter_path=None,
            base_model_name_or_path=None,
        )

    base_model: Optional[str] = None

    # Prefer PEFT config loading because it handles adapter metadata formats robustly.
    if PeftConfig is not None:
        try:
            peft_cfg = PeftConfig.from_pretrained(str(checkpoint_dir))
            inferred = getattr(peft_cfg, "base_model_name_or_path", None)
            if isinstance(inferred, str) and inferred.strip():
                base_model = inferred.strip()
        except Exception:
            base_model = None

    if not base_model:
        try:
            with adapter_cfg.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            inferred = cfg.get("base_model_name_or_path", "")
            if isinstance(inferred, str) and inferred.strip():
                base_model = inferred.strip()
        except Exception:
            base_model = None

    return LoRAAdapterInfo(
        has_adapter=True,
        adapter_path=checkpoint_dir,
        base_model_name_or_path=base_model,
    )


def read_adapter_rank(checkpoint_dir: Path, default: int = 16) -> int:
    """Read the LoRA rank ('r') from adapter_config.json. vLLM's max_lora_rank must be
    >= this or native adapter loading fails (the reason eval previously merged instead)."""
    cfg = checkpoint_dir / "adapter_config.json"
    try:
        with cfg.open("r", encoding="utf-8") as f:
            r = int(json.load(f).get("r", default))
        # vLLM only accepts max_lora_rank from a fixed set; round up to the next supported value.
        for allowed in (8, 16, 32, 64, 128, 256, 320, 512):
            if r <= allowed:
                return allowed
        return r
    except Exception:
        return default


def _load_json_or_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            if isinstance(obj.get("data"), list):
                return [x for x in obj["data"] if isinstance(x, dict)]
            if isinstance(obj.get("examples"), list):
                return [x for x in obj["examples"] if isinstance(x, dict)]
            return [obj]

    raise ValueError(f"Unsupported file extension for data: {path}")


def load_eval_examples(
    data_path: str,
    split: str,
    input_key: str,
    output_key: str,
    max_articles: Optional[int],
    max_questions_per_article: Optional[int],
) -> List[EvalExample]:
    p = Path(data_path)
    if p.exists() and p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
        records = _load_json_or_jsonl_records(p)
    else:
        ds = load_dataset(data_path, split=split)
        records = [dict(row) for row in ds]

    examples: List[EvalExample] = []
    for idx, rec in enumerate(records):
        article = rec.get(input_key, "")
        if not isinstance(article, str) or not article.strip():
            continue

        raw_questions = parse_questions(rec.get(output_key, []))
        q_list: List[EvalQuestion] = []
        for q in raw_questions:
            question_text = q.get("question", "")
            options = q.get("options", [])
            answer = normalize_gold_letter(q.get("answer", ""), len(options) if isinstance(options, list) else 4)
            if not isinstance(question_text, str) or not question_text.strip():
                continue
            if not isinstance(options, list) or len(options) < 2:
                continue
            if not answer:
                continue

            q_list.append(
                EvalQuestion(
                    question=question_text,
                    options=[str(o) for o in options],
                    answer=answer,
                )
            )
            if max_questions_per_article is not None and len(q_list) >= max_questions_per_article:
                break

        if not q_list:
            continue

        example_id = str(rec.get("example_id", f"example_{idx}"))
        examples.append(EvalExample(example_id=example_id, article=article, questions=q_list))

        if max_articles is not None and len(examples) >= max_articles:
            break

    return examples


def discover_checkpoints(
    checkpoint_root: Optional[str],
    checkpoint_dirs: Iterable[str],
) -> List[Path]:
    discovered: List[Path] = []
    for ckpt in checkpoint_dirs:
        p = Path(ckpt)
        if p.exists() and p.is_dir():
            discovered.append(p.resolve())

    if checkpoint_root:
        root = Path(checkpoint_root)
        if root.exists():
            for candidate in root.glob("**/saved_models/step_*"):
                if candidate.is_dir() and (candidate / "adapter_config.json").exists():
                    discovered.append(candidate.resolve())

    deduped = sorted(set(discovered), key=lambda x: str(x))

    def sort_key(path: Path) -> Tuple[str, int, str]:
        run_name = path.parent.parent.name if path.parent.name == "saved_models" else path.parent.name
        m = re.match(r"step_(\d+)", path.name)
        step_int = int(m.group(1)) if m else -1
        return run_name, step_int, path.name

    return sorted(deduped, key=sort_key)


def infer_base_model(
    checkpoint_dir: Path,
    base_model_override: Optional[str],
    lora_info: LoRAAdapterInfo,
) -> str:
    if base_model_override:
        return base_model_override

    if lora_info.base_model_name_or_path:
        return lora_info.base_model_name_or_path

    # Fallback to loading checkpoint as a full model path if no adapter metadata exists.
    return str(checkpoint_dir)


def infer_instruct_flag(base_model: str, is_instruct_arg: Optional[bool]) -> bool:
    if is_instruct_arg is not None:
        return bool(is_instruct_arg)
    return "instruct" in (base_model or "").lower()


def _extract_text_outputs(outputs) -> List[str]:
    texts: List[str] = []
    for out in outputs:
        if not out.outputs:
            texts.append("")
            continue
        texts.append(out.outputs[0].text or "")
    return texts


def patch_vllm_lru_cache_touch_for_cachetools_compat() -> None:
    """Patch vLLM LRUCache.touch for cachetools versions without _LRUCache__update.

    vLLM 0.8.x may call a cachetools private method removed in newer cachetools,
    which crashes during LoRA adapter activation.
    """
    try:
        from vllm.utils import LRUCache as VLLMLRUCache
    except Exception:
        return

    if getattr(VLLMLRUCache, "_vllm_touch_compat_patched", False):
        return

    def _compat_touch(self, key: Hashable) -> None:  # type: ignore[name-defined]
        if key not in self:
            return

        update = getattr(self, "_LRUCache__update", None)
        if callable(update):
            update(key)
            return

        touch = getattr(self, "_LRUCache__touch", None)
        if callable(touch):
            touch(key)
            return

        order = getattr(self, "_LRUCache__order", None)
        if order is not None and hasattr(order, "move_to_end"):
            order.move_to_end(key)

    # Patch once at runtime so adapter activation works across cachetools versions.
    VLLMLRUCache.touch = _compat_touch
    VLLMLRUCache._vllm_touch_compat_patched = True


def _torch_dtype_from_str(dtype: str, torch_mod: Any) -> Optional[Any]:
    key = (dtype or "").strip().lower()
    name_map = {
        "float16": "float16",
        "fp16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    attr = name_map.get(key)
    if not attr:
        return None
    return getattr(torch_mod, attr, None)


def merge_lora_adapter_into_base(
    checkpoint_dir: Path,
    base_model: str,
    merged_model_root: Path,
    *,
    dtype: str,
    trust_remote_code: bool,
) -> Path:
    """Merge a LoRA adapter into a full model directory for vLLM loading.

    This avoids runtime LoRA adapter activation paths in vLLM.
    """
    if checkpoint_dir.parent.name == "saved_models":
        run_name = checkpoint_dir.parent.parent.name
    else:
        run_name = checkpoint_dir.parent.name
    merged_dir = merged_model_root / run_name / checkpoint_dir.name

    if (merged_dir / "config.json").exists() and (merged_dir / "tokenizer_config.json").exists():
        return merged_dir

    try:
        transformers_mod = importlib.import_module("transformers")
        peft_mod = importlib.import_module("peft")
        torch_mod = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Merging LoRA adapters requires transformers, peft, and torch in the runtime environment."
        ) from exc

    AutoModelForCausalLM = getattr(transformers_mod, "AutoModelForCausalLM", None)
    AutoTokenizer = getattr(transformers_mod, "AutoTokenizer", None)
    PeftModel = getattr(peft_mod, "PeftModel", None)
    if AutoModelForCausalLM is None or AutoTokenizer is None or PeftModel is None:
        raise RuntimeError("Could not resolve transformers/peft classes required for LoRA merge.")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(trust_remote_code),
        "low_cpu_mem_usage": True,
    }
    torch_dtype = _torch_dtype_from_str(dtype, torch_mod)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    merged_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    peft_model = PeftModel.from_pretrained(base, str(checkpoint_dir), is_trainable=False)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=bool(trust_remote_code))
    tokenizer.save_pretrained(str(merged_dir))

    # Best effort: preserve generation config if exposed on the base model object.
    try:
        generation_config = getattr(base, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(str(merged_dir))
    except Exception:
        pass

    del merged_model
    del peft_model
    del base
    gc.collect()
    try:
        torch_mod.cuda.empty_cache()
    except Exception:
        pass

    return merged_dir


class CheckpointEvaluator:
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        lora_info: LoRAAdapterInfo,
        base_model: str,
        is_instruct: bool,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        dtype: str,
        max_model_len: Optional[int],
        batch_size: int,
        reasoning_max_items: int,
        reasoning_max_tokens: int,
        answer_max_tokens: int,
        trust_remote_code: bool,
        system_prompt: str = "You are a helpful assistant.",
        dataset_name: str = "race-c",
        understanding_with_passage: bool = False,
        cot_samples: int = 1,
        cot_sample_temperature: float = 0.6,
        eval_seed: int = 42,
        max_lora_rank: Optional[int] = None,
        cot_eval_only: bool = False,
    ) -> None:
        # Fixed seed for the vLLM engine AND every sampling request, so each checkpoint
        # is evaluated deterministically -- a fresh model load mid-run reseeds to this,
        # giving identical numbers on re-runs (avg@N sampling is reproducible).
        self.eval_seed = int(eval_seed)
        self.checkpoint_dir = checkpoint_dir
        self.lora_info = lora_info
        self.base_model = base_model
        self.is_instruct = is_instruct
        self.batch_size = int(batch_size)
        self.reasoning_max_items = int(reasoning_max_items)
        self.reasoning_max_tokens = int(reasoning_max_tokens)
        self.answer_max_tokens = int(answer_max_tokens)
        self.system_prompt = system_prompt
        self.dataset_name = str(dataset_name)
        # Must match the training-time `use_baseline_reward` flag: when the
        # understanding was scored with the passage present during training,
        # include the passage here too.
        self.understanding_with_passage = bool(understanding_with_passage)
        # Multi-sample CoT eval: sample N answers/question at temperature and report the
        # mean per-question accuracy (avg@N) -> a lower-variance estimate of the (stochastic)
        # policy's accuracy. cot_samples=1 keeps the original greedy single-sample behaviour.
        self.cot_samples = max(1, int(cot_samples))
        self.cot_sample_temperature = float(cot_sample_temperature)
        # When True, evaluate ONLY the plain-CoT mode (skip understanding generation and the
        # understanding_plus_cot / understand_and_answer modes) -- ~3-4x faster; the skipped
        # modes report 0/0 (accuracy 0.0).
        self.cot_eval_only = bool(cot_eval_only)

        patch_vllm_lru_cache_touch_for_cachetools_compat()

        has_adapter = bool(lora_info.has_adapter)
        if has_adapter and LoRARequest is None:
            raise RuntimeError(
                "This vLLM build does not expose LoRARequest, but adapter checkpoints were provided. "
                "Please upgrade vLLM to a LoRA-capable version."
            )

        llm_kwargs: Dict[str, Any] = {
            "model": base_model,
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "dtype": dtype,
            "trust_remote_code": bool(trust_remote_code),
            "seed": self.eval_seed,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if has_adapter:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = 1
            # CRITICAL: vLLM's default max_lora_rank is 16; our adapters are r64, so without
            # this native LoRA loading fails -- which is why eval used to merge full models.
            if max_lora_rank is not None:
                llm_kwargs["max_lora_rank"] = int(max_lora_rank)

        self.llm = vllm.LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        # Each distinct adapter gets a unique integer id so vLLM loads it fresh (it caches by
        # id). set_adapter() lets one persistent engine hot-swap adapters across checkpoints.
        self._lora_id_counter = 0
        self.lora_request = None
        if has_adapter:
            self.set_adapter(checkpoint_dir, lora_info.adapter_path or checkpoint_dir)

    def set_adapter(self, checkpoint_dir: Path, adapter_path: Path) -> None:
        """Point the engine at a new LoRA adapter without reloading the base model."""
        self._lora_id_counter += 1
        self.lora_request = LoRARequest(
            checkpoint_dir.name,
            self._lora_id_counter,
            str(adapter_path),
        )

    def _apply_template(self, prompt_text: str) -> str:
        if not self.is_instruct:
            return prompt_text

        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None) is not None:
            messages = []
            if getattr(self, "system_prompt", None):
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt_text})
            
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt_text

    def _generate(
        self,
        prompts: List[str],
        *,
        max_tokens: int,
    ) -> List[str]:
        if not prompts:
            return []

        sampling_params = vllm.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(max_tokens),
            n=1,
            seed=self.eval_seed,
        )

        all_texts: List[str] = []
        outputs = self.llm.generate(
                prompts,
                sampling_params,
                lora_request=self.lora_request,
            )
        all_texts = _extract_text_outputs(outputs)
        return all_texts

    def _generate_n(self, prompts: List[str], *, max_tokens: int, n: int, temperature: float) -> List[List[str]]:
        """Sample n completions per prompt. Returns a list (per prompt) of n texts."""
        if not prompts:
            return []
        sampling_params = vllm.SamplingParams(
            temperature=float(temperature),
            top_p=0.95,
            max_tokens=int(max_tokens),
            n=int(n),
            seed=self.eval_seed,
        )
        outputs = self.llm.generate(prompts, sampling_params, lora_request=self.lora_request)
        return [[c.text for c in o.outputs] for o in outputs]

    def _score_predictions(self, predictions: List[str], answers: List[str], options_list: List[List[str]]) -> Tuple[int, int]:
        correct = 0
        total = min(len(predictions), len(answers))
        for pred_text, gold, opts in zip(predictions, answers, options_list):
            pred = extract_boxed_letter(pred_text, len(opts))
            correct += int(pred == gold)
        return correct, total

    def evaluate(self, examples: List[EvalExample]) -> Tuple[EvalResult, EvalResult, EvalResult, List[QuestionEvalTrace]]:
        if self.cot_eval_only:
            # Plain-CoT-only fast path: no understanding generation at all.
            implications = [""] * len(examples)
        else:
            implication_prompts: List[str] = []
            for ex in examples:
                implication_prompts.append(
                    self._apply_template(understanding_prompt(ex.article, self.dataset_name))
                )
            implications = self._generate(
                implication_prompts,
                max_tokens=self.reasoning_max_tokens,
            )
            if len(implications) < len(examples):
                implications.extend([""] * (len(examples) - len(implications)))

        cot_prompts: List[str] = []
        imp_plus_cot_prompts: List[str] = []
        u_and_a_prompts: List[str] = []
        answers: List[str] = []
        question_meta: List[Tuple[str, int, str, List[str], str, str]] = []

        for ex, imp in zip(examples, implications):
            extracted_imp = _extract_understanding_from_tags(imp)
            # Match training: understanding is scored via qa_eval_understanding_only_prompt,
            # with the passage included only when training used use_baseline_reward=True.
            imp_article = ex.article if self.understanding_with_passage else ""
            for q_idx, q in enumerate(ex.questions):
                cot_prompts.append(self._apply_template(qa_cot_prompt(ex.article, q.question, q.options, self.dataset_name)))
                imp_plus_cot_prompts.append(
                    self._apply_template(qa_eval_understanding_only_prompt(imp_article, extracted_imp, q.question, q.options, self.dataset_name))
                )
                u_and_a_prompts.append(
                    self._apply_template(understand_and_answer_prompt(ex.article, q.question, q.options, self.dataset_name))
                )
                answers.append(q.answer)
                question_meta.append(
                    (
                        ex.example_id,
                        int(q_idx),
                        q.question,
                        q.options,
                        q.answer,
                        imp,
                    )
                )

        options_list = [t[3] for t in question_meta]
        # Defaults for the understanding modes -- overwritten below unless cot_eval_only.
        imp_outputs = [""] * len(cot_prompts)
        u_and_a_outputs = [""] * len(cot_prompts)
        imp_correct = imp_total = 0
        u_and_a_correct = u_and_a_total = 0
        if self.cot_samples > 1:
            # Sample N answers/question and score each mode by mean per-question accuracy
            # (avg@N) -> lower-variance estimate of the stochastic policy's accuracy.
            def _avg_score(samples_list: List[List[str]]) -> Tuple[float, int]:
                correct = 0.0
                for samples, gold, opts in zip(samples_list, answers, options_list):
                    if not samples:
                        continue
                    correct += sum(int(extract_boxed_letter(t, len(opts)) == gold) for t in samples) / len(samples)
                return correct, len(answers)

            n, temp = self.cot_samples, self.cot_sample_temperature
            cot_s = self._generate_n(cot_prompts, max_tokens=self.answer_max_tokens, n=n, temperature=temp)
            cot_correct, cot_total = _avg_score(cot_s)
            cot_outputs = [s[0] if s else "" for s in cot_s]
            if not self.cot_eval_only:
                imp_s = self._generate_n(imp_plus_cot_prompts, max_tokens=self.answer_max_tokens, n=n, temperature=temp)
                una_s = self._generate_n(u_and_a_prompts, max_tokens=self.answer_max_tokens, n=n, temperature=temp)
                imp_correct, imp_total = _avg_score(imp_s)
                u_and_a_correct, u_and_a_total = _avg_score(una_s)
                # keep the first sample of each for the per-question traces
                imp_outputs = [s[0] if s else "" for s in imp_s]
                u_and_a_outputs = [s[0] if s else "" for s in una_s]
        else:
            cot_outputs = self._generate(cot_prompts, max_tokens=self.answer_max_tokens)
            cot_correct, cot_total = self._score_predictions(cot_outputs, answers, options_list)
            if not self.cot_eval_only:
                imp_outputs = self._generate(imp_plus_cot_prompts, max_tokens=self.answer_max_tokens)
                u_and_a_outputs = self._generate(u_and_a_prompts, max_tokens=self.answer_max_tokens)
                imp_correct, imp_total = self._score_predictions(imp_outputs, answers, options_list)
                u_and_a_correct, u_and_a_total = self._score_predictions(u_and_a_outputs, answers, options_list)

        traces: List[QuestionEvalTrace] = []
        for i, (example_id, q_idx, question, options, gold, implication_text) in enumerate(question_meta):
            cot_text = cot_outputs[i] if i < len(cot_outputs) else ""
            imp_text = imp_outputs[i] if i < len(imp_outputs) else ""
            una_text = u_and_a_outputs[i] if i < len(u_and_a_outputs) else ""
            traces.append(
                QuestionEvalTrace(
                    example_id=example_id,
                    question_index=q_idx,
                    question=question,
                    options=options,
                    gold_answer=gold,
                    understanding_output=implication_text,
                    cot_output=cot_text,
                    understanding_qa_output=imp_text,
                    u_and_a_output=una_text,
                    cot_prediction=extract_boxed_letter(cot_text, len(options)),
                    understanding_qa_prediction=extract_boxed_letter(imp_text, len(options)),
                    u_and_a_prediction=extract_boxed_letter(una_text, len(options)),
                )
            )

        return (
            EvalResult(mode="cot_only", total_questions=cot_total, correct=cot_correct),
            EvalResult(
                mode="understanding_plus_cot",
                total_questions=imp_total,
                correct=imp_correct,
            ),
            EvalResult(
                mode="understand_and_answer",
                total_questions=u_and_a_total,
                correct=u_and_a_correct,
            ),
            traces,
        )



def _run_name_and_step(checkpoint_path: Path) -> Tuple[str, str]:
    if checkpoint_path.parent.name == "saved_models":
        return checkpoint_path.parent.parent.name, checkpoint_path.name
    return checkpoint_path.parent.name, checkpoint_path.name


def _sample_question_traces(
    traces: List[QuestionEvalTrace],
    sample_count: int,
    sample_seed: int,
) -> List[QuestionEvalTrace]:
    if sample_count <= 0 or not traces:
        return []
    if sample_count >= len(traces):
        return list(traces)

    rng = random.Random(int(sample_seed))
    selected_idx = rng.sample(range(len(traces)), sample_count)
    return [traces[i] for i in selected_idx]


def evaluate_checkpoints(args: argparse.Namespace) -> Tuple[List[CheckpointEvalResult], List[Dict[str, Any]]]:
    import glob
    
    data_path = args.data_path
    if not data_path and args.dataset_name:
        dataset_dir = Path("data") / args.dataset_name
        test_files = list(dataset_dir.glob("test*.json*"))
        if len(test_files) == 1:
            data_path = str(test_files[0])
        elif len(test_files) > 1:
            raise ValueError(f"Found multiple test files in {dataset_dir}: {test_files}")
        else:
            raise ValueError(f"No test*.json or test*.jsonl found in {dataset_dir}")
    elif not data_path:
        raise ValueError("Must provide either --data_path or --dataset_name")

    examples = load_eval_examples(
        data_path=data_path,
        split=args.split,
        input_key=args.input_key,
        output_key=args.output_key,
        max_articles=args.max_articles,
        max_questions_per_article=args.max_questions_per_article,
    )
    if not examples:
        raise ValueError("No valid evaluation examples loaded from --data_path")

    # If explicit checkpoint_dir values are provided, evaluate only those.
    checkpoint_root = args.checkpoint_root if not args.checkpoint_dir else None
    checkpoints = discover_checkpoints(
        checkpoint_root=checkpoint_root,
        checkpoint_dirs=args.checkpoint_dir,
    )
    if not checkpoints:
        raise ValueError("No checkpoints found. Provide --checkpoint_root or --checkpoint_dir.")
    if int(getattr(args, "last_n_steps", 0)) > 0:
        # Only evaluate the last N checkpoints (by step) -- for slow datasets (PW) where
        # the full curve isn't worth it; the converged tail is what tests generalization.
        checkpoints = checkpoints[-int(args.last_n_steps):]
        print(f"[eval] --last_n_steps={args.last_n_steps} -> evaluating {[c.name for c in checkpoints]}")
    _mn = int(getattr(args, "min_step", 0)); _mx = int(getattr(args, "max_step", 0))
    if _mn or _mx:
        # Restrict to checkpoints in [min_step, max_step] -- lets a dense curve be
        # PARTITIONED across several jobs (each a disjoint step range -> its own CSV).
        def _stepof(c):
            m = re.match(r"step_(\d+)", c.name)
            return int(m.group(1)) if m else -1
        checkpoints = [c for c in checkpoints if (_mn == 0 or _stepof(c) >= _mn) and (_mx == 0 or _stepof(c) <= _mx)]
        print(f"[eval] step range [{_mn},{_mx}] -> {len(checkpoints)} checkpoints")
        if not checkpoints:
            raise ValueError(f"No checkpoints in step range [{_mn},{_mx}].")

    results: List[CheckpointEvalResult] = []
    sampled_outputs: List[Dict[str, Any]] = []

    # Resume: if the output CSV already has rows, keep them (frozen_rows) and SKIP
    # those steps -- unless --redo_all. So a re-run only fills in missing checkpoints.
    out_csv = Path(args.output_csv) if getattr(args, "output_csv", None) else None
    frozen_rows: List[List[str]] = []
    done_steps: set = set()
    if out_csv is not None and not getattr(args, "redo_all", False):
        frozen_rows, done_steps = _read_canonical_rows(out_csv)
        if done_steps:
            print(f"[eval] resume: {len(done_steps)} step(s) already in {out_csv}; will skip them")

    def _flush() -> None:
        if out_csv is not None:
            _write_curve(out_csv, frozen_rows, results)

    # Evaluate LAST steps first (descending) so the final-checkpoint results land in the
    # CSV ASAP. _write_curve sorts every flush by step, so the CSV stays in ascending
    # step order regardless -- step_k is always inserted at its correct position.
    checkpoints = list(reversed(checkpoints))

    # FAST PATH: all checkpoints of a run share one base model + LoRA rank, so build the vLLM
    # engine ONCE and hot-swap adapters per checkpoint (set_adapter) -- no per-step full-model
    # merge, no engine reload. ~order-of-magnitude faster + no 100s of GB of merged-model temp.
    # The old merge path is still available via --merge_lora_before_eval (and is used as a
    # fallback for non-adapter / full-model checkpoints).
    persistent_evaluator: Optional[CheckpointEvaluator] = None
    persistent_base: Optional[str] = None

    for ckpt_idx, ckpt in enumerate(checkpoints):
        _, _step_name = _run_name_and_step(ckpt)
        if _step_name in done_steps:
            print(f"[eval] skip {_step_name} (already evaluated; use --redo_all to redo)")
            continue
        merged_model_path_to_cleanup: Optional[Path] = None
        try:
            # Loading the adapter/merging reads several JSON files; a transient FS/NFS
            # read glitch here shouldn't kill the whole job -> skip just this checkpoint.
            lora_info = load_lora_adapter_info(ckpt)
            base_model = infer_base_model(ckpt, args.base_model, lora_info)
            is_instruct = infer_instruct_flag(base_model, args.is_instruct)

            eval_base_model = base_model
            eval_lora_info = lora_info
            if args.merge_lora_before_eval and lora_info.has_adapter:
                merged_model_path = merge_lora_adapter_into_base(
                    checkpoint_dir=ckpt,
                    base_model=base_model,
                    merged_model_root=Path(args.merged_model_root),
                    dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code,
                )
                eval_base_model = str(merged_model_path)
                eval_lora_info = LoRAAdapterInfo(
                    has_adapter=False,
                    adapter_path=None,
                    base_model_name_or_path=eval_base_model,
                )
                if args.delete_merged_model_after_eval:
                    merged_model_path_to_cleanup = merged_model_path
                print(f"[eval] merged LoRA adapter into model dir: {eval_base_model}")
        except Exception as _load_exc:
            print(f"[eval] WARNING: failed to load checkpoint {ckpt} "
                  f"({type(_load_exc).__name__}: {_load_exc}); skipping this step")
            continue

        print(f"[eval] Loading checkpoint: {ckpt}")
        print(
            f"[eval] base_model={base_model}, eval_model={eval_base_model}, "
            f"is_instruct={is_instruct}, has_adapter={int(eval_lora_info.has_adapter)}"
        )

        # Reseed on every model load so each checkpoint is evaluated deterministically
        # (identical numbers on re-runs), regardless of position in the loop.
        random.seed(args.eval_seed)
        np.random.seed(args.eval_seed)
        try:
            import torch
            torch.manual_seed(args.eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.eval_seed)
        except Exception:
            pass

        # Hot-swap fast path: reuse one engine + swap adapters; only build a transient engine
        # for the merge path or for non-adapter (full-model) checkpoints.
        use_lora_swap = bool(eval_lora_info.has_adapter) and not args.merge_lora_before_eval
        evaluator: Optional[CheckpointEvaluator] = None
        transient_evaluator = False
        try:
            if use_lora_swap:
                if persistent_evaluator is None or persistent_base != eval_base_model:
                    if persistent_evaluator is not None:
                        del persistent_evaluator
                        persistent_evaluator = None
                        gc.collect()
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    persistent_evaluator = CheckpointEvaluator(
                        checkpoint_dir=ckpt,
                        lora_info=eval_lora_info,
                        base_model=eval_base_model,
                        is_instruct=is_instruct,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        dtype=args.dtype,
                        max_model_len=args.max_model_len,
                        batch_size=args.batch_size,
                        reasoning_max_items=args.reasoning_max_items,
                        reasoning_max_tokens=args.reasoning_max_tokens,
                        answer_max_tokens=args.answer_max_tokens,
                        trust_remote_code=args.trust_remote_code,
                        system_prompt=args.system_prompt,
                        dataset_name=(args.dataset_name or "race-c"),
                        understanding_with_passage=args.understanding_with_passage,
                        cot_samples=args.cot_samples,
                        cot_sample_temperature=args.cot_temperature,
                        eval_seed=args.eval_seed,
                        max_lora_rank=read_adapter_rank(ckpt),
                        cot_eval_only=args.cot_eval_only,
                    )
                    persistent_base = eval_base_model
                else:
                    persistent_evaluator.set_adapter(ckpt, eval_lora_info.adapter_path or ckpt)
                evaluator = persistent_evaluator
            else:
                evaluator = CheckpointEvaluator(
                    checkpoint_dir=ckpt,
                    lora_info=eval_lora_info,
                    base_model=eval_base_model,
                    is_instruct=is_instruct,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    dtype=args.dtype,
                    max_model_len=args.max_model_len,
                    batch_size=args.batch_size,
                    reasoning_max_items=args.reasoning_max_items,
                    reasoning_max_tokens=args.reasoning_max_tokens,
                    answer_max_tokens=args.answer_max_tokens,
                    trust_remote_code=args.trust_remote_code,
                    system_prompt=args.system_prompt,
                    dataset_name=(args.dataset_name or "race-c"),
                    understanding_with_passage=args.understanding_with_passage,
                    cot_samples=args.cot_samples,
                    cot_sample_temperature=args.cot_temperature,
                    eval_seed=args.eval_seed,
                    cot_eval_only=args.cot_eval_only,
                )
                transient_evaluator = True

            cot_result, imp_result, u_and_a_result, traces = evaluator.evaluate(examples)
            run_name, step = _run_name_and_step(ckpt)

            row = CheckpointEvalResult(
                checkpoint_path=str(ckpt),
                run_name=run_name,
                step=step,
                base_model=base_model,
                is_instruct=is_instruct,
                num_articles=len(examples),
                cot_only=cot_result,
                understanding_plus_cot=imp_result,
                u_and_a=u_and_a_result,
            )
            results.append(row)

            # Flush incrementally (frozen prior rows + new results) so a long all-steps
            # job killed at the embers 2h wall keeps every checkpoint evaluated so far.
            _flush()

            print(
                "[eval] "
                f"run={run_name} step={step} "
                f"cot_acc={cot_result.accuracy:.4f} ({cot_result.correct}/{cot_result.total_questions}) "
                f"understanding_plus_cot_acc={imp_result.accuracy:.4f} ({imp_result.correct}/{imp_result.total_questions}) "
                f"u_and_a_acc={u_and_a_result.accuracy:.4f} ({u_and_a_result.correct}/{u_and_a_result.total_questions})"
            )

            sampled = _sample_question_traces(
                traces,
                sample_count=int(args.sample_output_count_per_checkpoint),
                sample_seed=int(args.sample_output_seed) + int(ckpt_idx),
            )
            for sampled_idx, tr in enumerate(sampled, start=1):
                sampled_outputs.append(
                    {
                        "run_name": run_name,
                        "step": step,
                        "checkpoint_path": str(ckpt),
                        "base_model": base_model,
                        "is_instruct": bool(is_instruct),
                        "sample_index": sampled_idx,
                        "example_id": tr.example_id,
                        "question_index": tr.question_index,
                        "question": tr.question,
                        "options": tr.options,
                        "gold_answer": tr.gold_answer,
                        "understanding_output": tr.understanding_output,
                        "cot_output": tr.cot_output,
                        "cot_prediction": tr.cot_prediction,
                        "cot_correct": int(tr.cot_prediction == tr.gold_answer),
                        "understanding_qa_output": tr.understanding_qa_output,
                        "understanding_qa_prediction": tr.understanding_qa_prediction,
                        "understanding_qa_correct": int(tr.understanding_qa_prediction == tr.gold_answer),
                        "u_and_a_output": tr.u_and_a_output,
                        "u_and_a_prediction": tr.u_and_a_prediction,
                        "u_and_a_correct": int(tr.u_and_a_prediction == tr.gold_answer),
                    }
                )
        except Exception as _eval_exc:
            run_name, step = _run_name_and_step(ckpt)
            print(f"[eval] WARNING: eval failed at {ckpt} "
                  f"({type(_eval_exc).__name__}: {_eval_exc}); skipping this step")
        finally:
            # Only tear down transient (merge / full-model) engines; the persistent LoRA-swap
            # engine is reused across checkpoints and freed after the loop.
            if transient_evaluator and evaluator is not None:
                del evaluator
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if merged_model_path_to_cleanup and merged_model_path_to_cleanup.exists():
                try:
                    shutil.rmtree(merged_model_path_to_cleanup)
                    print(f"[eval] deleted merged model dir: {merged_model_path_to_cleanup}")
                except Exception as exc:
                    print(
                        f"[eval] warning: failed to delete merged model dir "
                        f"{merged_model_path_to_cleanup}: {exc}"
                    )

    if persistent_evaluator is not None:
        del persistent_evaluator
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    _flush()  # ensure the final CSV (frozen + all new rows) is written for named output
    return results, sampled_outputs


_SUMMARY_HEADERS = [
    "cot_accuracy", "cot_correct",
    "understanding_plus_cot_accuracy", "understanding_plus_cot_correct",
    "u_and_a_accuracy", "u_and_a_correct",
    "total_questions", "is_instruct", "run_name", "step",
    "checkpoint_path", "base_model", "num_articles",
]


def _result_to_row(r: CheckpointEvalResult) -> List[Any]:
    return [
        f"{r.cot_only.accuracy:.6f}", r.cot_only.correct,
        f"{r.understanding_plus_cot.accuracy:.6f}", r.understanding_plus_cot.correct,
        f"{r.u_and_a.accuracy:.6f}", r.u_and_a.correct,
        r.cot_only.total_questions, int(r.is_instruct), r.run_name, r.step,
        r.checkpoint_path, r.base_model, r.num_articles,
    ]


def _read_canonical_rows(output_csv: Path) -> Tuple[List[List[str]], set]:
    """Read an existing summary CSV into canonical-order rows + the set of done steps.
    Tolerates the legacy 'cot_total' / 'imp_plus_cot_*' column names."""
    rows: List[List[str]] = []
    done: set = set()
    if not output_csv.exists():
        return rows, done
    with output_csv.open("r", encoding="utf-8") as f:
        lines = list(csv.reader(f))
    if not lines:
        return rows, done
    old = lines[0]
    for raw in lines[1:]:
        d = {old[i]: (raw[i] if i < len(raw) else "") for i in range(len(old))}
        step = d.get("step", "")
        if not step:
            continue
        done.add(step)
        rows.append([
            d.get("cot_accuracy", ""), d.get("cot_correct", ""),
            d.get("understanding_plus_cot_accuracy", d.get("imp_plus_cot_accuracy", "")),
            d.get("understanding_plus_cot_correct", d.get("imp_plus_cot_correct", "")),
            d.get("u_and_a_accuracy", ""), d.get("u_and_a_correct", ""),
            d.get("cot_total", d.get("total_questions", "")),
            d.get("is_instruct", ""), d.get("run_name", ""), step,
            d.get("checkpoint_path", ""), d.get("base_model", ""), d.get("num_articles", ""),
        ])
    return rows, done


def _write_curve(output_csv: Path, frozen_rows: List[List[str]], results: List[CheckpointEvalResult]) -> None:
    """Overwrite output_csv with header + all rows sorted by step.

    Combines frozen prior rows with freshly evaluated results and sorts by numeric step,
    so each step_k is written at its correct position even when checkpoints are evaluated
    out of order (e.g. descending == last-steps-first)."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows = list(frozen_rows) + [_result_to_row(r) for r in results]

    def _step_num(row: List[Any]) -> int:
        m = re.search(r"(\d+)", str(row[9]))  # index 9 == "step" in _SUMMARY_HEADERS
        return int(m.group(1)) if m else -1

    all_rows.sort(key=_step_num)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_SUMMARY_HEADERS)
        for row in all_rows:
            writer.writerow(row)


def write_summary(results: List[CheckpointEvalResult], output_csv: Path, merge: bool = True) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    # Read existing lines if file exists (only when merging across invocations).
    # merge=False overwrites fresh from `results` -- used for incremental in-loop
    # writes (so a timeout/preemption keeps completed rows without duplicating them).
    existing_lines = []
    if merge and file_exists:
        with output_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_lines = list(reader)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        # Write updated header
        headers = [
            "cot_accuracy",
            "cot_correct",
            "understanding_plus_cot_accuracy",
            "understanding_plus_cot_correct",
            "u_and_a_accuracy",
            "u_and_a_correct",
            "total_questions",
            "is_instruct",
            "run_name",
            "step",
            "checkpoint_path",
            "base_model",
            "num_articles",
        ]
        writer.writerow(headers)
        
        # Re-write existing rows safely using header mapping
        if existing_lines and len(existing_lines) > 0:
            old_header = existing_lines[0]
            for row in existing_lines[1:]:
                row_dict = {old_header[i]: row[i] if i < len(row) else "" for i in range(len(old_header))}
                
                # Resolve single 'total_questions' from older format's cot_total
                tot = row_dict.get("cot_total", row_dict.get("total_questions", ""))

                # Support both the new column names and the legacy "imp_plus_cot_*" ones.
                upc_acc = row_dict.get(
                    "understanding_plus_cot_accuracy", row_dict.get("imp_plus_cot_accuracy", "")
                )
                upc_correct = row_dict.get(
                    "understanding_plus_cot_correct", row_dict.get("imp_plus_cot_correct", "")
                )

                writer.writerow(
                    [
                        row_dict.get("cot_accuracy", ""),
                        row_dict.get("cot_correct", ""),
                        upc_acc,
                        upc_correct,
                        row_dict.get("u_and_a_accuracy", ""),
                        row_dict.get("u_and_a_correct", ""),
                        tot,
                        row_dict.get("is_instruct", ""),
                        row_dict.get("run_name", ""),
                        row_dict.get("step", ""),
                        row_dict.get("checkpoint_path", ""),
                        row_dict.get("base_model", ""),
                        row_dict.get("num_articles", ""),
                    ]
                )
                 
        for r in results:
            writer.writerow(
                [
                    f"{r.cot_only.accuracy:.6f}",
                    r.cot_only.correct,
                    f"{r.understanding_plus_cot.accuracy:.6f}",
                    r.understanding_plus_cot.correct,
                    f"{r.u_and_a.accuracy:.6f}",
                    r.u_and_a.correct,
                    r.cot_only.total_questions,
                    int(r.is_instruct),
                    r.run_name,
                    r.step,
                    r.checkpoint_path,
                    r.base_model,
                    r.num_articles,
                ]
            )


def write_sample_outputs(samples: List[Dict[str, Any]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    existing_samples = []
    if output_json.exists():
        try:
            with output_json.open("r", encoding="utf-8") as f:
                existing_samples = json.load(f)
        except Exception:
            pass
            
    if isinstance(existing_samples, list):
        existing_samples.extend(samples)
    else:
        existing_samples = samples
        
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(existing_samples, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved OAT LoRA checkpoints with deterministic vLLM decoding in two modes: "
            "(1) CoT-only and (2) implication generation + CoT answer."
        )
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Eval data path (.json/.jsonl) or HF dataset name",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Dataset name to find test*.json or test*.jsonl in data/{dataset_name}/",
    )
    parser.add_argument("--split", default="train", help="Dataset split when using HF dataset name")
    parser.add_argument("--input_key", default="article")
    parser.add_argument("--output_key", default="questions")

    parser.add_argument(
        "--checkpoint_root",
        default="oat-output/simpl",
        help="Root directory to recursively discover saved_models/step_* checkpoints",
    )
    parser.add_argument(
        "--checkpoint_dir",
        action="append",
        default=[],
        help="Specific checkpoint directory (can be repeated)",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Optional base model override. By default this is inferred from adapter_config.json",
    )
    parser.add_argument(
        "--no_merge_lora_before_eval",
        dest="merge_lora_before_eval",
        action="store_false",
        help="Disable merging adapters before eval and use runtime LoRA in vLLM.",
    )
    parser.add_argument(
        "--merged_model_root",
        default="oat-output/merged_eval_models",
        help="Output root for merged model directories when --merge_lora_before_eval is enabled.",
    )
    parser.add_argument(
        "--keep_merged_model_after_eval",
        dest="delete_merged_model_after_eval",
        action="store_false",
        help="Keep merged model directories on disk after eval.",
    )
    parser.add_argument(
        "--system_prompt",
        default="You are a helpful assistant.",
        help="System prompt for the chat template.",
    )

    parser.add_argument(
        "--merge_lora_before_eval",
        dest="merge_lora_before_eval",
        action="store_true",
        help="Opt back into merging full models per checkpoint (slow legacy path). Default is "
             "native vLLM LoRA hot-swap (one engine, no merge).",
    )
    parser.set_defaults(
        merge_lora_before_eval=False,
        delete_merged_model_after_eval=True,
    )

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reasoning_max_items", type=int, default=8)
    parser.add_argument("--reasoning_max_tokens", type=int, default=384)
    parser.add_argument("--answer_max_tokens", type=int, default=256)
    parser.add_argument("--cot_samples", type=int, default=1,
                        help="Sample N answers/question and report mean accuracy (avg@N). 1=greedy (default).")
    parser.add_argument("--cot_temperature", type=float, default=0.6,
                        help="Sampling temperature when cot_samples>1.")
    parser.add_argument("--cot_eval_only", action="store_true", default=False,
                        help="Only evaluate plain CoT (skip understanding generation and the "
                             "understanding_plus_cot / understand_and_answer modes). ~3-4x faster; "
                             "skipped modes report 0/0.")

    parser.add_argument("--max_articles", type=int, default=None)
    parser.add_argument("--max_questions_per_article", type=int, default=None)

    parser.add_argument(
        "--understanding_with_passage",
        dest="understanding_with_passage",
        action="store_true",
        default=True,
        help=(
            "Include the passage alongside the understanding when scoring the "
            "understanding_plus_cot mode (default). Match this to the training "
            "config's use_baseline_reward."
        ),
    )
    parser.add_argument(
        "--no_understanding_with_passage",
        dest="understanding_with_passage",
        action="store_false",
        help="Score the understanding_plus_cot mode without the passage (understanding only).",
    )

    parser.add_argument(
        "--is_instruct",
        dest="is_instruct",
        action="store_true",
        default=None,
        help="Force instruct chat templating",
    )
    parser.add_argument(
        "--no_is_instruct",
        dest="is_instruct",
        action="store_false",
        help="Disable instruct chat templating",
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV path. If omitted, a timestamped file is written to oat-output/eval/.",
    )
    parser.add_argument(
        "--sample_output_count_per_checkpoint",
        type=int,
        default=10,
        help="Number of QA generations to randomly sample per checkpoint for JSON output.",
    )
    parser.add_argument(
        "--sample_output_seed",
        type=int,
        default=0,
        help="Base random seed used for sampling output rows.",
    )
    parser.add_argument(
        "--last_n_steps",
        type=int,
        default=0,
        help="If >0, only evaluate the last N checkpoints (by step). For slow datasets.",
    )
    parser.add_argument("--min_step", type=int, default=0,
                        help="Only evaluate checkpoints with step >= this (0 = no lower bound). For partitioning.")
    parser.add_argument("--max_step", type=int, default=0,
                        help="Only evaluate checkpoints with step <= this (0 = no upper bound). For partitioning.")
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Seed for the vLLM engine and all sampling, reset on every model load so "
             "each checkpoint evaluates deterministically (reproducible avg@N).",
    )
    parser.add_argument(
        "--redo_all",
        action="store_true",
        help="Re-evaluate every checkpoint even if it already has a row in --output_csv "
             "(default: skip checkpoints already present, i.e. resume).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    results, sampled_outputs = evaluate_checkpoints(args)

    ts = time.strftime("%Y%m%d_%H%M%S")

    if args.output_csv:
        # Already written incrementally by evaluate_checkpoints (frozen prior rows +
        # new results), so don't rewrite here -- that would drop resumed/skipped rows.
        output_csv = Path(args.output_csv)
    else:
        output_csv = Path("oat-output") / "eval" / f"saved_model_eval_{ts}.csv"
        write_summary(results, output_csv, merge=False)

    sample_output_json = output_csv.with_suffix(".json")
    write_sample_outputs(sampled_outputs, sample_output_json)

    print(f"[eval] Wrote summary to: {output_csv}")
    print(
        f"[eval] Wrote sampled outputs to: {sample_output_json} "
        f"({len(sampled_outputs)} rows)"
    )
    print(f"[eval] Done in {(time.time() - t0):.1f}s for {len(results)} checkpoints")


if __name__ == "__main__":
    main()
