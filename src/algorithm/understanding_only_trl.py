from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from src.utils.oat_prompt_templates import (
    qa_cot_prompt,
    qa_eval_understanding_only_prompt,
    understanding_prompt,
)
from src.utils.preprocess_data import create_or_load_preprocessed_data
from src.utils.parsing_utils import (
    extract_boxed_letter,
    normalize_gold_letter,
    parse_questions,
    valid_questions,
)
from src.utils.preprocess_data import DATASET_REGISTRY


def completion_to_text(completion: Any) -> str:
    if (
        isinstance(completion, list)
        and len(completion) > 0
        and isinstance(completion[-1], dict)
        and "content" in completion[-1]
    ):
        return str(completion[-1]["content"])
    return str(completion or "")


def extract_understanding_from_tags(text: str) -> str:
    match = re.search(r"<understanding>\s*(.*?)\s*</understanding>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _maybe_prepare_data(config: dict[str, Any]) -> str | None:
    preprocess_cfg = config.get("preprocess", {})
    if not isinstance(preprocess_cfg, dict):
        raise ValueError("preprocess must be a JSON object when provided")

    if not bool(preprocess_cfg.get("enabled", False)):
        return None

    output_dir = str(preprocess_cfg.get("output_dir", "data"))
    dataset_name = str(preprocess_cfg.get("dataset_name", "race-c"))
    split = str(preprocess_cfg.get("split", "train"))
    subset_raw = preprocess_cfg.get("subset", None)
    subset = None if subset_raw is None else str(subset_raw)
    seed = int(preprocess_cfg.get("seed", 42))
    num_samples_raw = preprocess_cfg.get("num_samples", None)
    num_samples = None if num_samples_raw is None else int(num_samples_raw)

    _, output_path = create_or_load_preprocessed_data(
        num_samples=num_samples,
        split=split,
        subset=subset,
        seed=seed,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )
    return os.path.abspath(output_path)


@dataclass
class UnderstandingReward:
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    qa_num_samples: int = 1
    qa_eval_max_tokens: int = 128
    qa_eval_temperature: float = 1.0
    conciseness_penalty_k: float = 0.3
    use_baseline_reward: bool = True
    baseline_forgiven: int = 0
    reward_scale: float = 5.0
    num_options: int = 4

    def __post_init__(self):
        self.__name__ = "understanding_reward"
        self.trainer: Optional[GRPOTrainer] = None
        self._baseline_cache: dict[str, Tuple[float, int, int]] = {}

    def attach_trainer(self, trainer: GRPOTrainer) -> None:
        self.trainer = trainer

    def __call__(
        self,
        prompts,
        completions,
        article: Sequence[str],
        questions: Sequence[Any],
        **kwargs,
    ) -> list[float]:
        del prompts, kwargs
        
        all_qa_prompts = []
        tasks = []
        
        for completion, passage, reference in zip(completions, article, questions):
            qs = valid_questions(reference)
            if not qs:
                tasks.append({"has_questions": False})
                continue
                
            understanding = extract_understanding_from_tags(completion_to_text(completion))
            
            task = {
                "has_questions": True,
                "qs": qs,
                "understanding": understanding,
                "passage": passage,
                "understanding_start_idx": len(all_qa_prompts),
                "understanding_end_idx": -1,
                "baseline_start_idx": -1,
                "baseline_end_idx": -1,
                "cache_key": None
            }
            
            for q in qs:
                all_qa_prompts.append(qa_eval_understanding_only_prompt(understanding, q["question"], q["options"]))
            task["understanding_end_idx"] = len(all_qa_prompts)
            
            if self.use_baseline_reward:
                cache_key = json.dumps(
                    {"article": passage, "questions": qs, "n": self.qa_num_samples},
                    sort_keys=True,
                    ensure_ascii=True,
                )
                task["cache_key"] = cache_key
                if cache_key not in self._baseline_cache:
                    task["baseline_start_idx"] = len(all_qa_prompts)
                    for q in qs:
                        all_qa_prompts.append(qa_cot_prompt(passage, q["question"], q["options"]))
                    task["baseline_end_idx"] = len(all_qa_prompts)
                    
            tasks.append(task)
            
        all_qa_outputs = []
        if all_qa_prompts:
            all_qa_outputs = self._generate_qa(all_qa_prompts)
            
        rewards = []
        for task in tasks:
            if not task["has_questions"]:
                rewards.append(0.0)
                continue
                
            qs = task["qs"]
            golds = [q["answer"] for q in qs]
            
            if self.use_baseline_reward:
                cache_key = task["cache_key"]
                if cache_key not in self._baseline_cache:
                    b_start = task["baseline_start_idx"]
                    b_end = task["baseline_end_idx"]
                    b_outputs = all_qa_outputs[b_start:b_end]
                    
                    b_correct = 0
                    for generated, gold in zip(b_outputs, golds):
                        for text in generated:
                            b_correct += int(extract_boxed_letter(text, self.num_options) == gold)
                            
                    b_total = len(golds) * self.qa_num_samples
                    b_reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * (b_correct / max(b_total, 1)) if b_total > 0 else 0.0
                    self._baseline_cache[cache_key] = (b_reward, b_correct, b_total)
                    
            u_start = task["understanding_start_idx"]
            u_end = task["understanding_end_idx"]
            u_outputs = all_qa_outputs[u_start:u_end]
            
            u_correct = 0
            for generated, gold in zip(u_outputs, golds):
                for text in generated:
                    u_correct += int(extract_boxed_letter(text, self.num_options) == gold)
                    
            u_total = len(golds) * self.qa_num_samples
            reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * (u_correct / max(u_total, 1)) if u_total > 0 else 0.0
            
            if self.use_baseline_reward:
                b_reward, b_correct, _ = self._baseline_cache[task["cache_key"]]
                if b_correct > u_correct + self.baseline_forgiven:
                    reward = -1.0
                else:
                    reward = (reward - b_reward) * self.reward_scale
            else:
                reward = reward * self.reward_scale
                
            reward -= self._conciseness_penalty(task["passage"], task["understanding"])
            rewards.append(float(reward))
            
        return rewards

    def _generate_qa(self, prompts: List[str]) -> List[List[str]]:
        if self.trainer is None:
            raise RuntimeError("UnderstandingReward must be attached to the GRPOTrainer before training.")

        llm = getattr(self.trainer, "llm", None)
        if llm is not None:
            sampling_params = None
            try:
                import vllm
                sampling_params = vllm.SamplingParams(
                    temperature=self.qa_eval_temperature if self.qa_num_samples > 1 else 0.0,
                    max_tokens=self.qa_eval_max_tokens,
                    n=self.qa_num_samples,
                )
            except ImportError:
                pass
                
            all_outputs = []
            chunk_size = 256
            for i in range(0, len(prompts), chunk_size):
                chunk = prompts[i : i + chunk_size]
                try:
                    outputs = llm.generate(chunk, sampling_params, use_tqdm=False)
                except TypeError:
                    outputs = llm.generate(chunk, sampling_params)
                all_outputs.extend([[completion.text or "" for completion in out.outputs] for out in outputs])
            return all_outputs

        model = getattr(self.trainer, "model", None)
        tokenizer = getattr(self.trainer, "processing_class", None) or getattr(self.trainer, "tokenizer", None)
        if model is None or tokenizer is None:
            raise RuntimeError("No vLLM engine or model/tokenizer pair found on GRPOTrainer for QA reward generation.")

        device = next(model.parameters()).device
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_kwargs = {
            "do_sample": self.qa_num_samples > 1,
            "max_new_tokens": self.qa_eval_max_tokens,
            "num_return_sequences": self.qa_num_samples,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        }
        if self.qa_num_samples > 1:
            generation_kwargs["temperature"] = self.qa_eval_temperature

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )

        prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()
        decoded = []
        cursor = 0
        for prompt_len in prompt_lens:
            per_prompt = []
            for _ in range(self.qa_num_samples):
                output_ids = generated[cursor][int(prompt_len):]
                per_prompt.append(tokenizer.decode(output_ids, skip_special_tokens=True))
                cursor += 1
            decoded.append(per_prompt)
        return decoded

    def _conciseness_penalty(self, article: str, understanding: str) -> float:
        tokenizer = None
        if self.trainer is not None:
            tokenizer = getattr(self.trainer, "processing_class", None) or getattr(self.trainer, "tokenizer", None)

        if tokenizer is not None and hasattr(tokenizer, "encode"):
            passage_len = len(tokenizer.encode(article))
            understanding_len = len(tokenizer.encode(understanding))
        else:
            passage_len = len(article.split())
            understanding_len = len(understanding.split())

        if understanding_len > 0.75 * max(passage_len, 1):
            return (understanding_len / max(passage_len, 1)) * self.conciseness_penalty_k
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    oat_args = config.get("oat_args", {})
    prepared_data_path = _maybe_prepare_data(config)

    prompt_data = prepared_data_path if prepared_data_path else oat_args.get("prompt_data")
    if not prompt_data:
        raise ValueError("Must provide prompt_data in oat_args or enable preprocess")

    model_name_or_path = oat_args.get("pretrain", "Qwen/Qwen2.5-3B-Instruct")
    dataset_name = config.get("preprocess", {}).get("dataset_name", "race-c")
    base_run_name = oat_args.get("wb_run_name") or os.path.basename(str(model_name_or_path).rstrip("/"))
    run_suffix = f"{base_run_name}-trl-understanding-only-{datetime.now().strftime('%m%d_T%H')}"
    output_dir = os.path.join(oat_args.get("save_path", "oat-output"), dataset_name, run_suffix)

    reasoning_num_samples = int(oat_args.get("reasoning_num_samples", oat_args.get("num_generations", 4)))
    max_prompt_length = int(oat_args.get("prompt_max_length", 2048))
    max_completion_length = int(oat_args.get("reasoning_max_tokens", oat_args.get("generate_max_length", 512)))
    learning_rate = float(oat_args.get("learning_rate", 1e-6))
    per_device_train_batch_size = int(oat_args.get("train_batch_size_per_device", 1))

    if "gradient_accumulation_steps" in oat_args and not isinstance(oat_args["gradient_accumulation_steps"], bool):
        grad_acc = int(oat_args["gradient_accumulation_steps"])
    else:
        total_batch = int(oat_args.get("train_batch_size", per_device_train_batch_size))
        gpus = int(oat_args.get("gpus", 1))
        grad_acc = max(1, total_batch // (per_device_train_batch_size * gpus))

    use_wb = bool(oat_args.get("use_wb", False))
    if use_wb:
        os.environ["WANDB_PROJECT"] = oat_args.get("wb_project", "SImpL-new")
        if oat_args.get("wb_org"):
            os.environ["WANDB_ENTITY"] = oat_args["wb_org"]

    if os.path.isfile(prompt_data):
        dataset = load_dataset("json", data_files=prompt_data, split="train")
    else:
        dataset = load_dataset(prompt_data, split=oat_args.get("train_split", "train"))

    input_key = oat_args.get("input_key", "article")
    output_key = oat_args.get("output_key", "questions")
    if input_key not in dataset.column_names:
        input_key = "article"
    if output_key not in dataset.column_names:
        output_key = "questions"

    def format_dataset(examples):
        prompts = []
        articles = []
        refs = []
        for article, reference in zip(examples[input_key], examples[output_key]):
            qs = valid_questions(reference)
            if not qs:
                continue
            prompts.append(understanding_prompt(article))
            articles.append(article)
            refs.append(json.dumps(qs, ensure_ascii=True))
        return {"prompt": prompts, "article": articles, "questions": refs}

    dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=int(oat_args.get("seed", 42)))

    if "max_train" in oat_args:
        dataset = dataset.select(range(min(len(dataset), int(oat_args["max_train"]))))

    def apply_chat_format(examples):
        if not bool(oat_args.get("is_instruct", False)):
            return {"prompt": examples["prompt"]}
        return {"prompt": [[{"role": "user", "content": prompt}] for prompt in examples["prompt"]]}

    dataset = dataset.map(apply_chat_format, batched=True)

    num_options = DATASET_REGISTRY.get(dataset_name, {}).get("num_options", 4)
    reward_func = UnderstandingReward(
        correct_reward=float(oat_args.get("correct_reward", 1.0)),
        incorrect_reward=float(oat_args.get("incorrect_reward", 0.0)),
        qa_num_samples=int(oat_args.get("qa_num_samples", 1)),
        qa_eval_max_tokens=int(oat_args.get("qa_eval_max_tokens", 128)),
        qa_eval_temperature=float(oat_args.get("qa_eval_temperature", 1.0)),
        conciseness_penalty_k=float(oat_args.get("conciseness_penalty_k", 0.3)),
        use_baseline_reward=bool(oat_args.get("use_baseline_reward", True)),
        baseline_forgiven=int(oat_args.get("baseline_forgiven", 0)),
        reward_scale=float(oat_args.get("reward_scale", 5.0)),
        num_options=num_options,
    )

    training_args = GRPOConfig(
        num_train_epochs=oat_args.get("num_prompt_epoch", 3),
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc,
        num_generations=reasoning_num_samples,
        # max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        use_vllm=bool(oat_args.get("use_vllm", True)),
        vllm_gpu_memory_utilization=float(oat_args.get("vllm_gpu_ratio", 0.4)),
        logging_steps=int(oat_args.get("logging_steps", 1)),
        save_steps=int(oat_args.get("save_steps", 200)),
        bf16=bool(oat_args.get("bf16", False)),
        run_name=run_suffix if use_wb else None,
        report_to="wandb" if use_wb else "none",
        loss_type=oat_args.get("loss_type", "dr_grpo"),
    )

    peft_config = None
    if "lora_rank" in oat_args:
        peft_config = LoraConfig(
            r=int(oat_args["lora_rank"]),
            lora_alpha=int(oat_args.get("lora_alpha", 32)),
            lora_dropout=float(oat_args.get("lora_dropout", 0.05)),
            target_modules=oat_args.get("target_modules", ["q_proj", "v_proj"]),
        )

    trainer = GRPOTrainer(
        model=model_name_or_path,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    reward_func.attach_trainer(trainer)

    logging.info("Starting understanding-only TRL run with %d training passages", len(dataset))
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
