from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer, ModelConfig
from datetime import datetime

from src.utils.oat_prompt_templates import qa_cot_prompt
from src.utils.preprocess_data import create_or_load_preprocessed_data
from src.utils.parsing_utils import extract_boxed_letter, normalize_gold_letter, parse_questions
from src.utils.preprocess_data import create_or_load_preprocessed_data, DATASET_REGISTRY

def get_correctness_reward_func(dataset_name="race-c"):
    num_options = DATASET_REGISTRY.get(dataset_name, {}).get("num_options", 4)
    valid_letters = [chr(ord('A') + i) for i in range(num_options)]

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        rewards = []
        for completion, expected_answer in zip(completions, answer):
            # TRL GRPO may pass chat-formatted completions: e.g. [{"role": "assistant", "content": "..."}]
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict) and "content" in completion[-1]:
                completion_text = completion[-1]["content"]
            else:
                completion_text = str(completion)
                
            pred = extract_boxed_letter(completion_text, num_options)
            if pred == expected_answer:
                rewards.append(1.0)
            elif pred in valid_letters:
                rewards.append(0.1)
            else:
                rewards.append(0.0)
        return rewards
    return correctness_reward_func

def _maybe_prepare_data(config: dict[str, Any]) -> str | None:
    preprocess_cfg = config.get("preprocess", {})
    if not isinstance(preprocess_cfg, dict):
        raise ValueError("preprocess must be a JSON object when provided")

    enabled = bool(preprocess_cfg.get("enabled", False))
    if not enabled:
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

    # Map oat_args to TRL equivalents
    model_name_or_path = oat_args.get("pretrain", "Qwen/Qwen2.5-3B-Instruct")
    dataset_name = config.get('preprocess', {}).get('dataset_name', 'race-c')
    output_dir = oat_args.get("save_path", "oat-output") + f"/{dataset_name}" + f"/{oat_args.get('wb_run_name', '')}-trl-cot-only-{datetime.now().strftime('%m%d_T%H')}"
    
    # Model/Generation
    qa_num_samples = oat_args.get("reasoning_num_samples", 4) # or reasoning_num_samples
    max_prompt_length = oat_args.get("prompt_max_length", 2048)
    max_completion_length = oat_args.get("generate_max_length", 512)
    learning_rate = oat_args.get("learning_rate", 1e-6)
    
    # Batches
    per_device_train_batch_size = oat_args.get("train_batch_size_per_device", 1)
    
    # Check if gradient_accumulation_steps is present and is an int
    if "gradient_accumulation_steps" in oat_args:
        grad_acc_raw = oat_args.get("gradient_accumulation_steps")
        grad_acc = int(grad_acc_raw) if not isinstance(grad_acc_raw, bool) else 1
    else:
        # Infer from train_batch_size if missing
        total_batch = oat_args.get("train_batch_size", per_device_train_batch_size)
        gpus = oat_args.get("gpus", 1)
        grad_acc = max(1, total_batch // (per_device_train_batch_size * gpus))
    
    # vLLM
    vllm_ratio = oat_args.get("vllm_gpu_ratio", 0.4)

    # W&B
    use_wb = False #oat_args.get("use_wb", False)
    wb_project = oat_args.get("wb_project", "SImpL-new")
    ## add time to run name
    wb_run_name = oat_args.get("wb_run_name", "") + f"-trl-cot-only-{datetime.now().strftime('%m%d_T%H')}"
    
    if use_wb:
        os.environ["WANDB_PROJECT"] = wb_project
        if oat_args.get("wb_org"):
            os.environ["WANDB_ENTITY"] = oat_args["wb_org"]

    # Load data
    if os.path.isfile(prompt_data):
        dataset = load_dataset("json", data_files=prompt_data, split="train")
    else:
        dataset = load_dataset(prompt_data, split="train")

    def format_dataset(examples):
        new_prompts = []
        new_answers = []
        for article, qs_json in zip(examples["article"], examples["questions"]):
            qs = parse_questions(qs_json)
            for q in qs:
                q_text = q.get("question", "")
                opts = q.get("options", [])
                gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
                if isinstance(opts, list) and len(opts) >= 2 and q_text and gold:
                    prompt_text = qa_cot_prompt(article=article, question_text=q_text, options=opts)
                    new_prompts.append(prompt_text)
                    new_answers.append(gold)
        return {"prompt": new_prompts, "answer": new_answers}

    dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=42)
    
    def apply_chat_format(examples):
        return {
            "prompt": [[{"role": "user", "content": prompt}] for prompt in examples["prompt"]]
        }
    dataset = dataset.map(apply_chat_format, batched=True)

    if "max_train" in oat_args:
        dataset = dataset.select(range(min(len(dataset), oat_args["max_train"])))

    training_args = GRPOConfig(
        num_train_epochs=oat_args.get("num_prompt_epoch", 3),
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc,
        num_generations=qa_num_samples,
        max_completion_length=max_completion_length,
        use_vllm=True,
        vllm_gpu_memory_utilization=vllm_ratio,
        logging_steps=1,
        save_steps=oat_args.get("save_steps", 200),
        bf16=oat_args.get("bf16", False),
        run_name=wb_run_name if use_wb else None,
        report_to="wandb" if use_wb else "none",
        loss_type="dr_grpo",
    )

    peft_config = None
    if "lora_rank" in oat_args:
        peft_config = LoraConfig(
            r=oat_args["lora_rank"],
            lora_alpha=oat_args.get("lora_alpha", 32),
            lora_dropout=oat_args.get("lora_dropout", 0.05),
            target_modules=oat_args.get("target_modules", ["q_proj", "v_proj"])
        )

    trainer = GRPOTrainer(
        model=model_name_or_path,
        reward_funcs=[get_correctness_reward_func(dataset_name=dataset_name)],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config
    )
    trainer.train()
    ## save final model and tokenizer

if __name__ == "__main__":
    main()
