from __future__ import annotations

import functools
import json
import logging
import os
import re
import string
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import vllm
from datasets import load_dataset

from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TransitionData
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from oat.utils.ops import masked_mean, masked_sum

from src.utils.oat_prompt_templates import qa_cot_prompt


from src.utils.parsing_utils import extract_boxed_letter, normalize_gold_letter, parse_questions

def collate_prompt_batch(batch):
    processed_prompts = [item[0] for item in batch]
    raw_prompts = [item[1] for item in batch]
    references = [item[2] for item in batch]
    return processed_prompts, raw_prompts, references


@dataclass
class CoTOnlyArgs(PPOArgs):
    prompt_data: str = ""
    input_key: str = "article"
    output_key: str = "questions"
    train_split: str = "train"
    max_train: int = 999999
    seed: int = 42

    reasoning_num_samples: int = 8
    reasoning_max_tokens: int = 512

    incorrect_reward: float = 0.0
    correct_reward: float = 1.0
    reward_scale: float = 2.0

    critic_type: Literal["ppo", "grpo", "drgrpo"] = field(default="drgrpo")
    remove_len_bias: bool = False
    remove_std_bias: bool = False

    # Keep eval scheduler disabled, but keep online_evaluation enabled so OAT forwards references.
    eval_steps: int = -1
    online_evaluation: bool = True
    apply_chat_template: bool = False
    is_instruct: bool = False
    beta: float = 0.0


def configure_cot_only_args(args: CoTOnlyArgs) -> CoTOnlyArgs:
    if int(args.reasoning_num_samples) < 1:
        raise ValueError("reasoning_num_samples must be >= 1")

    args.algo = "PPO"
    args.oracle = ""
    args.oracle_type = "reward"
    args.is_instruct = bool(args.is_instruct)
    # Keep raw passages in the dataset path; actor-level generation applies template.
    args.apply_chat_template = False
    args.online_evaluation = True
    args.eval_steps = -1
    args.beta = 0.04
    if 'qwen' in args.pretrain.lower():
        args.use_fused_lm_head = False

    args.num_samples = int(args.reasoning_num_samples)
    minimum_buffer = int(args.rollout_batch_size_per_device) * int(args.num_samples)
    if args.pi_buffer_maxlen_per_device < minimum_buffer:
        args.pi_buffer_maxlen_per_device = minimum_buffer
    return args


class CoTOnlyActor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.reasoning_num_samples = int(self.args.reasoning_num_samples)
        self.reasoning_max_samples = int(self.args.reasoning_max_tokens)
        self.correct_reward = float(self.args.correct_reward)
        self.incorrect_reward = float(self.args.incorrect_reward)
        self.is_instruct = bool(getattr(self.args, "is_instruct", False))
        self.scale_reward = float(getattr(self.args, "reward_scale", 2.0))

        base_seed = int(getattr(self.args, "seed", 0))
        self.rng = np.random.default_rng(base_seed + int(actor_id))

        self.sampling_params.stop = None
        self.sampling_params.stop_token_ids = None

    def _build_sampling_params(
        self,
        *,
        temperature: float,
        max_tokens: int,
        with_logprobs: bool,
    ):
        return vllm.SamplingParams(
            temperature=temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
            max_tokens=max_tokens,
            n=1,
            logprobs=1 if with_logprobs else None,
        )

    def _fallback_token_id(self) -> int:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        return int(eos_id) if eos_id is not None else 0

    def _terminal_reward(self, token_count: int, reward: float) -> List[float]:
        n = max(1, int(token_count))
        dense = [0.0] * n
        dense[-1] = float(reward)
        return dense

    def _extract_output(self, output) -> Tuple[List[int], str, List[int], List[float], bool]:
        completion = output.outputs[0]
        prompt_ids = list(output.prompt_token_ids or [])
        text = completion.text or ""
        token_ids = list(completion.token_ids or [])
        if not token_ids:
            token_ids = [self._fallback_token_id()]

        logprobs = []
        if completion.logprobs:
            for i, token_id in enumerate(token_ids):
                if i >= len(completion.logprobs):
                    break
                token_map = completion.logprobs[i]
                if token_map and token_id in token_map:
                    logprobs.append(token_map[token_id].logprob)
                else:
                    logprobs.append(0.0)

        if len(logprobs) != len(token_ids):
            logprobs = [0.0] * len(token_ids)

        is_truncated = completion.finish_reason == "length"
        return prompt_ids, text, token_ids, logprobs, is_truncated

    def _train_question_for_doc(self, questions: List[Dict]) -> Optional[Dict]:
        valid = []
        for q in questions:
            q_text = q.get("question", "")
            opts = q.get("options", [])
            gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
            if isinstance(opts, list) and len(opts) >= 2 and q_text and gold:
                valid.append({"question": q_text, "options": opts, "answer": gold})

        if not valid:
            return None
        picked_idx = int(self.rng.integers(0, len(valid)))
        return valid[picked_idx]

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TransitionData]:
        del formatted_prompts
        assert not self.eval_mode

        t0 = time.time()
        if not prompts:
            return self.ipc_client.serialize_ipc([])
        if references is None:
            references = [None] * len(prompts)

        qa_prompts = []
        qa_meta = []
        for doc_idx, article in enumerate(prompts):
            doc_questions = parse_questions(references[doc_idx])
            train_q = self._train_question_for_doc(doc_questions)

            prompt_text = qa_cot_prompt(
                article=article,
                question_text=train_q["question"],
                options=train_q["options"],
            )
            for _ in range(self.reasoning_num_samples):
                if self.is_instruct and hasattr(self.tokenizer, "apply_chat_template"):
                    prompt_str = self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt_str = prompt_text
                    
                qa_prompts.append(prompt_str)
                qa_meta.append((doc_idx, train_q["answer"], False, len(train_q["options"])))

        qa_outputs = self.generate(
            qa_prompts,
            self._build_sampling_params(
                temperature=self.sampling_params.temperature,
                max_tokens=self.reasoning_max_samples,
                with_logprobs=True,
            ),
        )

        all_rewards = []
        valid_count = 0
        correct_count = 0
        trajectory_data: List[TransitionData] = []
        for out, meta, prompt_text in zip(qa_outputs, qa_meta, qa_prompts):
            doc_idx, gold, is_placeholder, num_opt = meta
            prompt_ids, text, token_ids, logprobs, is_truncated = self._extract_output(out)

            if is_placeholder:
                reward = 0.0
                loss_mask = False
            else:
                pred = extract_boxed_letter(text, num_opt)
                if pred == gold:
                    reward = self.correct_reward
                elif pred in [chr(ord('A') + i) for i in range(num_opt)]:
                    reward = 0.1
                else:
                    reward = self.incorrect_reward
                
                reward = reward * self.scale_reward
                
                loss_mask = True
                valid_count += 1
                correct_count += int(pred == gold)

            if self.args.ignore_no_eos and is_truncated:
                loss_mask = False

            all_rewards.append(float(reward))
            info = {
                "actor/num_documents": float(len(prompts)),
                "actor/qa_samples": float(self.reasoning_num_samples),
                "actor/qa_reward_mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "actor/qa_accuracy": float(correct_count / max(valid_count, 1)),
                "actor/step_time": float(time.time() - t0),
            }

            trajectory_data.append(
                TransitionData(
                    prompt=prompt_text,
                    prompt_ids=prompt_ids,
                    response=text,
                    response_ids=token_ids,
                    response_logprobs=logprobs,
                    rewards=self._terminal_reward(len(token_ids), float(reward)),
                    loss_mask=loss_mask,
                    info=info,
                )
            )

        expected = len(prompts) * self.reasoning_num_samples
        assert len(trajectory_data) == expected
        logging.info(
            "CoT-only actor done: docs=%d transitions=%d per_doc=%d",
            len(prompts),
            len(trajectory_data),
            self.reasoning_num_samples,
        )
        return self.ipc_client.serialize_ipc(trajectory_data)


class CoTOnlyLearner(PPOLearner):
    def _init(self, args: CoTOnlyArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.args = args
        self.args.max_queries = np.inf
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.reasoning_max_tokens)
            if args.critic_type == "drgrpo"
            else masked_mean
        )
        if args.critic_type in ["grpo", "ppo"] and args.remove_len_bias:
            self.masked_aggregator = functools.partial(
                masked_sum,
                constant_normalizer=args.reasoning_max_tokens,
            )

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        rewards = rewards.sum(-1)
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if (self.args.critic_type in ["grpo"]) and (not self.args.remove_std_bias):
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples,
                dim=0,
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages

    def _load_prompt_data(self):
        prompt_data = self.args.prompt_data
        if os.path.isfile(prompt_data):
            ext = os.path.splitext(prompt_data)[1].lower()
            if ext in {".jsonl", ".json"}:
                logging.info("Loading prompt data from JSON file: %s", prompt_data)
                return load_dataset("json", data_files=prompt_data, split="train")
        return load_data_from_disk_or_hf(prompt_data)

    def prepare_data(self, strategy, tokenizer):
        data_obj = self._load_prompt_data()
        if hasattr(data_obj, "keys") and self.args.train_split in data_obj:
            train_dataset = data_obj[self.args.train_split]
        else:
            train_dataset = data_obj

        input_key = self.args.input_key
        if input_key not in train_dataset.column_names:
            input_key = "article"
        output_key = self.args.output_key
        if output_key not in train_dataset.column_names:
            output_key = "questions"

        def flatten_questions(batch):
            new_articles = []
            new_questions = []
            for article, qs_json in zip(batch[input_key], batch[output_key]):
                qs = parse_questions(qs_json)
                for q in qs:
                    q_text = q.get("question", "")
                    opts = q.get("options", [])
                    gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
                    if isinstance(opts, list) and len(opts) >= 2 and q_text and gold:
                        new_articles.append(article)
                        # Store as a JSON string to prevent Hugging Face datasets from mangling the types
                        new_questions.append(json.dumps([{"question": q_text, "options": opts, "answer": gold}]))
            return {input_key: new_articles, output_key: new_questions}

        train_dataset = train_dataset.map(
            flatten_questions,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        # train_dataset = train_dataset.shuffle(seed=42)

        max_train = min(int(self.args.max_train), len(train_dataset))
        train_dataset = train_dataset.select(range(max_train))

        train_dataset = train_dataset.select_columns([input_key, output_key])

        self.prompts_dataset = PromptDataset(
            train_dataset,
            tokenizer,
            strategy,
            input_key=input_key,
            output_key=output_key,
            apply_chat_template=False,
            get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_prompt_batch,
        )

        self.eval_prompts_dataset = None
        self.eval_prompts_dataloader = None

    def evaluate(self, dataloader, steps):
        del dataloader, steps
        return {}


def run_cot_only_oat(args: CoTOnlyArgs):
    args = configure_cot_only_args(args)
    args = default_args_validation(args)

    program, local_resources = get_program(
        args,
        learner_cls=CoTOnlyLearner,
        actor_cls=CoTOnlyActor,
    )
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    cli_args: CoTOnlyArgs = get_default_args(CoTOnlyArgs)
    run_cot_only_oat(cli_args)