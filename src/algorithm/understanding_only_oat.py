from __future__ import annotations

import functools
import json
import logging
import os
import re
import string
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

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

from src.utils.oat_prompt_templates import understanding_prompt, maybe_apply_chat_template, qa_cot_prompt


def extract_boxed_letter(text: str) -> str:
    matches = re.findall(r"\\boxed\s*\{\s*([A-Da-d])\s*\}", text or "")
    if matches:
        return matches[-1].upper()

    tail = text[-128:] if text else ""
    plain = re.findall(r"\b([A-Da-d])\b", tail)
    if plain:
        return plain[-1].upper()
    return ""


def normalize_gold_letter(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip().upper()
    if len(v) == 1 and v in string.ascii_uppercase[:4]:
        return v
    return ""


def parse_questions(reference_obj) -> List[Dict]:
    if isinstance(reference_obj, list):
        return [q for q in reference_obj if isinstance(q, dict)]

    if isinstance(reference_obj, dict) and isinstance(reference_obj.get("questions"), list):
        return [q for q in reference_obj["questions"] if isinstance(q, dict)]

    if isinstance(reference_obj, str):
        try:
            parsed = json.loads(reference_obj)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
            return [q for q in parsed["questions"] if isinstance(q, dict)]

    return []


def collate_prompt_batch(batch):
    processed_prompts = [item[0] for item in batch]
    raw_prompts = [item[1] for item in batch]
    references = [item[2] for item in batch]
    return processed_prompts, raw_prompts, references


@dataclass
class UnderstandingOnlyArgs(PPOArgs):
    prompt_data: str = ""
    input_key: str = "article"
    output_key: str = "questions"
    train_split: str = "train"
    max_train: int = 999999

    reasoning_num_samples: int = 8
    reasoning_max_tokens: int = 512
    qa_eval_max_tokens: int = 128
    qa_num_samples: int = 1
    qa_eval_temperature: float = 1.0

    incorrect_reward: float = 0.0
    correct_reward: float = 1.0
    conciseness_penalty_k: float = 0.3
    use_baseline_reward: bool = True
    baseline_forgiven: int = 0

    critic_type: Literal["ppo", "grpo", "drgrpo"] = field(default="drgrpo")
    remove_len_bias: bool = False
    remove_std_bias: bool = False

    eval_steps: int = -1
    online_evaluation: bool = True
    apply_chat_template: bool = False
    is_instruct: bool = False
    beta: float = 0.0


def configure_understandings_only_args(args: UnderstandingOnlyArgs) -> UnderstandingOnlyArgs:
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
    args.beta = 0.0

    args.num_samples = int(args.reasoning_num_samples)
    minimum_buffer = int(args.rollout_batch_size_per_device) * int(args.num_samples)
    if args.pi_buffer_maxlen_per_device < minimum_buffer:
        args.pi_buffer_maxlen_per_device = minimum_buffer
    return args


class UnderstandingOnlyActor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.reasoning_num_samples = int(self.args.reasoning_num_samples)
        self.reasoning_max_tokens = int(self.args.reasoning_max_tokens)
        self.qa_eval_max_tokens = int(self.args.qa_eval_max_tokens)
        self.qa_num_samples = int(getattr(self.args, "qa_num_samples", 1))
        self.qa_eval_temperature = float(getattr(self.args, "qa_eval_temperature", 1.0))
        self.correct_reward = float(self.args.correct_reward)
        self.incorrect_reward = float(self.args.incorrect_reward)
        self.conciseness_penalty_k = float(self.args.conciseness_penalty_k)
        self.use_baseline_reward = bool(getattr(self.args, "use_baseline_reward", True))
        self.baseline_forgiven = int(getattr(self.args, "baseline_forgiven", 0))
        self.is_instruct = bool(getattr(self.args, "is_instruct", False))
        self.scale_reward = getattr(self.args, "reward_scale", 5.0)

        self.sampling_params.stop = None
        self.sampling_params.stop_token_ids = None
        self.eval_sampling_params.stop = None
        self.eval_sampling_params.stop_token_ids = None

    def _build_sampling_params(
        self,
        *,
        temperature: float,
        max_tokens: int,
        with_logprobs: bool,
        n: int = 1,
    ):
        return vllm.SamplingParams(
            temperature=temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
            max_tokens=max_tokens,
            n=n,
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

    def _extract_text_only(self, output) -> str:
        completion = output.outputs[0]
        return completion.text or ""

    def _score_document_baseline(
        self,
        article: str,
        questions: List[Dict],
    ) -> Tuple[float, int, int]:
        if not questions:
            return 0.0, 0, 0

        eval_prompts = []
        golds = []
        for q in questions:
            q_text = q.get("question", "")
            opts = q.get("options", [])
            gold = normalize_gold_letter(q.get("answer", ""))
            if not isinstance(opts, list) or len(opts) < 4 or not q_text or not gold:
                continue
            eval_prompts.append(
                maybe_apply_chat_template(
                    self.tokenizer,
                    qa_cot_prompt(article, q_text, opts),
                    self.is_instruct,
                )
            )
            golds.append(gold)

        if not eval_prompts:
            return 0.0, 0, 0

        eval_outputs = self.generate(
            eval_prompts,
            self._build_sampling_params(
                temperature=self.qa_eval_temperature if self.qa_num_samples > 1 else 0.0,
                max_tokens=self.qa_eval_max_tokens,
                with_logprobs=False,
                n=self.qa_num_samples,
            ),
        )

        correct = 0
        for out, gold in zip(eval_outputs, golds):
            for completion in out.outputs:
                pred = extract_boxed_letter(completion.text or "")
                correct += int(pred == gold)

        total = len(golds) * self.qa_num_samples
        if total == 0:
            return 0.0, 0, 0

        acc = float(correct) / float(total)
        reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * acc
        return reward, correct, total

    def _extract_understanding_from_tags(self, text: str) -> str:
        match = re.search(r"<understanding>\s*(.*?)\s*</understanding>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _score_one_understanding(
        self,
        article: str,
        understandings_text: str,
        questions: List[Dict],
    ) -> Tuple[float, int, int]:
        if not questions:
            return 0.0, 0, 0
        
        extracted_understanding = self._extract_understanding_from_tags(understandings_text)

        eval_prompts = []
        golds = []
        for q in questions:
            q_text = q.get("question", "")
            opts = q.get("options", [])
            gold = normalize_gold_letter(q.get("answer", ""))
            if not isinstance(opts, list) or len(opts) < 4 or not q_text or not gold:
                continue
            eval_prompts.append(
                maybe_apply_chat_template(
                    self.tokenizer,
                    qa_cot_prompt(extracted_understanding, q_text, opts),
                    self.is_instruct,
                )
            )
            golds.append(gold)

        if not eval_prompts:
            return 0.0, 0, 0

        eval_outputs = self.generate(
            eval_prompts,
            self._build_sampling_params(
                temperature=self.qa_eval_temperature if self.qa_num_samples > 1 else 0.0,
                max_tokens=self.qa_eval_max_tokens,
                with_logprobs=False,
                n=self.qa_num_samples,
            ),
        )

        correct = 0
        for out, gold in zip(eval_outputs, golds):
            for completion in out.outputs:
                pred = extract_boxed_letter(completion.text or "")
                correct += int(pred == gold)

        total = len(golds) * self.qa_num_samples
        if total == 0:
            return 0.0, 0, 0

        acc = float(correct) / float(total)
        reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * acc
        return reward, correct, total

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

        understanding_prompts = []
        understanding_owner = []
        for doc_idx, article in enumerate(prompts):
            prompt_text = understanding_prompt(article)
            for _ in range(self.reasoning_num_samples):
                understanding_prompts.append(
                    maybe_apply_chat_template(
                        self.tokenizer,
                        prompt_text,
                        self.is_instruct,
                    )
                )
                understanding_owner.append(doc_idx)

        imp_outputs = self.generate(
            understanding_prompts,
            self._build_sampling_params(
                temperature=self.sampling_params.temperature,
                max_tokens=self.reasoning_max_tokens,
                with_logprobs=True,
            ),
        )

        understanding_data: List[List[Dict]] = [[] for _ in range(len(prompts))]
        for out, owner, prompt_text in zip(imp_outputs, understanding_owner, understanding_prompts):
            prompt_ids, text, token_ids, logprobs, is_truncated = self._extract_output(out)
            understanding_data[owner].append(
                {
                    "prompt": prompt_text,
                    "prompt_ids": prompt_ids,
                    "response": text,
                    "response_ids": token_ids,
                    "response_logprobs": logprobs,
                    "is_truncated": is_truncated,
                }
            )

        understanding_rewards: Dict[Tuple[int, int], float] = {}
        valid_doc_flags: Dict[int, bool] = {}
        total_eval_q = 0
        total_eval_correct = 0

        for doc_idx, article in enumerate(prompts):
            questions = parse_questions(references[doc_idx])
            valid_questions = []
            for q in questions:
                q_text = q.get("question", "")
                opts = q.get("options", [])
                gold = normalize_gold_letter(q.get("answer", ""))
                if isinstance(opts, list) and len(opts) >= 4 and q_text and gold:
                    valid_questions.append(q)
            valid_doc_flags[doc_idx] = len(valid_questions) > 0

            passage_len = len(self.tokenizer.encode(article)) if hasattr(self.tokenizer, "encode") else len(article.split())

            if self.use_baseline_reward:
                b_reward, b_correct, b_total = self._score_document_baseline(article, valid_questions)
            else:
                b_reward, b_correct, b_total = 0.0, 0, 0

            for sample_idx, sample in enumerate(understanding_data[doc_idx]):
                reward, n_correct, n_total = self._score_one_understanding(
                    article=article,
                    understandings_text=sample["response"],
                    questions=valid_questions,
                )
                
                if self.use_baseline_reward:
                    if b_correct > n_correct + self.baseline_forgiven:
                        reward = -1.0
                    else:
                        reward = (reward - b_reward) * self.scale_reward
                else:
                    reward = reward * self.scale_reward
                
                extracted_understanding = self._extract_understanding_from_tags(sample["response"])
                if hasattr(self.tokenizer, "encode"):
                    understanding_len = len(self.tokenizer.encode(extracted_understanding))
                else:
                    understanding_len = len(extracted_understanding.split())

                # Apply conciseness penalty only if it exceeds 3/4 of the passage length
                if understanding_len > 0.75 * passage_len:
                    penalty = (understanding_len / max(passage_len, 1)) * self.conciseness_penalty_k
                    reward -= penalty
                
                understanding_rewards[(doc_idx, sample_idx)] = reward
                total_eval_correct += n_correct
                total_eval_q += n_total

        all_imp_rewards = list(understanding_rewards.values())
        info = {
            "actor/num_documents": float(len(prompts)),
            "actor/understanding_samples": float(self.reasoning_num_samples),
            "actor/understanding_reward_mean": float(np.mean(all_imp_rewards)) if all_imp_rewards else 0.0,
            "actor/understanding_reward_std": float(np.std(all_imp_rewards)) if all_imp_rewards else 0.0,
            "actor/eval_question_accuracy": float(total_eval_correct / max(total_eval_q, 1)),
            "actor/step_time": float(time.time() - t0),
        }

        trajectory_data: List[TransitionData] = []
        for doc_idx in range(len(prompts)):
            doc_has_valid_q = valid_doc_flags.get(doc_idx, False)
            for sample_idx, sample in enumerate(understanding_data[doc_idx]):
                reward = understanding_rewards.get((doc_idx, sample_idx), 0.0)
                loss_mask = doc_has_valid_q
                if self.args.ignore_no_eos and sample["is_truncated"]:
                    loss_mask = False

                trajectory_data.append(
                    TransitionData(
                        prompt=sample["prompt"],
                        prompt_ids=sample["prompt_ids"],
                        response=sample["response"],
                        response_ids=sample["response_ids"],
                        response_logprobs=sample["response_logprobs"],
                        rewards=self._terminal_reward(len(sample["response_ids"]), reward),
                        loss_mask=loss_mask,
                        info=info,
                    )
                )

        expected = len(prompts) * self.reasoning_num_samples
        assert len(trajectory_data) == expected
        logging.info(
            "Understanding-only actor done: docs=%d transitions=%d per_doc=%d",
            len(prompts),
            len(trajectory_data),
            self.reasoning_num_samples,
        )
        return self.ipc_client.serialize_ipc(trajectory_data)


class UnderstandingOnlyLearner(PPOLearner):
    def _init(self, args: UnderstandingOnlyArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.args = args
        self.args.max_queries = np.inf
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )
        if args.critic_type in ["grpo", "ppo"] and args.remove_len_bias:
            self.masked_aggregator = functools.partial(
                masked_sum,
                constant_normalizer=args.generate_max_length,
            )

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        rewards = rewards.sum(-1)
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if (self.args.critic_type == "grpo") and (not self.args.remove_std_bias):
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

        max_train = min(int(self.args.max_train), len(train_dataset))
        train_dataset = train_dataset.select(range(max_train))

        input_key = self.args.input_key
        if input_key not in train_dataset.column_names:
            input_key = "article"
        output_key = self.args.output_key
        if output_key not in train_dataset.column_names:
            output_key = "questions"

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


def run_understandings_only_oat(args: UnderstandingOnlyArgs):
    args = configure_understandings_only_args(args)
    args = default_args_validation(args)

    program, local_resources = get_program(
        args,
        learner_cls=UnderstandingOnlyLearner,
        actor_cls=UnderstandingOnlyActor,
    )
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    cli_args: UnderstandingOnlyArgs = get_default_args(UnderstandingOnlyArgs)
    run_understandings_only_oat(cli_args)