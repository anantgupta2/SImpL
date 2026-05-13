
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

from src.utils.oat_prompt_templates import (
    implication_prompt,
    maybe_apply_chat_template,
    qa_eval_prompt,
    qa_train_prompt,
)


def extract_boxed_letter(text: str) -> str:
    # Take the last boxed answer if multiple are present.
    matches = re.findall(r"\\boxed\s*\{\s*([A-Da-d])\s*\}", text or "")
    if matches:
        return matches[-1].upper()

    # Fallback: plain standalone letter near the end.
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
    """Keep ragged references as raw Python objects during batching."""
    processed_prompts = [item[0] for item in batch]
    raw_prompts = [item[1] for item in batch]
    references = [item[2] for item in batch]
    return processed_prompts, raw_prompts, references


@dataclass
class SImpLArgs(PPOArgs):
    prompt_data: str = ""
    input_key: str = "article"
    output_key: str = "questions"
    train_split: str = "train"
    max_train: int = 999999

    reasoning_num_samples: int = 4
    reasoning_max_tokens: int = 384
    reasoning_max_items: int = 8

    qa_num_samples: int = 4
    qa_train_max_tokens: int = 512
    qa_eval_max_tokens: int = 256

    incorrect_reward: float = 0.0
    correct_reward: float = 1.0

    critic_type: Literal["ppo", "grpo", "drgrpo"] = field(default="drgrpo")
    remove_len_bias: bool = False
    remove_std_bias: bool = False

    # Keep eval scheduler disabled, but keep online_evaluation enabled so OAT forwards references.
    eval_steps: int = -1
    online_evaluation: bool = True
    apply_chat_template: bool = False
    is_instruct: bool = False
    beta: float = 0.0


def configure_simpl_args(args: SImpLArgs) -> SImpLArgs:
    if int(args.reasoning_num_samples) < 1:
        raise ValueError("reasoning_num_samples must be >= 1")
    if int(args.qa_num_samples) < 1:
        raise ValueError("qa_num_samples must be >= 1")

    # OAT validates `algo` against RLAlgo enum, which only includes PPO.
    # Keep DR-GRPO behavior behind `critic_type` while using PPO training loop.
    args.algo = "PPO"
    args.oracle = ""
    args.oracle_type = "reward"
    args.is_instruct = bool(args.is_instruct)
    # Keep raw passages in the dataset path; actor-level generation applies template.
    args.apply_chat_template = False
    args.online_evaluation = True
    args.eval_steps = -1
    args.beta = 0.0

    args.num_samples = int(args.reasoning_num_samples) + int(args.qa_num_samples)
    minimum_buffer = int(args.rollout_batch_size_per_device) * int(args.num_samples)
    if args.pi_buffer_maxlen_per_device < minimum_buffer:
        args.pi_buffer_maxlen_per_device = minimum_buffer
    return args


class SImpLActor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.reasoning_num_samples = int(self.args.reasoning_num_samples)
        self.qa_num_samples = int(self.args.qa_num_samples)
        self.reasoning_max_tokens = int(self.args.reasoning_max_tokens)
        self.qa_eval_max_tokens = int(self.args.qa_eval_max_tokens)
        self.qa_train_max_tokens = int(self.args.qa_train_max_tokens)
        self.reasoning_max_items = int(self.args.reasoning_max_items)

        self.correct_reward = float(self.args.correct_reward)
        self.incorrect_reward = float(self.args.incorrect_reward)
        self.is_instruct = bool(getattr(self.args, "is_instruct", False))

        # Custom step logic controls stopping behavior.
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

    def _extract_text_only(self, output) -> str:
        completion = output.outputs[0]
        return completion.text or ""

    def _score_one_implication(
        self,
        article: str,
        implications_text: str,
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
                    qa_eval_prompt(article, implications_text, q_text, opts),
                    self.is_instruct,
                )
            )
            golds.append(gold)

        if not eval_prompts:
            return 0.0, 0, 0

        eval_outputs = self.generate(
            eval_prompts,
            self._build_sampling_params(
                temperature=0.0,
                max_tokens=self.qa_eval_max_tokens,
                with_logprobs=False,
            ),
        )

        correct = 0
        for out, gold in zip(eval_outputs, golds):
            pred = extract_boxed_letter(self._extract_text_only(out))
            correct += int(pred == gold)

        total = len(golds)
        if total == 0:
            return 0.0, 0, 0

        acc = float(correct) / float(total)
        reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * acc
        return reward, correct, total

    def _train_question_for_doc(self, questions: List[Dict]) -> Optional[Dict]:
        for q in questions:
            q_text = q.get("question", "")
            opts = q.get("options", [])
            gold = normalize_gold_letter(q.get("answer", ""))
            if isinstance(opts, list) and len(opts) >= 4 and q_text and gold:
                return {"question": q_text, "options": opts, "answer": gold}
        return None

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

        # 1) Implication rollouts (policy trajectories that receive GRPO rewards).
        implication_prompts = []
        implication_owner = []
        for doc_idx, article in enumerate(prompts):
            p = implication_prompt(article, self.reasoning_max_items)
            for _ in range(self.reasoning_num_samples):
                implication_prompts.append(
                    maybe_apply_chat_template(
                        self.tokenizer,
                        p,
                        self.is_instruct,
                    )
                )
                implication_owner.append(doc_idx)

        imp_outputs = self.generate(
            implication_prompts,
            self._build_sampling_params(
                temperature=self.sampling_params.temperature,
                max_tokens=self.reasoning_max_tokens,
                with_logprobs=True,
            ),
        )

        implication_data: List[List[Dict]] = [[] for _ in range(len(prompts))]
        for out, owner, prompt_text in zip(imp_outputs, implication_owner, implication_prompts):
            prompt_ids, text, token_ids, logprobs, is_truncated = self._extract_output(out)
            implication_data[owner].append(
                {
                    "prompt": prompt_text,
                    "prompt_ids": prompt_ids,
                    "response": text,
                    "response_ids": token_ids,
                    "response_logprobs": logprobs,
                    "is_truncated": is_truncated,
                }
            )

        # 2) Use each implication set to answer all questions and compute reward.
        implication_rewards: Dict[Tuple[int, int], float] = {}
        best_idx_per_doc: Dict[int, int] = {}
        total_eval_q = 0
        total_eval_correct = 0

        for doc_idx, article in enumerate(prompts):
            doc_questions = parse_questions(references[doc_idx])
            best_reward = -1e9
            best_idx = 0

            for sample_idx, sample in enumerate(implication_data[doc_idx]):
                reward, n_correct, n_total = self._score_one_implication(
                    article=article,
                    implications_text=sample["response"],
                    questions=doc_questions,
                )
                implication_rewards[(doc_idx, sample_idx)] = reward
                total_eval_correct += n_correct
                total_eval_q += n_total

                if reward > best_reward:
                    best_reward = reward
                    best_idx = sample_idx

            best_idx_per_doc[doc_idx] = best_idx

        # 3) QA optimization: train on one valid question per document using best implications.
        qa_prompts = []
        qa_meta = []
        for doc_idx, article in enumerate(prompts):
            doc_questions = parse_questions(references[doc_idx])
            train_q = self._train_question_for_doc(doc_questions)
            best_sample = implication_data[doc_idx][best_idx_per_doc[doc_idx]]
            best_implications = best_sample["response"]

            if train_q is None:
                # Keep shape fixed with placeholder prompts and zero-reward masking.
                for _ in range(self.qa_num_samples):
                    qa_prompts.append(
                        maybe_apply_chat_template(
                            self.tokenizer,
                            "Question: N/A\nAnswer with \\boxed{A}.",
                            self.is_instruct,
                        )
                    )
                    qa_meta.append((doc_idx, "", [], "", True))
                continue

            q_prompt = qa_train_prompt(
                article=article,
                best_implications=best_implications,
                question_text=train_q["question"],
                options=train_q["options"],
            )
            for _ in range(self.qa_num_samples):
                qa_prompts.append(
                    maybe_apply_chat_template(
                        self.tokenizer,
                        q_prompt,
                        self.is_instruct,
                    )
                )
                qa_meta.append((doc_idx, train_q["question"], train_q["options"], train_q["answer"], False))

        qa_outputs = self.generate(
            qa_prompts,
            self._build_sampling_params(
                temperature=self.sampling_params.temperature,
                max_tokens=self.qa_train_max_tokens,
                with_logprobs=True,
            ),
        )

        qa_records: List[List[Dict]] = [[] for _ in range(len(prompts))]
        for out, meta, prompt_text in zip(qa_outputs, qa_meta, qa_prompts):
            doc_idx, _, _, gold, is_placeholder = meta
            prompt_ids, text, token_ids, logprobs, is_truncated = self._extract_output(out)
            if is_placeholder:
                reward = 0.0
                loss_mask = False
            else:
                pred = extract_boxed_letter(text)
                reward = self.correct_reward if pred == gold else self.incorrect_reward
                loss_mask = True

            qa_records[doc_idx].append(
                {
                    "prompt": prompt_text,
                    "prompt_ids": prompt_ids,
                    "response": text,
                    "response_ids": token_ids,
                    "response_logprobs": logprobs,
                    "is_truncated": is_truncated,
                    "reward": reward,
                    "loss_mask": loss_mask,
                }
            )

        # 4) Build transitions with fixed per-document count.
        all_imp_rewards = list(implication_rewards.values())
        info = {
            "actor/num_documents": float(len(prompts)),
            "actor/implication_samples": float(self.reasoning_num_samples),
            "actor/qa_samples": float(self.qa_num_samples),
            "actor/implication_reward_mean": float(np.mean(all_imp_rewards)) if all_imp_rewards else 0.0,
            "actor/implication_reward_std": float(np.std(all_imp_rewards)) if all_imp_rewards else 0.0,
            "actor/eval_question_accuracy": float(total_eval_correct / max(total_eval_q, 1)),
            "actor/step_time": float(time.time() - t0),
        }

        trajectory_data: List[TransitionData] = []
        for doc_idx in range(len(prompts)):
            for sample_idx, sample in enumerate(implication_data[doc_idx]):
                reward = implication_rewards.get((doc_idx, sample_idx), 0.0)
                loss_mask = True
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

            for sample in qa_records[doc_idx]:
                loss_mask = sample["loss_mask"]
                if self.args.ignore_no_eos and sample["is_truncated"]:
                    loss_mask = False
                trajectory_data.append(
                    TransitionData(
                        prompt=sample["prompt"],
                        prompt_ids=sample["prompt_ids"],
                        response=sample["response"],
                        response_ids=sample["response_ids"],
                        response_logprobs=sample["response_logprobs"],
                        rewards=self._terminal_reward(
                            len(sample["response_ids"]),
                            float(sample["reward"]),
                        ),
                        loss_mask=loss_mask,
                        info=info,
                    )
                )

        expected_per_doc = self.reasoning_num_samples + self.qa_num_samples
        assert len(trajectory_data) == len(prompts) * expected_per_doc
        logging.info(
            "SImpL actor done: docs=%d transitions=%d per_doc=%d",
            len(prompts),
            len(trajectory_data),
            expected_per_doc,
        )
        return self.ipc_client.serialize_ipc(trajectory_data)


class SImpLLearner(PPOLearner):
    def _init(self, args: SImpLArgs, actors: List[ActorBase]) -> None:
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


def run_simpl_oat(args: SImpLArgs):
    args = configure_simpl_args(args)
    args = default_args_validation(args)

    program, local_resources = get_program(
        args,
        learner_cls=SImpLLearner,
        actor_cls=SImpLActor,
    )
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    cli_args: SImpLArgs = get_default_args(SImpLArgs)
    run_simpl_oat(cli_args)