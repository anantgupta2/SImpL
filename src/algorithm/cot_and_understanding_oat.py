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

from src.utils.oat_prompt_templates import understanding_prompt, maybe_apply_chat_template, qa_cot_prompt, qa_eval_understanding_only_prompt
from src.utils.parsing_utils import extract_boxed_letter, normalize_gold_letter, parse_questions

from src.utils.parsing_utils import extract_boxed_letter, normalize_gold_letter, parse_questions


def collate_prompt_batch(batch):
    processed_prompts = [item[0] for item in batch]
    raw_prompts = [item[1] for item in batch]
    references = [item[2] for item in batch]
    return processed_prompts, raw_prompts, references


@dataclass
class CoTAndUnderstandingArgs(PPOArgs):
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


def configure_cot_and_understanding_args(args: CoTAndUnderstandingArgs) -> CoTAndUnderstandingArgs:
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


class CoTAndUnderstandingActor(PPOActor):
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
        self.scale_reward = getattr(self.args, "reward_scale", 2.0)

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

    # def _score_document_baseline(
    #     self,
    #     article: str,
    #     questions: List[Dict],
    # ) -> Tuple[float, int, int]:
    #     if not questions:
    #         return 0.0, 0, 0

    #     eval_prompts = []
    #     golds = []
    #     num_options_list = []
    #     for q in questions:
    #         q_text = q.get("question", "")
    #         opts = q.get("options", [])
    #         gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
    #         if not isinstance(opts, list) or len(opts) < 2 or not q_text or not gold:
    #             continue
    #         eval_prompts.append(
    #             maybe_apply_chat_template(
    #                 self.tokenizer,
    #                 qa_cot_prompt(article, q_text, opts),
    #                 self.is_instruct,
    #             )
    #         )
    #         golds.append(gold)
    #         num_options_list.append(len(opts))

    #     if not eval_prompts:
    #         return 0.0, 0, 0

    #     eval_outputs = self.generate(
    #         eval_prompts,
    #         self._build_sampling_params(
    #             temperature=self.qa_eval_temperature if self.qa_num_samples > 1 else 0.0,
    #             max_tokens=self.qa_eval_max_tokens,
    #             with_logprobs=False,
    #             n=self.qa_num_samples,
    #         ),
    #     )

    #     correct = 0
    #     for out, gold, num_opt in zip(eval_outputs, golds, num_options_list):
    #         for completion in out.outputs:
    #             pred = extract_boxed_letter(completion.text or "", num_opt)
    #             correct += int(pred == gold)

    #     total = len(golds) * self.qa_num_samples
    #     if total == 0:
    #         return 0.0, 0, 0

    #     acc = float(correct) / float(total)
    #     reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * acc
    #     return reward, correct, total

    def _extract_understanding_from_tags(self, text: str) -> str:
        match = re.search(r"<understanding>\s*(.*?)\s*</understanding>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        ## no need to end the understanding tag; just extract from the start tag to the end of text
        match2 = re.search(r"<understanding>\s*(.*?)\s*$", text, re.DOTALL | re.IGNORECASE)
        if match2:            
            return match2.group(1).strip()
        return ""

    def _score_one_understanding(
        self,
        article: str,
        understandings_text: str,
        questions: List[Dict],
    ) -> Tuple[float, int, int]:
        if not questions:
            return 0.0, 0, 0
        
        if not self.use_baseline_reward:
            article = ""
        
        extracted_understanding = self._extract_understanding_from_tags(understandings_text)
        if not extracted_understanding:
            return -0.2, 0, 0
        eval_prompts = []
        golds = []
        num_options_list = []
        for q in questions:
            q_text = q.get("question", "")
            opts = q.get("options", [])
            gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
            if not isinstance(opts, list) or len(opts) < 2 or not q_text or not gold:
                continue
            eval_prompts.append(
                maybe_apply_chat_template(
                    self.tokenizer,
                    qa_eval_understanding_only_prompt(article, extracted_understanding, q_text, opts),
                    self.is_instruct,
                )
            )
            golds.append(gold)
            num_options_list.append(len(opts))

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
        bonus_sum = 0.0
        for out, gold, num_opt in zip(eval_outputs, golds, num_options_list):
            for completion in out.outputs:
                pred = extract_boxed_letter(completion.text or "", num_opt)
                is_correct = int(pred == gold)
                correct += is_correct
                if is_correct:
                    num_toks = len(completion.token_ids) if completion.token_ids else 0
                    bonus_sum += self.conciseness_penalty_k * (1.0 - num_toks / self.qa_eval_max_tokens)

        total = len(golds) * self.qa_num_samples
        if total == 0:
            return 0.0, 0, 0

        acc = float(correct) / float(total)
        reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * acc
        
        if correct > 0:
            expected_bonus = bonus_sum / correct
            reward += expected_bonus

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

        generation_prompts = []
        generation_meta = []
        
        for doc_idx, (article, ref_str) in enumerate(zip(prompts, references)):
            try:
                parsed = json.loads(ref_str) if ref_str else {"task_type": "cot", "questions": []}
            except json.JSONDecodeError:
                parsed = {"task_type": "cot", "questions": []}
            
            task_type = parsed.get("task_type", "cot")
            valid_questions = parsed.get("questions", [])
            
            if task_type == "cot" and valid_questions:
                q = valid_questions[0]
                prompt_text = qa_cot_prompt(article, q["question"], q["options"])
                for sample_idx in range(self.reasoning_num_samples):
                    prompt_str = maybe_apply_chat_template(self.tokenizer, prompt_text, self.is_instruct)
                    generation_prompts.append(prompt_str)
                    generation_meta.append({
                        "doc_idx": doc_idx,
                        "task_type": "cot",
                        "sample_idx": sample_idx,
                        "gold": q.get("answer", ""),
                        "num_opt": len(q.get("options", [])),
                        "valid_questions": valid_questions
                    })
            else:
                prompt_text = understanding_prompt(article)
                for sample_idx in range(self.reasoning_num_samples):
                    prompt_str = maybe_apply_chat_template(self.tokenizer, prompt_text, self.is_instruct)
                    generation_prompts.append(prompt_str)
                    generation_meta.append({
                        "doc_idx": doc_idx,
                        "task_type": "understanding",
                        "sample_idx": sample_idx,
                        "valid_questions": valid_questions
                    })

        outputs = self.generate(
            generation_prompts,
            self._build_sampling_params(
                temperature=self.sampling_params.temperature,
                max_tokens=self.reasoning_max_tokens,
                with_logprobs=True,
            ),
        )

        trajectory_data: List[TransitionData] = []
        info_rewards_cot = []
        info_rewards_und = []
        correct_count_cot = 0
        valid_count_cot = 0
        total_eval_q = 0
        total_eval_correct = 0
        
        eval_prompts = []
        eval_meta = []
        
        extracted_data = []
        for i, (out, meta) in enumerate(zip(outputs, generation_meta)):
            prompt_ids, text, token_ids, logprobs, is_truncated = self._extract_output(out)
            extracted_data.append({
                "prompt_ids": prompt_ids,
                "text": text,
                "token_ids": token_ids,
                "logprobs": logprobs,
                "is_truncated": is_truncated
            })
            if meta["task_type"] == "understanding" and meta["valid_questions"]:
                extracted_understanding = self._extract_understanding_from_tags(text)
                if extracted_understanding:
                    for q in meta["valid_questions"]:
                        article_text = prompts[meta["doc_idx"]] if self.use_baseline_reward else ""
                        q_text = q.get("question", "")
                        opts = q.get("options", [])
                        gold = q.get("answer", "")
                        eval_text = qa_eval_understanding_only_prompt(article_text, extracted_understanding, q_text, opts)
                        eval_prompts.append(maybe_apply_chat_template(self.tokenizer, eval_text, self.is_instruct))
                        eval_meta.append({
                            "parent_idx": i,
                            "gold": gold,
                            "num_opt": len(opts)
                        })
        
        eval_results = []
        if eval_prompts:
            eval_outputs = self.generate(
                eval_prompts,
                self._build_sampling_params(
                    temperature=self.qa_eval_temperature if self.qa_num_samples > 1 else 0.0,
                    max_tokens=self.qa_eval_max_tokens,
                    with_logprobs=False,
                    n=self.qa_num_samples,
                ),
            )
            for out, e_meta in zip(eval_outputs, eval_meta):
                for completion in out.outputs:
                    pred = extract_boxed_letter(completion.text or "", e_meta["num_opt"])
                    is_correct = int(pred == e_meta["gold"])
                    num_toks = len(completion.token_ids) if completion.token_ids else 0
                    eval_results.append({
                        "parent_idx": e_meta["parent_idx"],
                        "is_correct": is_correct,
                        "num_toks": num_toks
                    })
                    total_eval_q += 1
                    total_eval_correct += is_correct
                    
        from collections import defaultdict
        eval_grouped = defaultdict(list)
        for res in eval_results:
            eval_grouped[res["parent_idx"]].append(res)
            
        for i, (meta, ext_data, prompt_text) in enumerate(zip(generation_meta, extracted_data, generation_prompts)):
            reward = 0.0
            loss_mask = False
            
            if meta["task_type"] == "cot":
                pred = extract_boxed_letter(ext_data["text"], meta["num_opt"])
                gold = meta["gold"]
                num_opt = meta["num_opt"]
                if pred == gold:
                    reward = self.correct_reward
                elif pred in [chr(ord('A') + k) for k in range(num_opt)]:
                    reward = 0.1
                else:
                    reward = self.incorrect_reward
                loss_mask = len(meta["valid_questions"]) > 0
                if loss_mask and not ext_data["is_truncated"]:
                    valid_count_cot += 1
                    correct_count_cot += int(pred == gold)
                info_rewards_cot.append(reward)
            else:
                results = eval_grouped[i]
                if not results and meta["valid_questions"]:
                    reward = -0.2 * self.scale_reward
                elif results:
                    correct = sum(r["is_correct"] for r in results)
                    total = len(results)
                    acc = float(correct) / float(total)
                    base_reward = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * acc
                    
                    if correct > 0:
                        bonus_sum = sum(self.conciseness_penalty_k * (1.0 - r["num_toks"] / self.qa_eval_max_tokens) for r in results if r["is_correct"])
                        base_reward += bonus_sum / correct
                    reward = base_reward * self.scale_reward
                    
                loss_mask = len(meta["valid_questions"]) > 0
                info_rewards_und.append(reward)
                
            if self.args.ignore_no_eos and ext_data["is_truncated"]:
                loss_mask = False
                
            info = {
                "actor/num_documents": float(len(prompts)),
                "actor/num_samples": float(self.reasoning_num_samples),
                "actor/cot_reward_mean": float(np.mean(info_rewards_cot)) if info_rewards_cot else 0.0,
                "actor/cot_accuracy": float(correct_count_cot / max(valid_count_cot, 1)),
                "actor/understanding_reward_mean": float(np.mean(info_rewards_und)) if info_rewards_und else 0.0,
                "actor/eval_question_accuracy": float(total_eval_correct / max(total_eval_q, 1)),
                "actor/step_time": float(time.time() - t0),
            }

            trajectory_data.append(
                TransitionData(
                    prompt=prompt_text,
                    prompt_ids=ext_data["prompt_ids"],
                    response=ext_data["text"],
                    response_ids=ext_data["token_ids"],
                    response_logprobs=ext_data["logprobs"],
                    rewards=self._terminal_reward(len(ext_data["token_ids"]), reward),
                    loss_mask=loss_mask,
                    info=info,
                )
            )

        expected = len(prompts) * self.reasoning_num_samples
        assert len(trajectory_data) == expected
        logging.info(
            "CoT+Understanding actor done: docs=%d transitions=%d per_doc=%d",
            len(prompts),
            len(trajectory_data),
            self.reasoning_num_samples,
        )
        return self.ipc_client.serialize_ipc(trajectory_data)


class CoTAndUnderstandingLearner(PPOLearner):
    def _init(self, args: CoTAndUnderstandingArgs, actors: List[ActorBase]) -> None:
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

        input_key = self.args.input_key
        if input_key not in train_dataset.column_names:
            input_key = "article"
        output_key = self.args.output_key
        if output_key not in train_dataset.column_names:
            output_key = "questions"

        def flatten_and_mix_questions(batch):
            new_articles = []
            new_questions = [] # We use this as output_key (references)
            for article, qs_json in zip(batch[input_key], batch[output_key]):
                qs = parse_questions(qs_json)
                valid_qs = []
                for q in qs:
                    q_text = q.get("question", "")
                    opts = q.get("options", [])
                    gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
                    if isinstance(opts, list) and len(opts) >= 2 and q_text and gold:
                        valid_qs.append({"question": q_text, "options": opts, "answer": gold})
                
                if not valid_qs:
                    continue
                
                # Add 'understanding' task
                new_articles.append(article)
                new_questions.append(json.dumps({
                    "task_type": "understanding",
                    "questions": valid_qs
                }))
                
                # Add 'cot' task for each question
                for vq in valid_qs:
                    new_articles.append(article)
                    new_questions.append(json.dumps({
                        "task_type": "cot",
                        "questions": [vq]
                    }))
            return {input_key: new_articles, output_key: new_questions}

        train_dataset = train_dataset.map(
            flatten_and_mix_questions,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        train_dataset = train_dataset.shuffle(seed=42)
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


def run_cot_and_understanding_oat(args: CoTAndUnderstandingArgs):
    args = configure_cot_and_understanding_args(args)
    args = default_args_validation(args)

    program, local_resources = get_program(
        args,
        learner_cls=CoTAndUnderstandingLearner,
        actor_cls=CoTAndUnderstandingActor,
    )
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    cli_args: CoTAndUnderstandingArgs = get_default_args(CoTAndUnderstandingArgs)
    run_cot_and_understanding_oat(cli_args)