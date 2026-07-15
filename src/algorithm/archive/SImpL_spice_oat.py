"""SImpL-SPICE (experimental): understanding co-training + frontier CURRICULUM for cot.

Idea (SPICE-like, see the 2026-06 "drawing board" discussion in FINDINGS.md):
the understanding objective keeps failing because its reward is redundant with cot.
Here the understanding/eval machinery instead earns its keep as a CURRICULUM signal.

Per passage, in ONE step:
  1. generate understanding rollouts (N) and score them (difficulty-weighted marginal
     reward, inherited) -- the understanding is still co-trained on shared weights;
  2. as a FREE by-product we already compute the direct passage-only answer accuracy
     per question (the marginal difficulty baseline) -- REUSE it (no new generations);
  3. SELECT the frontier question = the one whose direct accuracy is closest to 0.5
     (maximally informative / hard-but-solvable, SPICE's proposer-at-the-frontier);
  4. train cot ONLY on that frontier question (N rollouts), not on all questions.

So it is ~1:1 (N understanding : N cot per passage) and cheaper than full simpl
(which does N cot rollouts PER question). Run more prompt-epochs to match compute.

Difficulty signal = DIRECT cot (passage-only) accuracy, reusing the baseline pass
(decoupled from understanding quality -> a clean 'frontier of the policy's ability').
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from oat.actors.base import ActorBase
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TransitionData
from oat.utils.data import PromptDataset

from src.utils.oat_prompt_templates import (
    understanding_prompt,
    maybe_apply_chat_template,
    qa_cot_prompt,
    qa_eval_understanding_only_prompt,
)
from src.utils.parsing_utils import extract_boxed_letter, parse_questions, normalize_gold_letter
from src.utils.fail_fast import fail_fast

from src.algorithm.SImpL_marginal_oat import SImpLMarginalArgs, SImpLMarginalActor
from src.algorithm.SImpL_oat import SImpLLearner, configure_simpl_args, collate_prompt_batch


@dataclass
class SImpLSpiceArgs(SImpLMarginalArgs):
    # Target accuracy for frontier-question selection (the question whose DIRECT
    # passage-only accuracy is closest to this gets the cot training rollout).
    frontier_target: float = 0.5
    # How many frontier q/a pairs (closest to target) to train cot on per passage.
    # 1 = single frontier question (default). K>1 trains cot on the K nearest-to-0.5
    # questions -> each is its own GRPO group of N samples.
    num_frontier_questions: int = 1
    # Optional anneal of K over training: from num_frontier_questions down/up to
    # frontier_anneal_to, linearly over frontier_anneal_steps actor-step calls.
    # Rationale (from-scratch): early the model is at the floor so the 0.5-frontier is
    # meaningless -> select MORE questions to bootstrap, then NARROW as it improves.
    # frontier_anneal_steps<=0 disables annealing (K stays = num_frontier_questions).
    frontier_anneal_to: int = 1
    frontier_anneal_steps: int = 0
    # Which K questions to train cot on:
    #   "frontier" (closest to target, the method);
    #   "rotate"   (no-curriculum control: K=1, deterministic round-robin through a
    #               per-passage shuffled question order -> question[i] on the i-th
    #               time the passage is seen, so all questions get equal coverage
    #               across epochs at the SAME step-rate as frontier);
    #   "random"   (K=1, a fresh uniform draw each step -> all questions, no
    #               curriculum, but coverage is only uniform in expectation).
    selection_mode: str = "frontier"
    # Co-train the understanding objective. False = pure cot on the selected K
    # questions per passage (the "restructured cot" baseline, same step-rate as SPICE).
    train_understanding: bool = True
    # Pair-dataset mode (data-efficiency framing): emit ONE row per (passage, single
    # question) -- the same flattened structure as the CoT dataset -- instead of one
    # row per passage. Each row then carries exactly 1 question, so per step we do
    # understanding(passage)+cot(question) for the understanding arm, or pure cot for
    # the cot arm. Question coverage is handled by the dataloader (every pair is a
    # row), not the in-actor rotate counter. Cap questions/passage at the value below.
    pair_dataset: bool = False
    max_questions_per_passage: int = 8
    # Pair-dataset variant: keep cot trained on the single paired (sel) question, but
    # score the UNDERSTANDING on ALL of the passage's questions (difficulty-weighted
    # marginal over all), not just the paired one. The row then carries all questions
    # plus a `sel_index` pointing at the one cot is trained on. Only affects the
    # understanding arm (train_understanding=True); cot-only arms are unchanged.
    understand_all_questions: bool = False


def configure_simpl_spice_args(args: SImpLSpiceArgs) -> SImpLSpiceArgs:
    args = configure_simpl_args(args)
    # Each passage emits (K cot + maybe 1 understanding) groups of N, so the buffer
    # must hold (K + [understanding?])x the usual per-device rollout.
    k = max(1, int(getattr(args, "num_frontier_questions", 1)))
    u = 1 if bool(getattr(args, "train_understanding", True)) else 0
    need = int(args.rollout_batch_size_per_device) * int(args.num_samples) * (k + u)
    if args.pi_buffer_maxlen_per_device < need:
        args.pi_buffer_maxlen_per_device = need

    # selection_mode=="rotate" keeps its per-passage round-robin counter in the
    # actor PROCESS, so it is only authoritative when there is exactly ONE actor.
    # Replicate oat's launch-time actor count (interface.py) and fail fast if >1,
    # otherwise the no-curriculum coverage would silently degrade to ~"random".
    if str(getattr(args, "selection_mode", "frontier")).lower() == "rotate":
        actor_gpus = int(args.gpus) if args.collocate else (
            args.gpus // 2 if args.gpus % 2 == 0 else args.gpus // 2 + 1)
        total_actors = (actor_gpus // int(args.num_gpus_per_actor)) * int(getattr(args, "num_groups", 1))
        assert total_actors == 1, (
            f"selection_mode='rotate' requires a single actor process for its per-passage "
            f"round-robin to be authoritative, but this config implies {total_actors} actors "
            f"(gpus={args.gpus}, num_gpus_per_actor={args.num_gpus_per_actor}, "
            f"num_groups={getattr(args, 'num_groups', 1)}, collocate={args.collocate}). "
            f"Use 1 actor, or switch to selection_mode='random'."
        )
    return args


class SImpLSpiceActor(SImpLMarginalActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.frontier_target = float(getattr(self.args, "frontier_target", 0.5))
        self.num_frontier_questions = max(1, int(getattr(self.args, "num_frontier_questions", 1)))
        self.frontier_anneal_to = max(1, int(getattr(self.args, "frontier_anneal_to", 1)))
        self.frontier_anneal_steps = int(getattr(self.args, "frontier_anneal_steps", 0))
        self.selection_mode = str(getattr(self.args, "selection_mode", "frontier")).lower()
        self.train_understanding = bool(getattr(self.args, "train_understanding", True))
        self._step_calls = 0
        # Per-passage round-robin position for selection_mode=="rotate" (single
        # collocated actor -> this state is authoritative for the whole run).
        self._rotate_pos: Dict[str, int] = {}

    def _rotate_pick(self, passage: str, num_q: int) -> int:
        # Deterministic per-passage shuffled order (seeded by passage hash so it is
        # stable across the run), advanced one question each time the passage recurs.
        order = list(range(num_q))
        random.Random(hash(passage) & 0xFFFFFFFF).shuffle(order)
        pos = self._rotate_pos.get(passage, 0)
        self._rotate_pos[passage] = pos + 1
        return order[pos % num_q]

    def _current_k(self) -> int:
        # Linearly anneal K from num_frontier_questions -> frontier_anneal_to over
        # frontier_anneal_steps actor-step calls (then hold at the end value).
        if self.frontier_anneal_steps <= 0:
            return self.num_frontier_questions
        frac = min(1.0, self._step_calls / float(self.frontier_anneal_steps))
        k = round(self.num_frontier_questions + frac * (self.frontier_anneal_to - self.num_frontier_questions))
        return max(1, int(k))

    def _gen_reasoning(self, prompts):
        outs = self.generate(
            prompts,
            self._build_sampling_params(
                temperature=self.sampling_params.temperature,
                max_tokens=self.reasoning_max_tokens,
                with_logprobs=True,
            ),
        )
        data = []
        for o in outs:
            pid, text, tok, lp_, trunc = self._extract_output(o)
            data.append({"prompt_ids": pid, "text": text, "token_ids": tok, "logprobs": lp_, "is_truncated": trunc})
        return data

    @fail_fast("SImpLSpiceActor.step")
    def step(self, prompts, formatted_prompts, references=None) -> List[TransitionData]:
        del formatted_prompts
        assert not self.eval_mode
        t0 = time.time()
        if not prompts:
            return self.ipc_client.serialize_ipc([])
        if references is None:
            references = [None] * len(prompts)

        doc_qs: List[List[Dict]] = []
        doc_sel: List = []  # per-doc forced cot question index (pair+understand_all), else None
        for ref in references:
            try:
                parsed = json.loads(ref) if ref else {"questions": []}
            except json.JSONDecodeError:
                parsed = {"questions": []}
            doc_qs.append(parsed.get("questions", []))
            doc_sel.append(parsed.get("sel_index", None))

        # ---- Phase A: understanding rollouts (N per passage) ----
        # Skipped entirely when train_understanding is False (pure-cot baseline).
        u_prompts: List[str] = []
        u_doc: List[int] = []
        if self.train_understanding:
            for d, qs in enumerate(doc_qs):
                if not qs:
                    continue
                pt = understanding_prompt(prompts[d], self.dataset_name, tagged=not self.use_full_understanding_output, answer_ready=self.qa_direct_answer)
                for _ in range(self.reasoning_num_samples):
                    u_prompts.append(maybe_apply_chat_template(self.tokenizer, pt, self.is_instruct))
                    u_doc.append(d)
        u_data = self._gen_reasoning(u_prompts) if u_prompts else []

        # ---- Phase B: understanding-conditioned QA (marginal reward signal) ----
        eval_prompts: List[str] = []
        eval_meta: List[Dict] = []
        for i, d in enumerate(u_doc):
            understanding = self._extract_understanding_from_tags(u_data[i]["text"])
            if not understanding:
                continue
            article = prompts[d] if self.use_understanding_passage else ""
            for qi, q in enumerate(doc_qs[d]):
                eval_prompts.append(maybe_apply_chat_template(
                    self.tokenizer,
                    qa_eval_understanding_only_prompt(article, understanding, q.get("question", ""), q.get("options", []), self.dataset_name, direct=self.qa_direct_answer),
                    self.is_instruct,
                ))
                eval_meta.append({"u_idx": i, "q_index": qi, "gold": q.get("answer", ""), "num_opt": len(q.get("options", []))})
        eval_by_u_q: Dict[int, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        if eval_prompts:
            eo = self.generate(eval_prompts, self._build_sampling_params(
                temperature=self.sampling_params.temperature if self.qa_num_samples > 1 else 0.0,
                max_tokens=self.qa_eval_max_tokens, with_logprobs=False, n=self.qa_num_samples))
            for out, m in zip(eo, eval_meta):
                for comp in out.outputs:
                    pred = extract_boxed_letter(comp.text or "", m["num_opt"])
                    eval_by_u_q[m["u_idx"]][m["q_index"]].append({
                        "is_correct": int(pred == m["gold"]),
                        "num_toks": len(comp.token_ids) if comp.token_ids else 0,
                    })

        # ---- Phase B2: DIRECT passage-only baseline per question ----
        # (reused for BOTH the difficulty weight AND frontier selection; no new gen)
        # Only needed for frontier selection or difficulty-weighted understanding.
        need_baseline = (self.selection_mode == "frontier") or (self.train_understanding and self.difficulty_weighting)
        base_acc: Dict[tuple, float] = {}
        bprompts: List[str] = []
        bmeta: List[Dict] = []
        if need_baseline:
            for d, qs in enumerate(doc_qs):
                for qi, q in enumerate(qs):
                    article = prompts[d] if self.baseline_with_passage else ""
                    bprompts.append(maybe_apply_chat_template(
                        self.tokenizer,
                        qa_eval_understanding_only_prompt(article, "", q.get("question", ""), q.get("options", []), self.dataset_name, direct=self.qa_direct_answer),
                        self.is_instruct,
                    ))
                    bmeta.append({"doc": d, "qi": qi, "gold": q.get("answer", ""), "num_opt": len(q.get("options", []))})
        if bprompts:
            bo = self.generate(bprompts, self._build_sampling_params(
                temperature=self.sampling_params.temperature if self.baseline_num_samples > 1 else 0.0,
                max_tokens=self.qa_eval_max_tokens, with_logprobs=False, n=self.baseline_num_samples))
            for out, m in zip(bo, bmeta):
                cor = tot = 0
                for comp in out.outputs:
                    cor += int(extract_boxed_letter(comp.text or "", m["num_opt"]) == m["gold"])
                    tot += 1
                base_acc[(m["doc"], m["qi"])] = cor / max(tot, 1)

        # ---- Phase Select: K questions per passage (frontier = closest to target;
        #      random = uniformly sampled control with matched step-rate) ----
        self._step_calls += 1
        k_now = self._current_k()
        frontier: Dict[int, List[int]] = {}
        for d, qs in enumerate(doc_qs):
            if not qs:
                continue
            if doc_sel[d] is not None:
                # pair + understand_all: cot is fixed to the paired (sel) question,
                # while understanding above was scored on ALL of this passage's qs.
                frontier[d] = [int(doc_sel[d])]
            elif self.selection_mode == "rotate":
                # No-curriculum control: exactly 1 question per passage per step,
                # round-robin through all questions across epochs (k_now ignored).
                frontier[d] = [self._rotate_pick(prompts[d], len(qs))]
            elif self.selection_mode == "random":
                idxs = list(range(len(qs)))
                random.shuffle(idxs)
                frontier[d] = idxs[:k_now]
            else:
                order = sorted(range(len(qs)), key=lambda qi: abs(base_acc.get((d, qi), 0.0) - self.frontier_target))
                frontier[d] = order[:k_now]

        # ---- Phase A2: cot rollouts on each frontier question (N per question) ----
        c_prompts: List[str] = []
        c_doc: List[int] = []
        c_gold: List[str] = []
        c_nopt: List[int] = []
        for d, qs in enumerate(doc_qs):
            if not qs:
                continue
            for qi in frontier[d]:
                q = qs[qi]
                pt = qa_cot_prompt(prompts[d], q["question"], q["options"], self.dataset_name)
                for _ in range(self.reasoning_num_samples):
                    c_prompts.append(maybe_apply_chat_template(self.tokenizer, pt, self.is_instruct))
                    c_doc.append(d)
                    c_gold.append(q.get("answer", ""))
                    c_nopt.append(len(q.get("options", [])))
        c_data = self._gen_reasoning(c_prompts) if c_prompts else []

        # ---- understanding rewards (difficulty-weighted marginal) ----
        u_reward = [0.0] * len(u_doc)
        und_rewards = []
        for i, d in enumerate(u_doc):
            per_q = eval_by_u_q.get(i, {})
            if not per_q:
                u_reward[i] = -0.1 * self.scale_reward
                und_rewards.append(u_reward[i])
                continue
            num = den = 0.0
            correct_toks = []
            for qi, recs in per_q.items():
                qacc = sum(x["is_correct"] for x in recs) / max(len(recs), 1)
                w = max(self.min_question_weight, 1.0 - base_acc.get((d, qi), 0.0)) if self.difficulty_weighting else 1.0
                num += w * qacc
                den += w
                correct_toks.extend(x["num_toks"] for x in recs if x["is_correct"])
            wacc = (num / den) if den > 1e-8 else 0.0
            br = self.incorrect_reward + (self.correct_reward - self.incorrect_reward) * wacc
            if correct_toks:
                br += sum(self.conciseness_penalty_k * (1.0 - nt / self.qa_eval_max_tokens) for nt in correct_toks) / len(correct_toks)
            u_reward[i] = br * self.scale_reward * self.understanding_reward_scale
            und_rewards.append(u_reward[i])

        # ---- cot rewards (correctness on the frontier question) ----
        c_reward = [0.0] * len(c_doc)
        cot_correct = cot_valid = 0
        for j in range(len(c_doc)):
            pred = extract_boxed_letter(c_data[j]["text"], c_nopt[j])
            if pred == c_gold[j]:
                r = self.correct_reward
            elif pred in [chr(ord("A") + k) for k in range(c_nopt[j])]:
                r = 0.1
            else:
                r = self.incorrect_reward
            c_reward[j] = r * self.scale_reward
            if not c_data[j]["is_truncated"]:
                cot_valid += 1
                cot_correct += int(pred == c_gold[j])

        info = {
            "actor/num_documents": float(len(prompts)),
            "actor/num_samples": float(self.reasoning_num_samples),
            "actor/cot_accuracy": float(cot_correct / max(cot_valid, 1)),
            "actor/cot_reward_mean": float(np.mean(c_reward)) if c_reward else 0.0,
            "actor/understanding_reward_mean": float(np.mean(und_rewards)) if und_rewards else 0.0,
            "actor/baseline_direct_accuracy": float(np.mean(list(base_acc.values()))) if base_acc else 0.0,
            "actor/frontier_baseline_acc": float(np.mean([base_acc.get((d, qi), 0.0) for d in frontier for qi in frontier[d]])) if frontier else 0.0,
            "actor/frontier_k": float(k_now),
            "actor/step_time": float(time.time() - t0),
        }

        # ---- assemble: per passage, N understanding then N cot (contiguous groups) ----
        u_by_doc: Dict[int, List[int]] = defaultdict(list)
        for i, d in enumerate(u_doc):
            u_by_doc[d].append(i)
        c_by_doc: Dict[int, List[int]] = defaultdict(list)
        for j, d in enumerate(c_doc):
            c_by_doc[d].append(j)

        traj: List[TransitionData] = []
        def _emit(prompt, ext, reward):
            traj.append(TransitionData(
                prompt=prompt, prompt_ids=ext["prompt_ids"], response=ext["text"],
                response_ids=ext["token_ids"], response_logprobs=ext["logprobs"],
                rewards=self._terminal_reward(len(ext["token_ids"]), reward),
                loss_mask=not (self.args.ignore_no_eos and ext["is_truncated"]),
                info=info,
            ))
        for d in sorted(set(u_doc) | set(c_doc)):
            for i in u_by_doc.get(d, []):
                _emit(u_prompts[i], u_data[i], u_reward[i])
            for j in c_by_doc.get(d, []):
                _emit(c_prompts[j], c_data[j], c_reward[j])

        logging.info("SImpL-spice actor: docs=%d transitions=%d (u=%d c=%d) frontier_acc=%.3f",
                     len(prompts), len(traj), len(u_doc), len(c_doc), info["actor/frontier_baseline_acc"])
        return self.ipc_client.serialize_ipc(traj)


class SImpLSpiceLearner(SImpLLearner):
    def prepare_data(self, strategy, tokenizer):
        data_obj = self._load_prompt_data()
        if hasattr(data_obj, "keys") and self.args.train_split in data_obj:
            train_dataset = data_obj[self.args.train_split]
        else:
            train_dataset = data_obj
        input_key = self.args.input_key if self.args.input_key in train_dataset.column_names else "article"
        output_key = self.args.output_key if self.args.output_key in train_dataset.column_names else "questions"

        pair_mode = bool(getattr(self.args, "pair_dataset", False))
        max_q = max(1, int(getattr(self.args, "max_questions_per_passage", 8)))
        understand_all = bool(getattr(self.args, "understand_all_questions", False))

        def to_spice_rows(batch):
            arts, qss = [], []
            for article, qs_json in zip(batch[input_key], batch[output_key]):
                valid = []
                for q in parse_questions(qs_json):
                    opts = q.get("options", [])
                    gold = normalize_gold_letter(q.get("answer", ""), len(opts) if isinstance(opts, list) else 4)
                    if isinstance(opts, list) and len(opts) >= 2 and q.get("question", "") and gold:
                        valid.append({"question": q["question"], "options": opts, "answer": gold})
                if not valid:
                    continue
                valid = valid[:max_q]
                if pair_mode and understand_all:
                    # One row per (passage, sel question): carry ALL questions (so the
                    # understanding is scored on all of them) + sel_index for cot.
                    for j in range(len(valid)):
                        arts.append(article)
                        qss.append(json.dumps({"task_type": "spice", "questions": valid, "sel_index": j}))
                elif pair_mode:
                    # One row per (passage, single question) -- flattened like CoT.
                    for q in valid:
                        arts.append(article)
                        qss.append(json.dumps({"task_type": "spice", "questions": [q]}))
                else:
                    arts.append(article)
                    qss.append(json.dumps({"task_type": "spice", "questions": valid}))
            return {input_key: arts, output_key: qss}

        train_dataset = train_dataset.map(to_spice_rows, batched=True, remove_columns=train_dataset.column_names)
        max_train = min(int(self.args.max_train), len(train_dataset))
        train_dataset = train_dataset.select(range(max_train)).select_columns([input_key, output_key])

        self.prompts_dataset = PromptDataset(
            train_dataset, tokenizer, strategy,
            input_key=input_key, output_key=output_key,
            apply_chat_template=False, get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, strategy.args.rollout_batch_size_per_device,
            pin_memory=True, shuffle=True, collate_fn=collate_prompt_batch,
        )
        self.eval_prompts_dataset = None
        self.eval_prompts_dataloader = None


def run_simpl_spice_oat(args: SImpLSpiceArgs):
    args = configure_simpl_spice_args(args)
    args = default_args_validation(args)
    program, local_resources = get_program(args, learner_cls=SImpLSpiceLearner, actor_cls=SImpLSpiceActor)
    lp.launch(program, launch_type=args.launch_type, local_resources=local_resources, terminal="current_terminal")


if __name__ == "__main__":
    cli_args: SImpLSpiceArgs = get_default_args(SImpLSpiceArgs)
    run_simpl_spice_oat(cli_args)
