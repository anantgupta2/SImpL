from __future__ import annotations

from datasets import load_dataset, Dataset
import hashlib
import json
import os
import random
import string
from typing import Any


DATASET_REGISTRY = {
    "race-high": {
        "hf_dataset": "ehovy/race",
        "subset": "high",
        "format": "ehovy_race",
        "num_options": 4,
    },
    "race-c": {
        "hf_dataset": "tasksource/race-c",
        "subset": None,
        "format": "tasksource_race",
        "num_options": 4,
    },
    "lsat-ar": {
        "hf_dataset": "tasksource/lsat-ar",
        "subset": None,
        "format": "tasksource_lsat_ar",
        "num_options": 5,
    },
    # The other two LSAT sections, as near-transfer targets for LSAT-AR-trained models. Same exam
    # family and IDENTICAL tasksource schema (context/question/answers/label/id_string), so they
    # reuse the lsat-ar standardizer verbatim. We train ONLY on the AR section, so LR and RC are
    # unseen -- but the id_strings share exam ids (e.g. 200010_3-G_* vs 200010_1-LR1_*), so the
    # passages MUST be checked for overlap against the lsat-ar train/test pools before use.
    # lsat-lr = logical reasoning (what ReClor is modelled on); lsat-rc = reading comprehension.
    # val+test joined for a bigger held-out set (we never train on these): lsat-lr 510 -> 1016 q,
    # lsat-rc 269 -> 539 q / 40 -> 80 passages. Verified 0 id_string collisions and 0 shared
    # contexts between the two splits before joining.
    "lsat-lr": {
        "hf_dataset": "tasksource/lsat-lr",
        "subset": None,
        "format": "tasksource_lsat_ar",
        "num_options": 5,
        "join_splits": ["validation", "test"],
    },
    "lsat-rc": {
        "hf_dataset": "tasksource/lsat-rc",
        "subset": None,
        "format": "tasksource_lsat_ar",
        "num_options": 5,
        "join_splits": ["validation", "test"],
    },
    "reclor": {
        "hf_dataset": "metaeval/reclor",
        "subset": None,
        "format": "reclor",
        "num_options": 4,
        # ReClor's real test split has hidden labels (leaderboard); use the labeled
        # validation split as our held-out eval ("test").
        "split_map": {"test": "validation"},
    },
    # ProofWriter (rule-based deductive reasoning): one theory (facts+rules) -> many
    # interdependent True/False/Unknown queries -> joint understanding of the rule set.
    # `config_filter` selects the theory reasoning depth (difficulty knob). 3-way MCQ.
    "proofwriter-d2": {
        "hf_dataset": "tasksource/proofwriter", "subset": None, "format": "proofwriter",
        "num_options": 3, "config_filter": "depth-2",
    },
    "proofwriter-d3": {
        "hf_dataset": "tasksource/proofwriter", "subset": None, "format": "proofwriter",
        "num_options": 3, "config_filter": "depth-3",
    },
    "proofwriter-d5": {
        "hf_dataset": "tasksource/proofwriter", "subset": None, "format": "proofwriter",
        "num_options": 3, "config_filter": "depth-5",
    },
    # QuAIL: narrative reading comprehension with TYPED reasoning (causality, temporal,
    # event-duration, belief-states, ...) + unanswerable questions. ~18 Q/passage in the
    # source; we cap at 6/passage (max_questions_per_passage) to mirror lsat-ar's grouping.
    # A harder, NON-LSAT second multi-question dataset.
    "quail": {
        "hf_dataset": "textmachinelab/quail", "subset": None, "format": "quail",
        "num_options": 4, "max_questions_per_passage": 6,
        # QuAIL's hidden-label test split -> use validation as our held-out "test".
        "split_map": {"test": "validation"},
    },
    # LogiQA: DROPPED 2026-07-13 -- no usable HF source. Canonical `lucasmccabe/logiqa` is a
    # deprecated dataset SCRIPT (unsupported by current `datasets`); the only loadable mirror
    # (datatune/LogiQA2.0) is one unstructured `text` column. `_standardize_logiqa` is kept
    # (unused) in case a clean parquet source appears; just re-add a registry entry then.
    # ARC-Challenge: grade-school science MC, NO reading passage (question + ~4 options).
    # Very OOD -- knowledge/science, no passage or logic-puzzle structure.
    "arc-challenge": {
        "hf_dataset": "allenai/ai2_arc", "subset": "ARC-Challenge", "format": "arc",
        "num_options": 4,
    },
    # --- Transfer/OOD targets added 2026-07-14 (replacing proofwriter-d2, which the base model
    # already saturates at 76-82% with huge seed variance). ---
    # CLUTRR: inductive kinship reasoning. Story describes a chain of family relations; infer the
    # relation between two people. The passage IS a relation graph to extract -> ideal understanding
    # target. NOTE: canonical `CLUTRR/v1` is a dead dataset SCRIPT; `tasksource/clutrr` is the clean
    # parquet mirror (test=1048, 18 relations as a proper ClassLabel).
    # Two variants on purpose: 18-way is canonical CLUTRR (chance 5.6%, no distractor-sampling
    # confound); mc4 samples 3 distractors (chance 25%) to match RACE/ARC's option count. Comparing
    # them isolates how much the option-set size alone moves the number.
    "clutrr": {
        "hf_dataset": "tasksource/clutrr", "subset": None, "format": "clutrr",
        "num_options": 18,
    },
    "clutrr-mc4": {
        "hf_dataset": "tasksource/clutrr", "subset": None, "format": "clutrr",
        "num_options": 4,
    },
    # FOLIO: human-written natural-language first-order logic. premises -> conclusion is
    # True/False/Uncertain. Deductive like ProofWriter but genuinely hard (real headroom).
    # Test split is unlabeled -> use validation (203 examples; small, expect wide SEM).
    "folio": {
        "hf_dataset": "yale-nlp/FOLIO", "subset": None, "format": "folio",
        "num_options": 3, "split_map": {"test": "validation"},
    },
    # CosmosQA: commonsense reading-comprehension MC over a passage (RACE-shaped near-transfer).
    # NOTE: `allenai/cosmos_qa` is a dead dataset SCRIPT; `Samsoup/cosmos_qa` is the loadable
    # mirror, but ITS TEST SPLIT IS UNLABELED (label == -1) -> use validation (2985, balanced).
    "cosmosqa": {
        "hf_dataset": "Samsoup/cosmos_qa", "subset": None, "format": "cosmosqa",
        "num_options": 4, "split_map": {"test": "validation"},
    },
    # BBH logical_deduction: ordering/constraint puzzles as MC -- structurally the closest public
    # analogue of LSAT-AR in a different surface form. Three configs (3/5/7 objects) pooled = 750.
    # Options are embedded in the prompt text and must be parsed out.
    "bbh-logical-deduction": {
        "hf_dataset": "lukaemon/bbh", "format": "bbh_logical_deduction",
        "subsets": ["logical_deduction_three_objects", "logical_deduction_five_objects",
                    "logical_deduction_seven_objects"],
    },
    # --- TRUE LSAT-AR analogues (analytical reasoning / "logic games": constraint satisfaction
    # over a fixed entity set). Added 2026-07-14 because the near-LSAT column had none: ReClor and
    # lsat-lr are both LOGICAL reasoning (argument analysis), which is a different skill from
    # ANALYTICAL reasoning, and lsat-lr accordingly came back null.
    # ZebraLogic mc_mode: "There are 6 houses... each occupied by a different person..." -- an
    # Einstein puzzle, i.e. exactly an LSAT logic game, in MC form.
    "zebralogic": {
        "hf_dataset": "WildEval/ZebraLogic", "subset": "mc_mode", "format": "zebralogic",
    },
    # QuALITY: long-document 4-way RC (~6.5k-token short stories). Understanding stress test --
    # far longer passages than RACE (~435 tok), so if understanding = passage interpretation the
    # gain should be LARGER here. EVAL-ONLY (articles exceed the 2048 training cap; eval uses 32k
    # context). Test split unlabeled -> validation. `emozilla/quality` is the loadable parquet
    # mirror (the canonical NYU QuALITY is a dataset script).
    "quality": {
        "hf_dataset": "emozilla/quality", "subset": None, "format": "quality",
        "num_options": 4, "split_map": {"test": "validation"},
    },
    # tracking_shuffled_objects: track state through a sequence of swaps. Same input/target format
    # as logical_deduction, but its body ends mid-sentence -> needs a completion-style question.
    "bbh-tracking": {
        "hf_dataset": "lukaemon/bbh", "format": "bbh_logical_deduction",
        "subsets": ["tracking_shuffled_objects_three_objects",
                    "tracking_shuffled_objects_five_objects",
                    "tracking_shuffled_objects_seven_objects"],
        "question_text": "Which option correctly completes the statement above?",
    },
}


def _normalize_dataset_name(dataset_name="race-c", subset=None):
    if dataset_name:
        return str(dataset_name).strip()
    if subset:
        return f"race-{str(subset).strip()}"
    return "race-c"


def _sample_tag(num_samples):
    return num_samples if num_samples is not None else "all"


def preprocessed_data_path(dataset_name, split, seed, num_samples, output_dir="data"):
    sample_tag = _sample_tag(num_samples)
    return os.path.join(
        output_dir,
        dataset_name,
        f"{split}_{seed}_{sample_tag}.jsonl",
    )


def _legacy_preprocessed_data_path(split, subset, seed, num_samples, output_dir="data"):
    sample_tag = _sample_tag(num_samples)
    return os.path.join(output_dir, f"race_{split}_{subset}_{seed}_{sample_tag}.jsonl")


def _save_jsonl(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def _label_to_answer(label: Any) -> str:
    if label is None:
        return ""
    if isinstance(label, str):
        stripped = label.strip().upper()
        if len(stripped) == 1 and stripped in string.ascii_uppercase[:26]:
            return stripped
        try:
            label = int(stripped)
        except ValueError:
            return ""
    if isinstance(label, int) and 0 <= label < 26:
        return string.ascii_uppercase[label]
    return ""


def _coerce_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, list):
        return [str(option) for option in options]
    if isinstance(options, tuple):
        return [str(option) for option in options]
    return [str(options)]


def _load_registered_dataset(dataset_name, split, subset=None):
    spec = DATASET_REGISTRY.get(dataset_name, {})
    hf_dataset = spec.get("hf_dataset", dataset_name)
    hf_subset = spec.get("subset") if spec else subset

    # Some datasets name their splits differently (e.g. ReClor has no labeled
    # "test" split, so we map test -> validation).
    eff_split = spec.get("split_map", {}).get(split, split)

    # `subsets` (list) concatenates several HF configs into one dataset -- BBH ships each
    # logical_deduction size as its own config, and we want them pooled into one eval set.
    hf_subsets = spec.get("subsets")
    if hf_subsets:
        from datasets import concatenate_datasets
        parts = [load_dataset(hf_dataset, s, split=eff_split) for s in hf_subsets]
        return concatenate_datasets(parts), spec

    # `join_splits` pools several HF SPLITS into our "test" -- these are eval-only transfer targets
    # we never train on, so val+test is just a bigger held-out set (lsat-ar's own final_test is
    # val+test joined the same way). Only safe because val/test were verified to share no
    # id_strings and no contexts; a collision would fuse two passages into one record, since the
    # standardizer groups by id.
    join = spec.get("join_splits")
    if join and split == "test":
        from datasets import concatenate_datasets
        parts = [load_dataset(hf_dataset, hf_subset, split=s) if hf_subset
                 else load_dataset(hf_dataset, split=s) for s in join]
        return concatenate_datasets(parts), spec

    if hf_subset:
        return load_dataset(hf_dataset, hf_subset, split=eff_split), spec
    return load_dataset(hf_dataset, split=eff_split), spec


def _standardize_ehovy_race(raw_dataset):
    grouped = {}
    for row in raw_dataset:
        example_id = row["example_id"]
        if example_id not in grouped:
            grouped[example_id] = {
                "example_id": example_id,
                "article": row["article"],
                "questions": []
            }

        grouped[example_id]["questions"].append({
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"]
        })

    return list(grouped.values())


def _standardize_tasksource_race(raw_dataset):
    grouped = {}
    for idx, row in enumerate(raw_dataset):
        example_id = str(row.get("id") or f"example_{idx}")
        article = row.get("article", "")
        question = row.get("question", "")
        options = row.get("option", row.get("options", []))
        answer = _label_to_answer(row.get("label"))

        if example_id not in grouped:
            grouped[example_id] = {
                "example_id": example_id,
                "article": article,
                "questions": [],
            }

        grouped[example_id]["questions"].append({
            "question": question,
            "options": _coerce_options(options),
            "answer": answer,
        })

    return list(grouped.values())


def _split_lsat_id(id_string: Any, fallback_idx: int) -> tuple[str, int]:
    raw_id = str(id_string or f"example_{fallback_idx}_0")
    parts = raw_id.split("_")
    if len(parts) <= 1:
        return raw_id, fallback_idx

    passage_id = "_".join(parts[:-1])
    try:
        question_idx = int(parts[-1])
    except ValueError:
        question_idx = fallback_idx

    return passage_id, question_idx


def _standardize_tasksource_lsat_ar(raw_dataset):
    grouped = {}
    question_order = {}
    for idx, row in enumerate(raw_dataset):
        example_id, question_idx = _split_lsat_id(row.get("id_string"), idx)
        options = row.get("answers", [])

        if example_id not in grouped:
            grouped[example_id] = {
                "example_id": example_id,
                "article": row.get("context", ""),
                "questions": [],
            }
            question_order[example_id] = []

        grouped[example_id]["questions"].append({
            "question": row.get("question", ""),
            "options": _coerce_options(options),
            "answer": _label_to_answer(row.get("label")),
        })
        question_order[example_id].append(question_idx)

    for example_id, record in grouped.items():
        ordered_questions = sorted(
            zip(question_order[example_id], record["questions"]),
            key=lambda item: item[0],
        )
        record["questions"] = [question for _, question in ordered_questions]

    return list(grouped.values())


def _standardize_reclor(raw_dataset):
    # ReClor rows are (context, question, answers[4], label). Group questions that
    # share the exact same context into one passage record (multi-Q where it exists,
    # otherwise a single-question record).
    grouped = {}
    order = []
    for idx, row in enumerate(raw_dataset):
        context = row.get("context", "")
        if context not in grouped:
            grouped[context] = {
                "example_id": str(row.get("id_string") or f"reclor_{idx}"),
                "article": context,
                "questions": [],
            }
            order.append(context)
        grouped[context]["questions"].append({
            "question": row.get("question", ""),
            "options": _coerce_options(row.get("answers", [])),
            "answer": _label_to_answer(row.get("label")),
        })
    return [grouped[c] for c in order]


def _standardize_logiqa(raw_dataset):
    # LogiQA rows: context, query, options[4], correct_option (int 0-3). One Q per record.
    out = []
    for idx, row in enumerate(raw_dataset):
        ci = row.get("correct_option")
        if isinstance(ci, bool):
            ci = None
        if isinstance(ci, int) or (isinstance(ci, str) and str(ci).isdigit()):
            ans = chr(ord("A") + int(ci))
        else:
            ans = _label_to_answer(ci)
        out.append({
            "example_id": f"logiqa_{idx}",
            "article": str(row.get("context", "") or ""),
            "questions": [{
                "question": str(row.get("query", "") or ""),
                "options": _coerce_options(row.get("options", [])),
                "answer": ans,
            }],
        })
    return out


def _standardize_arc(raw_dataset):
    # ARC rows: question, choices={text:[...], label:[...]}, answerKey (letter or number).
    # NO passage -> article is empty. answerKey maps to the option index via the label list.
    out = []
    for idx, row in enumerate(raw_dataset):
        ch = row.get("choices", {}) or {}
        opts = _coerce_options(ch.get("text", []))
        labels = [str(x) for x in ch.get("label", [])]
        key = str(row.get("answerKey", "") or "").strip()
        if key in labels:
            ans = chr(ord("A") + labels.index(key))
        else:
            ans = _label_to_answer(key)
        if not opts or not ans:
            continue
        out.append({
            "example_id": str(row.get("id") or f"arc_{idx}"),
            "article": "",  # science QA has no reading passage
            "questions": [{
                "question": str(row.get("question", "") or ""),
                "options": opts,
                "answer": ans,
            }],
        })
    return out


def _standardize_clutrr(raw_dataset, num_options=18, seed=42):
    """CLUTRR: story of kinship relations -> infer the relation between two named people.

    Rows: sentence1 = story (names wrapped in [brackets]), sentence2 = "('A', 'B')" query tuple,
    labels = index into the 18-relation ClassLabel. The query (a, b) means "b is a's <relation>",
    so the question is phrased "How is b related to a?".

    num_options=18 -> every question offers the full relation set (canonical CLUTRR, chance 5.6%).
    num_options=N  -> correct + (N-1) distractors sampled from the other relations with a per-example
    seeded RNG, then shuffled (deterministic across runs).
    """
    import ast
    import random

    try:
        names = list(raw_dataset.features["labels"].names)
    except (KeyError, AttributeError):
        return []
    all_rel = sorted(names)

    out = []
    for idx, row in enumerate(raw_dataset):
        lab = row.get("labels")
        if not isinstance(lab, int) or not (0 <= lab < len(names)):
            continue
        gold = names[lab]
        story = str(row.get("sentence1", "") or "").replace("[", "").replace("]", "")
        try:
            a, b = ast.literal_eval(str(row.get("sentence2", "")))
        except (ValueError, SyntaxError):
            continue
        if not story.strip():
            continue

        if num_options >= len(all_rel):
            opts = all_rel
        else:
            rng = random.Random(f"{seed}_{idx}")
            distract = rng.sample([r for r in all_rel if r != gold], k=max(0, num_options - 1))
            opts = distract + [gold]
            rng.shuffle(opts)
        if gold not in opts:
            continue
        out.append({
            "example_id": f"clutrr_{idx}",
            "article": story,
            "questions": [{
                "question": f"How is {b} related to {a}?",
                "options": opts,
                "answer": chr(ord("A") + opts.index(gold)),
            }],
        })
    return out


_FOLIO_ANS2LETTER = {"true": "A", "false": "B", "uncertain": "C"}


def _standardize_folio(raw_dataset):
    """FOLIO: premises -> is the conclusion True / False / Uncertain."""
    opts = ["True", "False", "Uncertain"]
    out = []
    for idx, row in enumerate(raw_dataset):
        premises = str(row.get("premises", "") or "").strip()
        conclusion = str(row.get("conclusion", "") or "").strip()
        ans = _FOLIO_ANS2LETTER.get(str(row.get("label", "") or "").strip().lower())
        if not premises or not conclusion or not ans:
            continue
        out.append({
            "example_id": str(row.get("example_id") or f"folio_{idx}"),
            "article": premises,
            "questions": [{
                "question": ("Based on the premises above, is the following conclusion true, "
                             f"false, or uncertain?\n{conclusion}"),
                "options": opts,
                "answer": ans,
            }],
        })
    return out


def _standardize_cosmosqa(raw_dataset):
    """CosmosQA: passage + question + 4 options. Rows with label == -1 are the unlabeled
    leaderboard split and are dropped (the registry maps test -> validation to avoid them)."""
    out = []
    for idx, row in enumerate(raw_dataset):
        lab = row.get("label")
        if not isinstance(lab, int) or lab < 0 or lab > 3:
            continue
        opts = [str(row.get(f"answer{i}", "") or "") for i in range(4)]
        ctx = str(row.get("context", "") or "").strip()
        q = str(row.get("question", "") or "").strip()
        if not ctx or not q or not all(opts):
            continue
        out.append({
            "example_id": str(row.get("id") or f"cosmosqa_{idx}"),
            "article": ctx,
            "questions": [{"question": q, "options": opts, "answer": chr(ord("A") + lab)}],
        })
    return out


def _standardize_bbh_logical_deduction(raw_dataset, question_text=None):
    """BBH: one `input` blob (puzzle text + "Options:" + "(A) ..." lines) and a `target` like "(A)".
    Split the blob into the constraint paragraph (article) and the options, so the understanding
    role has the puzzle to work on.

    `question_text` matters: logical_deduction's body is a self-contained puzzle ("which statement
    is correct?"), but tracking_shuffled_objects' body ends mid-sentence ("Alice is dancing with"),
    so it needs a completion-style question instead.
    """
    import re

    question_text = question_text or "Which of the following statements is correct?"
    out = []
    for idx, row in enumerate(raw_dataset):
        text = str(row.get("input", "") or "")
        target = str(row.get("target", "") or "").strip()
        if "Options:" not in text:
            continue
        body, _, opts_blob = text.partition("Options:")
        opts, labels = [], []
        for m in re.finditer(r"\(([A-Z])\)\s*([^\n]+)", opts_blob):
            labels.append(m.group(1))
            opts.append(m.group(2).strip())
        ans = target.strip("()").strip().upper()
        if not opts or ans not in labels:
            continue
        # Re-letter to A.. in the parsed order (labels are already A,B,C.. but do not assume).
        out.append({
            "example_id": f"bbh_ld_{idx}",
            "article": body.strip(),
            "questions": [{
                "question": question_text,
                "options": opts,
                "answer": chr(ord("A") + labels.index(ans)),
            }],
        })
    return out


def _standardize_quality(raw_dataset):
    """QuALITY: LONG-document (short stories, ~6.5k tokens) 4-way RC. The ideal understanding
    stress test -- if understanding = passage interpretation, a long passage should benefit MOST.
    Rows: article, question, options (list of 4), answer (0-indexed int), hard (bool). Group
    questions that share an article into one record, like RACE.
    NOTE: articles far exceed the 2048 training prompt cap, so this is an EVAL/transfer target only
    (eval uses Qwen3's 32k context, no truncation). `hard` is dropped here; filter upstream if
    a hard-only split is wanted."""
    grouped, order = {}, []
    for idx, row in enumerate(raw_dataset):
        art = str(row.get("article", "") or "").strip()
        opts = _coerce_options(row.get("options", []))
        a = row.get("answer")
        if not art or len(opts) < 2 or not isinstance(a, int) or not (0 <= a < len(opts)):
            continue
        key = hashlib.md5(art.encode()).hexdigest()
        if key not in grouped:
            grouped[key] = {"example_id": f"quality_{key[:8]}", "article": art, "questions": []}
            order.append(key)
        grouped[key]["questions"].append({
            "question": str(row.get("question", "") or ""),
            "options": opts,
            "answer": chr(ord("A") + a),
        })
    return [grouped[k] for k in order if grouped[k]["questions"]]


def _standardize_zebralogic(raw_dataset):
    """ZebraLogic mc_mode: Einstein/zebra constraint puzzles -- the closest public analogue of an
    LSAT-AR logic game (fixed entity set, positional constraints, unique consistent assignment).
    Rows: puzzle (constraints), question, choices (list), answer (the choice TEXT, not an index)."""
    out = []
    for idx, row in enumerate(raw_dataset):
        opts = _coerce_options(row.get("choices", []))
        gold = str(row.get("answer", "") or "").strip()
        puzzle = str(row.get("puzzle", "") or "").strip()
        q = str(row.get("question", "") or "").strip()
        if not puzzle or not q or len(opts) < 2 or gold not in opts:
            continue
        out.append({
            "example_id": str(row.get("id") or f"zebra_{idx}"),
            "article": puzzle,
            "questions": [{
                "question": q,
                "options": opts,
                "answer": chr(ord("A") + opts.index(gold)),
            }],
        })
    return out


_PROOFWRITER_ANS2LETTER = {"True": "A", "False": "B", "Unknown": "C"}


def _standardize_proofwriter(raw_dataset, config_filter=None):
    # Group queries that share the same theory into one record (joint understanding of
    # the rule set). Optionally restrict to one reasoning depth via `config_filter`.
    grouped = {}
    order = []
    for idx, row in enumerate(raw_dataset):
        if config_filter is not None and str(row.get("config")) != config_filter:
            continue
        theory = row.get("theory", "")
        answer = _PROOFWRITER_ANS2LETTER.get(str(row.get("answer", "")), "")
        if not answer:
            continue
        if theory not in grouped:
            grouped[theory] = {
                "example_id": str(row.get("id") or f"proofwriter_{idx}"),
                "article": theory,
                "questions": [],
            }
            order.append(theory)
        grouped[theory]["questions"].append({
            "question": str(row.get("question", "")),
            "options": ["True", "False", "Unknown"],
            "answer": answer,
        })
    return [grouped[t] for t in order if grouped[t]["questions"]]


def _standardize_quail(raw_dataset, max_questions_per_passage=6):
    # Group QuAIL rows by context_id (one record per passage). Cap to the first
    # `max_questions_per_passage` questions (by question_id) so the per-passage question
    # count matches lsat-ar (~6) rather than QuAIL's native ~18. 4 options; gold is the
    # integer correct_answer_id (0-3) -> letter.
    grouped = {}
    order = []
    for idx, row in enumerate(raw_dataset):
        cid = str(row.get("context_id") or f"quail_{idx}")
        if cid not in grouped:
            grouped[cid] = {"example_id": cid, "article": row.get("context", ""),
                            "questions": [], "_qids": []}
            order.append(cid)
        grouped[cid]["questions"].append({
            "question": row.get("question", ""),
            "options": _coerce_options(row.get("answers", [])),
            "answer": _label_to_answer(row.get("correct_answer_id")),
        })
        grouped[cid]["_qids"].append(int(row.get("question_id", idx)))
    out = []
    for cid in order:
        rec = grouped[cid]
        ordered = [q for _, q in sorted(zip(rec["_qids"], rec["questions"]), key=lambda x: x[0])]
        rec["questions"] = [q for q in ordered if q["answer"]][:max_questions_per_passage]
        del rec["_qids"]
        if rec["questions"]:
            out.append(rec)
    return out


def preprocess_race_data(
    num_samples=None,
    split="train",
    subset=None,
    seed=42,
    save_jsonl=True,
    output_dir="data",
    dataset_name="race-c",
):
    dataset_name = _normalize_dataset_name(dataset_name, subset)
    raw_dataset, spec = _load_registered_dataset(dataset_name, split=split, subset=subset)
    dataset_format = spec.get("format", "tasksource_race")

    # Group rows by example_id into one record per article with a questions list.
    if dataset_format == "ehovy_race":
        grouped_examples = _standardize_ehovy_race(raw_dataset)
    elif dataset_format == "tasksource_race":
        grouped_examples = _standardize_tasksource_race(raw_dataset)
    elif dataset_format == "tasksource_lsat_ar":
        grouped_examples = _standardize_tasksource_lsat_ar(raw_dataset)
    elif dataset_format == "reclor":
        grouped_examples = _standardize_reclor(raw_dataset)
    elif dataset_format == "proofwriter":
        grouped_examples = _standardize_proofwriter(raw_dataset, spec.get("config_filter"))
    elif dataset_format == "quail":
        grouped_examples = _standardize_quail(raw_dataset, spec.get("max_questions_per_passage", 6))
    elif dataset_format == "logiqa":
        grouped_examples = _standardize_logiqa(raw_dataset)
    elif dataset_format == "arc":
        grouped_examples = _standardize_arc(raw_dataset)
    elif dataset_format == "clutrr":
        grouped_examples = _standardize_clutrr(raw_dataset, spec.get("num_options", 18), seed)
    elif dataset_format == "folio":
        grouped_examples = _standardize_folio(raw_dataset)
    elif dataset_format == "cosmosqa":
        grouped_examples = _standardize_cosmosqa(raw_dataset)
    elif dataset_format == "bbh_logical_deduction":
        grouped_examples = _standardize_bbh_logical_deduction(raw_dataset, spec.get("question_text"))
    elif dataset_format == "zebralogic":
        grouped_examples = _standardize_zebralogic(raw_dataset)
    elif dataset_format == "quality":
        grouped_examples = _standardize_quality(raw_dataset)
    else:
        raise ValueError(f"Unsupported dataset format for {dataset_name}: {dataset_format}")

    rng = random.Random(seed)
    rng.shuffle(grouped_examples)

    if num_samples is not None:
        grouped_examples = grouped_examples[:min(num_samples, len(grouped_examples))]

    output_path = None
    if save_jsonl:
        output_path = preprocessed_data_path(
            dataset_name=dataset_name,
            split=split,
            seed=seed,
            num_samples=num_samples,
            output_dir=output_dir,
        )
        _save_jsonl(grouped_examples, output_path)

    return Dataset.from_list(grouped_examples), output_path


def create_or_load_preprocessed_data(
    num_samples=None,
    split="train",
    subset=None,
    seed=42,
    output_dir="data",
    dataset_name="race-c",
):
    dataset_name = _normalize_dataset_name(dataset_name, subset)
    output_path = preprocessed_data_path(
        dataset_name=dataset_name,
        split=split,
        seed=seed,
        num_samples=num_samples,
        output_dir=output_dir,
    )
    if os.path.exists(output_path):
        print(f"Loading preprocessed data from {output_path}...")
        with open(output_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        return Dataset.from_list(records), output_path

    legacy_path = _legacy_preprocessed_data_path(split, subset, seed, num_samples, output_dir)
    if subset and dataset_name == f"race-{subset}" and os.path.exists(legacy_path):
        print(f"Loading legacy preprocessed data from {legacy_path}...")
        with open(legacy_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        _save_jsonl(records, output_path)
        print(f"Migrated legacy preprocessed data to {output_path}.")
        return Dataset.from_list(records), output_path

    print(f"No preprocessed data found at {output_path}. Preprocessing now...")
    return preprocess_race_data(
        num_samples=num_samples,
        split=split,
        subset=subset,
        seed=seed,
        save_jsonl=True,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    print("Preprocessing Test...")
    dataset, output_path = preprocess_race_data(
        num_samples=3,
        split="train",
        subset=None,
        seed=42,
        dataset_name="race-c",
    )
    print("Number of grouped examples:", len(dataset))
    print("Saved JSONL:", output_path)
    print("\nOne grouped example:\n")
    print(dataset[0])

    with open(output_path, "r", encoding="utf-8") as f:
        print("\nFirst JSONL line:\n")
        print(f.readline().strip())
