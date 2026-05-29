from __future__ import annotations

from datasets import load_dataset, Dataset
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

    if hf_subset:
        return load_dataset(hf_dataset, hf_subset, split=split), spec
    return load_dataset(hf_dataset, split=split), spec


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
