from datasets import load_dataset, Dataset
import json
import os
import random

def _save_jsonl(records, split, subset, seed, num_samples, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    sample_tag = num_samples if num_samples is not None else "all"
    output_path = os.path.join(
        output_dir,
        f"race_{split}_{subset}_{seed}_{sample_tag}.jsonl"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path

def preprocess_race_data(num_samples=None, split="train", subset="high", seed=42, save_jsonl=True):
    dataset_name = "ehovy/race"
    raw_dataset = load_dataset(dataset_name, subset, split=split)

    # Group rows by example_id into one record per article with a questions list.
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

    grouped_examples = list(grouped.values())

    rng = random.Random(seed)
    rng.shuffle(grouped_examples)

    if num_samples is not None:
        grouped_examples = grouped_examples[:min(num_samples, len(grouped_examples))]

    output_path = None
    if save_jsonl:
        output_path = _save_jsonl(
            grouped_examples,
            split=split,
            subset=subset,
            seed=seed,
            num_samples=num_samples
        )

    return Dataset.from_list(grouped_examples), output_path

def create_or_load_preprocessed_data(num_samples=None, split="train", subset="high", seed=42, output_dir="data"):
    sample_tag = num_samples if num_samples is not None else "all"
    output_path = os.path.join(
        output_dir,
        f"race_{split}_{subset}_{seed}_{sample_tag}.jsonl"
    )
    if os.path.exists(output_path):
        print(f"Loading preprocessed data from {output_path}...")
        with open(output_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        return Dataset.from_list(records), output_path
    else:
        print(f"No preprocessed data found at {output_path}. Preprocessing now...")
        return preprocess_race_data(num_samples=num_samples, split=split, subset=subset, seed=seed, save_jsonl=True)


if __name__ == "__main__":
    print("Preprocessing Test...")
    dataset, output_path = preprocess_race_data(num_samples=3, split="train", subset="high", seed=42)
    print("Number of grouped examples:", len(dataset))
    print("Saved JSONL:", output_path)
    print("\nOne grouped example:\n")
    print(dataset[0])

    with open(output_path, "r", encoding="utf-8") as f:
        print("\nFirst JSONL line:\n")
        print(f.readline().strip())