import json
import re
import string
from typing import Any, Dict, List

def extract_boxed_letter(text: str, num_options: int = 4) -> str:
    """Extract the final boxed letter (e.g. A-D) from the model's output text.
    The function first looks for the last occurrence of a boxed letter in the entire text.
    If none is found, it looks for the last standalone letter in the last 128 characters of the text.
    If still none is found, it returns an empty string."""
    
    valid_letters = string.ascii_uppercase[:num_options]
    pattern_letters = valid_letters + valid_letters.lower()
    
    matches = re.findall(r"\\boxed\s*\{\s*([" + pattern_letters + r"])\s*\}", text or "")
    if matches:
        return matches[-1].upper()

    # tail = text[-128:] if text else ""
    # plain = re.findall(r"\b([" + pattern_letters + r"])\b", tail)
    # if plain:
    #     return plain[-1].upper()
    return ""

def normalize_gold_letter(value: Any, num_options: int = 4) -> str:
    if value is None:
        return ""
    v = str(value).strip().upper()
    if len(v) == 1 and v in string.ascii_uppercase[:num_options]:
        return v
    return ""

def parse_questions(reference_obj: Any) -> List[Dict[str, Any]]:
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

def valid_questions(reference_obj: Any) -> List[Dict[str, Any]]:
    valid = []
    for q in parse_questions(reference_obj):
        q_text = q.get("question", "")
        opts = q.get("options", [])
        gold = normalize_gold_letter(q.get("answer", ""), num_options=len(opts) if isinstance(opts, list) else 4)
        if isinstance(opts, list) and len(opts) >= 2 and q_text and gold:
            valid.append({"question": q_text, "options": opts, "answer": gold})
    return valid

