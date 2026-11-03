"""Shared pieces: GSM8K loading, prompt format, answer extraction, rewards."""
import math
import re

from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "Reason step by step, showing your calculations, before giving the answer. Only after you have finished reasoning, write the final numeric answer on its own last line in the form '#### 42'."
)

_FINAL = re.compile(r"####\s*\$?\s*(-?[\d,]+\.?\d*)")
_ANY_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_number(s):
    """'1,234.0' -> '1234', '3.50' -> '3.5', anything unparsable -> None."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    if not math.isfinite(f):  # e.g. a degenerate 400-digit string overflows to inf
        return None
    return str(int(f)) if f.is_integer() else str(f)


def extract_answer(text):
    """Return (answer, used_final_marker). Prefers '#### n', else the last number."""
    found = _FINAL.findall(text)
    if found:
        return normalize_number(found[-1]), True
    nums = _ANY_NUM.findall(text)
    if nums:
        return normalize_number(nums[-1]), False
    return None, False


def has_final_marker(text):
    return bool(_FINAL.search(text))


def build_messages(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def load_gsm8k(split):
    """List of dicts: question, solution (human worked solution), gold (normalized number)."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for ex in ds:
        gold = normalize_number(ex["answer"].split("####")[-1])
        rows.append({"question": ex["question"], "solution": ex["answer"], "gold": gold})
    return rows


def correctness_reward(text, gold):
    pred, _ = extract_answer(text)
    return 1.0 if pred is not None and pred == gold else 0.0


def format_reward(text):
    return 0.1 if has_final_marker(text) else 0.0
