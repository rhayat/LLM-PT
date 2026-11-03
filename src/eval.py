"""Evaluate a model (optionally with a LoRA adapter) on GSM8K with greedy decoding.

Examples:
  python src/eval.py --out results/base.json
  python src/eval.py --adapter outputs/sft --out results/sft.json
  python src/eval.py --limit 64 --out results/smoke.json      # quick test
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (MODEL_NAME, build_messages, correctness_reward, extract_answer,
                    has_final_marker, load_gsm8k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    rows = load_gsm8k(args.split)
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s; {len(rows)} examples", flush=True)

    prompts = [
        tok.apply_chat_template(build_messages(r["question"]), tokenize=False,
                                add_generation_prompt=True)
        for r in rows
    ]
    # sort by length so each batch has little padding, then restore the order
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    outputs = [None] * len(rows)
    n_tokens = [0] * len(rows)

    for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            ids = new[j]
            n_tokens[i] = int((ids != tok.pad_token_id).sum())
            outputs[i] = tok.decode(ids, skip_special_tokens=True)
        print(f"  {min(start + args.batch_size, len(order))}/{len(order)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # save raw generations first, so a scoring bug can never lose them
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    raw_path = os.path.splitext(args.out)[0] + ".raw.json"
    with open(raw_path, "w") as f:
        json.dump({"outputs": outputs, "n_tokens": n_tokens}, f)
    print(f"saved raw outputs to {raw_path}", flush=True)

    results, correct, with_marker = [], 0, 0
    for r, text, n in zip(rows, outputs, n_tokens):
        pred, _ = extract_answer(text)
        ok = correctness_reward(text, r["gold"]) == 1.0
        correct += ok
        with_marker += has_final_marker(text)
        results.append({"question": r["question"], "gold": r["gold"], "pred": pred,
                        "correct": ok, "n_tokens": n, "output": text})

    summary = {
        "model": args.model, "adapter": args.adapter, "split": args.split, "n": len(rows),
        "accuracy": correct / len(rows),
        "format_rate": with_marker / len(rows),
        "mean_new_tokens": sum(n_tokens) / len(rows),
        "seconds": round(time.time() - t0),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
