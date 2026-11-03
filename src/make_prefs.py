"""Build DPO preference pairs from the SFT model's own samples, labeled by correctness.

For each training question: sample `group-size` answers, grade each against the gold answer, and pair a
correct answer (chosen) with an incorrect one (rejected). Questions where every answer is right, or every
answer is wrong, give no pair and are skipped.

Examples:
  python src/make_prefs.py --n-questions 32 --group-size 4 --prompts-per-batch 8 \
                           --out data/prefs_smoke.jsonl                          # quick test
  python src/make_prefs.py                                                       # full run
"""
import argparse
import json
import os
import random
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_NAME, correctness_reward, has_final_marker, load_gsm8k
from grpo import SFT_SPLIT_SEED, SFT_VAL, completion_mask, encode_prompts


def select_pairs(correct, wrong, max_pairs, rng):
    """Pair distinct correct and incorrect answers: up to max_pairs, never reusing an answer."""
    k = min(len(correct), len(wrong), max_pairs)
    return list(zip(rng.sample(correct, k), rng.sample(wrong, k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--adapter", default="outputs/sft", help="SFT LoRA adapter to sample from")
    ap.add_argument("--out", default="data/prefs.jsonl")
    ap.add_argument("--n-questions", type=int, default=2000)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-pairs", type=int, default=2, help="pairs kept per question")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    rows = load_gsm8k("train")
    random.Random(SFT_SPLIT_SEED).shuffle(rows)       # same split as sft.py / grpo.py
    train_rows = rows[SFT_VAL:][:args.n_questions]    # the 200 held-out questions are never used

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    eos_ids = tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    print(f"loaded in {time.time() - t0:.0f}s; sampling {args.group_size} answers for "
          f"{len(train_rows)} questions", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    G = args.group_size
    n_q = n_pairs = n_samples = n_correct = n_unfinished = 0
    all_right = all_wrong = mixed = 0
    tok_chosen, tok_rejected = [], []

    with open(args.out, "w") as out_f:
        for start in range(0, len(train_rows), args.prompts_per_batch):
            batch = train_rows[start:start + args.prompts_per_batch]
            enc = encode_prompts(tok, batch)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=args.temperature, top_p=1.0, top_k=0,
                                     repetition_penalty=1.0, num_return_sequences=G,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
            P = enc["input_ids"].shape[1]
            comp = gen[:, P:]
            cmask, finished = completion_mask(comp, eos_ids)
            n_comp, fin = cmask.sum(1).tolist(), finished.tolist()

            for qi, row in enumerate(batch):
                correct, wrong, oks = [], [], []
                for j in range(G):
                    i = qi * G + j
                    text = tok.decode(comp[i, :n_comp[i]], skip_special_tokens=True)
                    ok = correctness_reward(text, row["gold"]) == 1.0
                    oks.append(ok)
                    n_samples += 1
                    n_correct += ok
                    n_unfinished += not fin[i]
                    if not fin[i]:
                        continue  # cut-off answers are never used in a pair
                    if ok and has_final_marker(text):
                        correct.append((text, n_comp[i]))
                    elif not ok:
                        wrong.append((text, n_comp[i]))
                n_q += 1
                all_right += all(oks)
                all_wrong += not any(oks)
                mixed += (any(oks) and not all(oks))
                for (c_text, c_n), (r_text, r_n) in select_pairs(correct, wrong, args.max_pairs, rng):
                    out_f.write(json.dumps({"question": row["question"], "gold": row["gold"],
                                            "chosen": c_text, "rejected": r_text,
                                            "chosen_tokens": c_n, "rejected_tokens": r_n}) + "\n")
                    n_pairs += 1
                    tok_chosen.append(c_n)
                    tok_rejected.append(r_n)
            print(f"  {n_q}/{len(train_rows)} questions | {n_pairs} pairs | {time.time() - t0:.0f}s",
                  flush=True)

    stats = {
        "questions": n_q, "pairs": n_pairs, "samples": n_samples,
        "sample_accuracy": n_correct / n_samples, "unfinished_fraction": n_unfinished / n_samples,
        "questions_all_correct": all_right / n_q, "questions_all_wrong": all_wrong / n_q,
        "questions_mixed": mixed / n_q,
        "mean_tokens_chosen": sum(tok_chosen) / max(1, len(tok_chosen)),
        "mean_tokens_rejected": sum(tok_rejected) / max(1, len(tok_rejected)),
        "seconds": round(time.time() - t0),
    }
    with open(os.path.splitext(args.out)[0] + "_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
