"""LoRA SFT on GSM8K worked solutions (plain PyTorch loop, loss on the answer tokens only).

Examples:
  python src/sft.py --limit 64 --epochs 1 --val 16 --out outputs/sft_smoke   # quick test
  python src/sft.py                                                         # full run
Then evaluate:  sbatch jobs/eval.sbatch --adapter outputs/sft --max-new-tokens 512 --out results/sft.json
"""
import argparse
import json
import math
import os
import random
import re
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_NAME, build_messages, load_gsm8k

CALC = re.compile(r"<<[^>]*>>")  # GSM8K calculator annotations like <<48/2=24>>
END = "<|im_end|>"               # Qwen's end-of-turn token; teaches the model when to stop


def clean_solution(solution):
    return CALC.sub("", solution).strip()


def encode(tok, row, max_len):
    """(input_ids, labels) with the prompt masked to -100, or None if too long."""
    prompt = tok.apply_chat_template(build_messages(row["question"]), tokenize=False,
                                     add_generation_prompt=True)
    completion = clean_solution(row["solution"]) + END
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    assert c[-1] == tok.convert_tokens_to_ids(END), "end-of-turn token was not tokenized"
    if len(p) + len(c) > max_len:
        return None
    return p + c, [-100] * len(p) + c


def collate(examples, pad_id):
    n = max(len(ids) for ids, _ in examples)
    input_ids, labels, mask = [], [], []
    for ids, lab in examples:
        pad = n - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(mask)


def lr_factor(step, total, warmup):
    """Linear warmup, then cosine decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def to_cuda(batch):
    return [t.to("cuda") for t in batch]


@torch.no_grad()
def val_loss(model, data, pad_id, bs=32):
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(data), bs):
        ids, labels, mask = to_cuda(collate(data[i:i + bs], pad_id))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        n = int((labels[:, 1:] != -100).sum())
        total += loss.item() * n
        count += n
    model.train()
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out", default="outputs/sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val", type=int, default=200, help="held-out training questions for val loss")
    ap.add_argument("--limit", type=int, default=None, help="use only N training examples (debug)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    rows = load_gsm8k("train")
    rng.shuffle(rows)
    val_rows, train_rows = rows[:args.val], rows[args.val:]
    if args.limit:
        train_rows = train_rows[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id
    train = [e for e in (encode(tok, r, args.max_len) for r in train_rows) if e is not None]
    val = [e for e in (encode(tok, r, args.max_len) for r in val_rows) if e is not None]
    lens = [len(ids) for ids, _ in train]
    print(f"train {len(train)} (dropped {len(train_rows) - len(train)} too long), val {len(val)}; "
          f"tokens/example mean {sum(lens) / len(lens):.0f} max {max(lens)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    n_micro = math.ceil(len(train) / args.micro_batch)
    steps_per_epoch = math.ceil(n_micro / args.accum)
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warmup))
    print(f"{total} optimizer steps ({steps_per_epoch}/epoch), effective batch "
          f"{args.micro_batch * args.accum}, warmup {warmup}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "w")

    def log(record):
        record["elapsed_s"] = round(time.time() - t0)
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    if val:
        v = val_loss(model, val, pad_id)
        print(f"step 0 | val loss {v:.4f}", flush=True)
        log({"step": 0, "val_loss": v})

    model.train()
    step, run_loss, run_steps = 0, 0.0, 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        micro = [order[i:i + args.micro_batch] for i in range(0, len(order), args.micro_batch)]
        for g in range(0, len(micro), args.accum):
            group = micro[g:g + args.accum]
            step_loss = 0.0
            for idxs in group:
                ids, labels, mask = to_cuda(collate([train[i] for i in idxs], pad_id))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                (loss / len(group)).backward()
                step_loss += loss.item() / len(group)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += step_loss
            run_steps += 1

            if step % args.log_every == 0 or step == total:
                lr = sched.get_last_lr()[0]
                print(f"step {step}/{total} | epoch {epoch + 1} | train loss {run_loss / run_steps:.4f} "
                      f"| grad norm {grad_norm:.2f} | lr {lr:.2e} | {time.time() - t0:.0f}s", flush=True)
                log({"step": step, "epoch": epoch + 1, "train_loss": run_loss / run_steps,
                     "grad_norm": grad_norm, "lr": lr})
                run_loss, run_steps = 0.0, 0
            if val and (step % args.eval_every == 0 or step == total):
                v = val_loss(model, val, pad_id)
                print(f"step {step} | val loss {v:.4f}", flush=True)
                log({"step": step, "val_loss": v})

    model.save_pretrained(args.out)
    with open(os.path.join(args.out, "sft_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    print(f"saved adapter to {args.out}", flush=True)

    # eyeball a few greedy generations from held-out questions
    model.eval()
    for r in val_rows[:3]:
        prompt = tok.apply_chat_template(build_messages(r["question"]), tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, repetition_penalty=1.0,
                                 pad_token_id=pad_id, use_cache=True)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n--- gold {r['gold']} ---\nQ: {r['question']}\nA: {text}", flush=True)


if __name__ == "__main__":
    main()
