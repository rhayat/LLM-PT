"""GRPO on GSM8K, starting from the SFT LoRA adapter (plain PyTorch loop, verifiable reward).

One training step:
  1. sample `prompts-per-step` questions, generate `group-size` answers for each (temperature 1.0)
  2. score every answer with the reward function
  3. advantage = (reward - group mean) / group std: how much better than its siblings the answer was
  4. one gradient step: raise the probability of above-average answers, lower it for below-average
     ones, with a KL penalty that keeps the policy close to the frozen SFT model

Examples:
  python src/grpo.py --steps 2 --prompts-per-step 4 --group-size 4 --eval-every 2 \
                     --val-eval 16 --out outputs/grpo_smoke                       # quick test
  python src/grpo.py                                                              # full run
  python src/grpo.py --reward length --out outputs/grpo_hack_length               # reward-hacking demo
Then evaluate:  sbatch jobs/eval.sbatch --adapter outputs/grpo --max-new-tokens 512 --out results/grpo.json
"""
import argparse
import json
import os
import random
import re
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (MODEL_NAME, build_messages, correctness_reward, format_reward, load_gsm8k,
                    normalize_number)

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
SFT_SPLIT_SEED, SFT_VAL = 0, 200  # must match sft.py so the 200 held-out questions stay held out


def compute_reward(mode, text, gold, n_tokens, finished, max_new):
    if mode == "verifiable":      # the real objective: correct final answer, small format bonus
        if not finished:          # cut off at the token cap: no reward
            return 0.0
        return correctness_reward(text, gold) + format_reward(text)
    if mode == "length":          # flawed on purpose: pays for long answers, ignores correctness
        return n_tokens / max_new
    if mode == "contains":        # flawed on purpose: gold number anywhere in the text counts
        return 1.0 if gold in {normalize_number(m) for m in NUM.findall(text)} else 0.0
    raise ValueError(f"unknown reward mode {mode}")


def completion_mask(comp, eos_ids):
    """Mask of real completion tokens (up to and including the first end-of-turn token)."""
    is_eos = torch.zeros_like(comp, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= comp == e
    seen_before = is_eos.long().cumsum(1) - is_eos.long()
    return seen_before == 0, is_eos.any(1)


def group_advantages(rewards, eps=1e-4):
    """rewards: (prompts, group) -> flat per-answer advantages."""
    centered = rewards - rewards.mean(1, keepdim=True)
    return (centered / (rewards.std(1, keepdim=True) + eps)).reshape(-1)


def token_logprobs(model, ids, attn, P):
    """Log-probability of each completion token (positions P onward) under `model`."""
    pos = (attn.cumsum(-1) - 1).clamp(min=0)  # position ids that ignore left padding
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).logits
    lg = logits[:, P - 1:-1, :].float()  # the logits at position t predict the token at t+1
    labels = ids[:, P:]
    return lg.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1)


def encode_prompts(tok, rows):
    texts = [tok.apply_chat_template(build_messages(r["question"]), tokenize=False,
                                     add_generation_prompt=True) for r in rows]
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")


@torch.no_grad()
def greedy_accuracy(model, tok, rows, max_new, bs=100):
    correct = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = encode_prompts(tok, chunk)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None,
                             top_p=None, top_k=None, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id, use_cache=True)
        for r, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            text = tok.decode(g, skip_special_tokens=True)
            correct += correctness_reward(text, r["gold"]) == 1.0
    return correct / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--init-adapter", default="outputs/sft", help="SFT LoRA adapter to start from")
    ap.add_argument("--out", default="outputs/grpo")
    ap.add_argument("--reward", default="verifiable", choices=["verifiable", "length", "contains"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="rollout length cap")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.02, help="KL penalty weight")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--val-eval", type=int, default=200, help="held-out questions used for greedy eval")
    ap.add_argument("--eval-max-new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.group_size >= 2, "need at least 2 answers per question to compare them"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    rows = load_gsm8k("train")
    random.Random(SFT_SPLIT_SEED).shuffle(rows)  # same shuffle and split as sft.py
    val_rows, train_rows = rows[:SFT_VAL], rows[SFT_VAL:]
    val_rows = val_rows[:args.val_eval]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    eos_ids = tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])

    ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    ref = PeftModel.from_pretrained(ref, args.init_adapter).merge_and_unload().eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    policy = PeftModel.from_pretrained(policy, args.init_adapter, is_trainable=True)
    policy.eval()  # no LoRA dropout during RL; gradients still flow
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f"loaded in {time.time() - t0:.0f}s; trainable params {sum(p.numel() for p in params):,}; "
          f"{len(train_rows)} train questions, {len(val_rows)} held-out for eval", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "w")

    def log(rec):
        rec["elapsed_s"] = round(time.time() - t0)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    def evaluate(step):
        a = greedy_accuracy(policy, tok, val_rows, args.eval_max_new)
        print(f"step {step} | held-out greedy accuracy {a:.4f} ({len(val_rows)} questions)", flush=True)
        log({"step": step, "val_acc": a})

    evaluate(0)

    order, ptr = list(range(len(train_rows))), 0
    rng.shuffle(order)
    B, G = args.prompts_per_step, args.group_size
    for step in range(1, args.steps + 1):
        if ptr + B > len(order):
            rng.shuffle(order)
            ptr = 0
        batch = [train_rows[i] for i in order[ptr:ptr + B]]
        ptr += B

        # 1. rollouts
        with torch.no_grad():
            enc = encode_prompts(tok, batch)
            gen = policy.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                  temperature=args.temperature, top_p=1.0, top_k=0,
                                  repetition_penalty=1.0, num_return_sequences=G,
                                  pad_token_id=tok.pad_token_id, use_cache=True)
        P = enc["input_ids"].shape[1]
        comp = gen[:, P:]
        cmask, finished = completion_mask(comp, eos_ids)
        attn = torch.cat([enc["attention_mask"].repeat_interleave(G, dim=0), cmask.long()], dim=1)
        n_comp = cmask.sum(1)
        N = gen.shape[0]

        # 2. rewards
        texts = [tok.decode(comp[i, :n_comp[i]], skip_special_tokens=True) for i in range(N)]
        golds = [r["gold"] for r in batch for _ in range(G)]
        rewards = torch.tensor([
            compute_reward(args.reward, texts[i], golds[i], int(n_comp[i]), bool(finished[i]),
                           args.max_new_tokens) for i in range(N)], device="cuda")
        true_acc = sum(correctness_reward(texts[i], golds[i]) for i in range(N)) / N
        R = rewards.view(B, G)
        mixed = (R.std(1) > 0).float().mean().item()  # groups that actually teach something

        # 3. advantages
        adv = group_advantages(R)

        # 4. policy-gradient step with KL penalty
        total_tokens = cmask.sum().item()
        opt.zero_grad(set_to_none=True)
        kl_sum, loss_sum = 0.0, 0.0
        for s in range(0, N, args.micro_batch):
            sl = slice(s, s + args.micro_batch)
            C = int(n_comp[sl].max())
            ids, a_mask = gen[sl, :P + C], attn[sl, :P + C]
            m = cmask[sl, :C].float()
            lp = token_logprobs(policy, ids, a_mask, P)
            with torch.no_grad():
                rlp = token_logprobs(ref, ids, a_mask, P)
            ratio = torch.exp(lp - lp.detach())  # equals 1 in value; its gradient is grad(logprob)
            d = rlp - lp
            kl = torch.exp(d) - d - 1  # per-token KL estimate (k3)
            per_tok = -(adv[sl].unsqueeze(1) * ratio) + args.beta * kl
            loss = (per_tok * m).sum() / total_tokens
            loss.backward()
            loss_sum += loss.item()
            kl_sum += (kl.detach() * m).sum().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        opt.step()

        rec = {"step": step, "reward": rewards.mean().item(), "true_acc": true_acc,
               "frac_mixed_groups": mixed, "mean_len": n_comp.float().mean().item(),
               "frac_truncated": 1 - finished.float().mean().item(),
               "kl": kl_sum / total_tokens, "loss": loss_sum, "grad_norm": grad_norm, "lr": lr}
        print(f"step {step}/{args.steps} | reward {rec['reward']:.3f} | acc {true_acc:.3f} | "
              f"mixed {mixed:.2f} | len {rec['mean_len']:.0f} | trunc {rec['frac_truncated']:.2f} | "
              f"kl {rec['kl']:.4f} | gnorm {grad_norm:.2f} | {time.time() - t0:.0f}s", flush=True)
        log(rec)

        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            policy.save_pretrained(os.path.join(args.out, f"step_{step}"))

    policy.save_pretrained(args.out)
    with open(os.path.join(args.out, "grpo_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    print(f"saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
