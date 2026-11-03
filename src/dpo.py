"""DPO on self-generated preference pairs, starting from the SFT adapter (plain PyTorch loop).

loss = -log sigmoid( beta * [ (log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected)) ] )
where pi is the model being trained and ref is the frozen SFT model. Log-probabilities are summed over the
answer tokens of each sequence.

Examples:
  python src/dpo.py --pairs data/prefs_smoke.jsonl --epochs 1 --batch-pairs 4 --micro-pairs 2 \
                    --val-questions 2 --val-eval 16 --out outputs/dpo_smoke       # quick test
  python src/dpo.py                                                               # full run
Then evaluate:  sbatch jobs/eval.sbatch --adapter outputs/dpo --max-new-tokens 512 --out results/dpo.json
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_NAME, load_gsm8k
from grpo import SFT_SPLIT_SEED, SFT_VAL, greedy_accuracy
from sft import collate, encode, lr_factor, to_cuda


def seq_logprobs(model, ids, labels, mask):
    """Sum of log-probabilities of the answer tokens (labels != -100) for each sequence."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    lg = logits[:, :-1, :].float()
    tgt = labels[:, 1:]
    valid = tgt != -100
    tok_lp = lg.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1)
    return (tok_lp * valid).sum(1)


def dpo_terms(pi_c, pi_r, ref_c, ref_r, beta):
    chosen_reward = beta * (pi_c - ref_c)
    rejected_reward = beta * (pi_r - ref_r)
    margin = chosen_reward - rejected_reward
    return -F.logsigmoid(margin), chosen_reward, rejected_reward


@torch.no_grad()
def reference_logps(model, pairs, pad_id, bs=8):
    out = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        ids, labels, mask = to_cuda(collate([p[0] for p in chunk] + [p[1] for p in chunk], pad_id))
        lp = seq_logprobs(model, ids, labels, mask)
        m = len(chunk)
        out += list(zip(lp[:m].tolist(), lp[m:].tolist()))
    return out


@torch.no_grad()
def pair_metrics(model, pairs, ref, pad_id, beta, bs=8):
    loss = acc = margin = 0.0
    for i in range(0, len(pairs), bs):
        chunk, rchunk = pairs[i:i + bs], ref[i:i + bs]
        ids, labels, mask = to_cuda(collate([p[0] for p in chunk] + [p[1] for p in chunk], pad_id))
        lp = seq_logprobs(model, ids, labels, mask)
        m = len(chunk)
        rc = torch.tensor([r[0] for r in rchunk], device="cuda")
        rr = torch.tensor([r[1] for r in rchunk], device="cuda")
        l, cr, rj = dpo_terms(lp[:m], lp[m:], rc, rr, beta)
        loss += l.sum().item()
        acc += ((cr - rj) > 0).float().sum().item()
        margin += (cr - rj).sum().item()
    n = len(pairs)
    return loss / n, acc / n, margin / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--init-adapter", default="outputs/sft")
    ap.add_argument("--pairs", default="data/prefs.jsonl")
    ap.add_argument("--out", default="outputs/dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-pairs", type=int, default=16, help="pairs per optimizer step")
    ap.add_argument("--micro-pairs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--val-questions", type=int, default=50, help="questions whose pairs are held out")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--val-eval", type=int, default=200, help="held-out questions for greedy accuracy")
    ap.add_argument("--eval-max-new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    with open(args.pairs) as f:
        raw = [json.loads(line) for line in f]
    questions = sorted({r["question"] for r in raw})
    rng.shuffle(questions)
    val_q = set(questions[:args.val_questions])  # split by question so no question is in both sets

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id
    train, val = [], []
    for r in raw:
        c = encode(tok, {"question": r["question"], "solution": r["chosen"]}, args.max_len)
        j = encode(tok, {"question": r["question"], "solution": r["rejected"]}, args.max_len)
        if c is None or j is None:
            continue
        (val if r["question"] in val_q else train).append((c, j))
    assert len(train) >= 2, f"only {len(train)} usable training pairs; make more pairs first"
    print(f"{len(train)} train pairs, {len(val)} held-out pairs (dropped "
          f"{len(raw) - len(train) - len(val)} too long)", flush=True)

    base_kwargs = dict(dtype=torch.bfloat16)
    ref = AutoModelForCausalLM.from_pretrained(args.model, **base_kwargs).to("cuda")
    ref = PeftModel.from_pretrained(ref, args.init_adapter).merge_and_unload().eval()
    ref_train = reference_logps(ref, train, pad_id)
    ref_val = reference_logps(ref, val, pad_id) if val else []
    del ref
    torch.cuda.empty_cache()
    print(f"reference log-probs computed ({time.time() - t0:.0f}s)", flush=True)

    policy = AutoModelForCausalLM.from_pretrained(args.model, **base_kwargs).to("cuda")
    policy = PeftModel.from_pretrained(policy, args.init_adapter, is_trainable=True)
    policy.eval()  # no LoRA dropout; gradients still flow
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    rows = load_gsm8k("train")
    random.Random(SFT_SPLIT_SEED).shuffle(rows)
    val_rows = rows[:SFT_VAL][:args.val_eval]
    eval_tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "w")

    def log(rec):
        rec["elapsed_s"] = round(time.time() - t0)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    def evaluate(step):
        rec = {"step": step}
        if val:
            rec["val_loss"], rec["val_pair_acc"], rec["val_margin"] = pair_metrics(
                policy, val, ref_val, pad_id, args.beta)
        rec["heldout_greedy_acc"] = greedy_accuracy(policy, eval_tok, val_rows, args.eval_max_new)
        print("step {} | ".format(step) + " | ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "step"),
              flush=True)
        log(rec)

    steps_per_epoch = -(-len(train) // args.batch_pairs)
    total = steps_per_epoch * args.epochs
    print(f"{total} optimizer steps ({steps_per_epoch}/epoch), {args.batch_pairs} pairs per step", flush=True)
    evaluate(0)

    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for g in range(0, len(order), args.batch_pairs):
            group = order[g:g + args.batch_pairs]
            n = len(group)
            opt.zero_grad(set_to_none=True)
            loss_sum = acc_sum = cr_sum = rj_sum = 0.0
            for s in range(0, n, args.micro_pairs):
                idxs = group[s:s + args.micro_pairs]
                m = len(idxs)
                ids, labels, mask = to_cuda(collate(
                    [train[i][0] for i in idxs] + [train[i][1] for i in idxs], pad_id))
                lp = seq_logprobs(policy, ids, labels, mask)
                rc = torch.tensor([ref_train[i][0] for i in idxs], device="cuda")
                rr = torch.tensor([ref_train[i][1] for i in idxs], device="cuda")
                loss_vec, cr, rj = dpo_terms(lp[:m], lp[m:], rc, rr, args.beta)
                (loss_vec.sum() / n).backward()
                loss_sum += loss_vec.sum().item()
                acc_sum += ((cr - rj) > 0).float().sum().item()
                cr_sum += cr.sum().item()
                rj_sum += rj.sum().item()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            lr = args.lr * lr_factor(step, total, args.warmup)
            for g_ in opt.param_groups:
                g_["lr"] = lr
            opt.step()
            step += 1

            rec = {"step": step, "epoch": epoch + 1, "loss": loss_sum / n, "pair_acc": acc_sum / n,
                   "chosen_reward": cr_sum / n, "rejected_reward": rj_sum / n,
                   "grad_norm": grad_norm, "lr": lr}
            print(f"step {step}/{total} | epoch {epoch + 1} | loss {rec['loss']:.4f} | "
                  f"pair acc {rec['pair_acc']:.2f} | chosen {rec['chosen_reward']:+.3f} | "
                  f"rejected {rec['rejected_reward']:+.3f} | gnorm {grad_norm:.2f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
            log(rec)
            if step % args.eval_every == 0 and step != total:
                evaluate(step)

    evaluate(step)
    policy.save_pretrained(args.out)
    with open(os.path.join(args.out, "dpo_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    print(f"saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
