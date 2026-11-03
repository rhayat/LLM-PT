# Post-training a small LLM on GSM8K: SFT, DPO and GRPO from scratch

A small, fully reproducible post-training study on **Qwen2.5-0.5B-Instruct** and **GSM8K**. It compares three post-training methods under one evaluation protocol, with every stage written as a short, readable PyTorch loop (no training-framework black boxes):

| Stage | What it does | Feedback signal |
|---|---|---|
| **SFT** | LoRA fine-tuning on GSM8K worked solutions | human-written solutions |
| **DPO** | Preference optimization on correct-vs-incorrect answer pairs the SFT model wrote itself | answer correctness (program-checked) |
| **GRPO** | Online RL: sample a group of answers per question, reward the better-than-average ones | verifiable reward (final answer correct) |

Everything ran on a **single NVIDIA A40** (TAMU HPRC, FASTER cluster). A full pipeline (SFT, GRPO, pair generation, DPO and all evaluations) takes roughly two hours of GPU time.

> **Terminology.** No humans label anything here. The GRPO stage is *reinforcement learning from verifiable rewards*, and the DPO stage is *preference optimization with preferences derived from correctness*. This is not RLHF from human feedback.

---

## Results

All models were evaluated on the **full GSM8K test set (1,319 questions)** with the same prompt, greedy decoding, a 512-token cap and no repetition penalty. The test set was never used for training or tuning.

| Model | Accuracy (95% CI) | Valid `#### n` answer line | Mean answer length (tokens) |
|---|---|---|---|
| Base model (no training) | 39.9% (37.3 to 42.5) | 38.2% | 261 |
| + SFT | 36.3% (33.8 to 38.9) | 98.7% | 95 |
| + SFT, then DPO | 29.2% (26.8 to 31.7) | 96.8% | 133 |
| + SFT, then GRPO (step 100) | 40.5% (37.9 to 43.2) | 99.4% | 92 |
| **+ SFT, then GRPO (step 150, final)** | **41.9%** (39.2 to 44.5) | **99.8%** | 93 |

Paired comparisons on the same 1,319 questions (exact two-sided McNemar test):

| Comparison | Difference | Questions A-only / B-only right | p |
|---|---|---|---|
| GRPO final vs SFT | **+5.5 points** | 166 / 93 | 0.000007 |
| GRPO final vs base | +2.0 points | 232 / 206 | 0.23 (not significant) |
| GRPO step 100 vs GRPO final | -1.4 points | 83 / 101 | 0.21 (not significant) |
| SFT vs base | -3.6 points | 189 / 236 | 0.026 |
| DPO vs SFT | -7.1 points | 109 / 203 | 0.0000001 |
| GRPO final vs DPO | +12.7 points | 251 / 84 | 2e-20 |

### How the outputs differ (test set)

| Model | Valid `#### n` | Answer-first outputs | Hit the 512-token cap | Median / 90th percentile tokens | Outputs with a line repeated 3+ times |
|---|---|---|---|---|---|
| Base | 38.2% | 199 | 65 | 261 / 412 | 32 |
| SFT | 98.7% | 0 | 15 | 82 / 151 | 10 |
| SFT + DPO | 96.8% | 0 | 40 | 111 / 219 | 18 |
| SFT + GRPO (step 100) | 99.4% | 0 | 9 | 82 / 145 | 4 |
| SFT + GRPO (final) | 99.8% | 0 | 2 | 84 / 146 | 1 |

"Answer-first" means the output begins with `#### n` before any reasoning.

### Prompt sensitivity (base model, same decoding)

| | First prompt | Final prompt |
|---|---|---|
| Accuracy | 13.8% | **39.9%** |
| Answer-first outputs | 90% | 15% |
| Median answer length | 19 tokens | 261 tokens |
| Output contains `####` | 95.2% (mostly bare guesses, often copying the literal `<number>` placeholder) | 38.2% |

The first prompt was *"Solve the math problem step by step. Finish with the final numeric answer on its own line in the form '#### \<number\>'."* The final prompt is given under [Method](#task-and-evaluation-protocol). When the base model did reason first under the first prompt, it scored 42.1% (126 outputs), so the low number measured behavior, not ability.

### Training curves in numbers

**SFT** validation loss (200 held-out questions): 0.677 at the start, 0.523 at step 100, 0.514 at step 200 (end of epoch 1), then flat at about 0.515 through step 456.

**GRPO**, averaged over blocks of steps (training columns are from sampled answers at temperature 1.0 on training questions):

| Steps | Mean reward | Sampled accuracy | Groups with mixed outcomes | Mean length (tokens) | Truncated | KL to SFT |
|---|---|---|---|---|---|---|
| 1 to 10 | 0.453 | 35.4% | 0.78 | 100 | 1.3% | 0.0019 |
| 11 to 25 | 0.501 | 40.3% | 0.78 | 92 | 0.9% | 0.0046 |
| 26 to 50 | 0.495 | 39.6% | 0.82 | 94 | 0.8% | 0.0085 |
| 51 to 75 | 0.539 | 44.0% | 0.70 | 89 | 0.6% | 0.0144 |
| 76 to 100 | 0.563 | 46.4% | 0.70 | 92 | 0.6% | 0.0230 |
| 101 to 125 | 0.570 | 47.1% | 0.70 | 92 | 0.4% | 0.0268 |
| 126 to 150 | 0.583 | 48.4% | 0.72 | 92 | 0.6% | 0.0294 |

Held-out greedy accuracy (200 questions, plus or minus about 3.5 points) at steps 0, 25, 50, 75, 100, 125, 150: 0.485, 0.525, 0.525, 0.550, 0.560, 0.515, 0.555.

**DPO** (294 optimizer steps over 2 epochs):

| Step | Held-out pair loss | Held-out pair accuracy | Held-out greedy accuracy |
|---|---|---|---|
| 0 | 0.689 | 0.63 | 0.485 |
| 50 | 0.662 | 0.58 | 0.525 |
| 100 | 0.552 | 0.72 | 0.485 |
| 150 | 0.520 | 0.70 | 0.445 |
| 200 | 0.489 | 0.78 | 0.420 |
| 250 | 0.492 | 0.78 | 0.410 |
| 294 | 0.490 | 0.74 | 0.435 |

The preference objective kept improving while greedy accuracy on the held-out questions declined after step 50.

### The `results/` folder

Each evaluation has a scored file (`<name>.json`: per-question outputs, predictions and a summary) and the raw generations (`<name>.raw.json`). `base.json` is the base model under the *first* prompt; `base1.json` is the base-model baseline under the final prompt (`baseline.json` is an identical repeat of it); `sft.json`, `dpo.json`, `grpo.json` (step 150) and `grpo_step100.json` are the trained models.

### Observations

1. **Prompt wording was the biggest lever, bigger than any training stage.** With the same model, changing the system prompt so that the model must *finish reasoning before writing the answer line* raised accuracy from **13.8% to 39.9%**. With the first prompt, 90% of outputs were answer-first guesses (a median of 19 tokens) at 10.8% accuracy; with the reasoning-first prompt that fell to 15%.
2. **SFT fixed structure, not reasoning.** Valid answer lines went from 38% to 98.7%, truncation from 65 cases to 15, and answers became 2.7 times shorter. Accuracy dropped slightly (-3.6 points, p = 0.026). Validation loss was flat after the first epoch.
3. **GRPO is the only stage that improved accuracy on top of its starting point** (+5.5 points over SFT, 166 questions flipped to right and 93 to wrong). It did so *without making answers longer* (93 tokens versus SFT's 95), and nearly eliminated truncation (15 cases down to 2).
4. **GRPO's gain over the untrained base is small and not statistically significant** (+2.0 points, p = 0.23). The honest summary is that GRPO recovered the accuracy SFT lost and ended slightly above the base, with a clean answer format and short outputs.
5. **GRPO training was stable.** Over 150 steps (about 43 minutes): sampled training accuracy rose from 35% to 48%, the KL divergence from the SFT reference grew slowly from 0.002 to 0.029 (maximum 0.039), mean answer length stayed at 90 to 100 tokens, gradient norms were flat around 0.35, and 70 to 82% of question groups contained a mix of right and wrong answers (so there was always learning signal). Held-out greedy accuracy on 200 held-out questions went 0.485, 0.525, 0.525, 0.55, 0.56, 0.515, 0.555 at steps 0, 25, ..., 150, so it plateaued around step 75 to 100 within the roughly plus or minus 3.5 point noise of 200 questions.
6. **DPO, in this configuration, hurt.** It optimized its own objective (held-out pair accuracy rose from 0.63 to between 0.70 and 0.80) while greedy accuracy on the held-out questions peaked early (0.525 at step 50) and then fell to 0.435. Both the chosen and the rejected answers' log-probability ratios drifted negative (about -1.1 and -3.3 at the end), a pattern consistent with *likelihood displacement*; its cause was not isolated here. The 2,423 training pairs came from 2,000 questions (72% of questions had a mix of right and wrong answers); chosen answers were slightly shorter than rejected ones (84 versus 91 tokens), so length bias in the data does not explain the longer outputs. Treat this as a result for one untuned configuration (beta 0.1, learning rate 2e-5, 2 epochs), not a verdict on DPO.
7. **Hidden defaults matter.** Qwen2.5's `generation_config.json` ships `repetition_penalty=1.1` and sampling defaults that stay active during "greedy" decoding unless overridden. All evaluation and rollout code sets them explicitly.

### Caveats

- One training run (seed 0) per method, so each number is a single draw from its procedure. A second seed would show how stable the GRPO gain is.
- Accuracy numbers during training come from 200 held-out *training-split* questions, which are easier than the test set (the SFT model scores 48.5% on them versus 36.3% on the test set). Held-out and test numbers must not be compared with each other.
- Scoring takes the last `#### n` in the output, falling back to the last number in the text. The fallback matters only for the base model (62% of its outputs lack the marker); its error rate was not measured.
- Plain Hugging Face `generate` is used for rollouts, so the setup is not throughput-optimized. A rollout engine such as vLLM would speed up GRPO and pair generation considerably.

---

## Method

### Task and evaluation protocol

- **Data:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (grade-school math word problems). Training uses the train split minus **200 held-out questions** (fixed shuffle, seed 0) that are used only for monitoring. The 1,319-question test split is used only for final evaluation.
- **Prompt:** the same system prompt everywhere (SFT targets, rollouts, evaluation): *"Reason step by step, showing your calculations, before giving the answer. Only after you have finished reasoning, write the final numeric answer on its own last line in the form '#### \<number\>'."* (see `src/common.py`).
- **Scoring:** an answer is correct if the last `#### n` (else the last number) equals the gold answer. Outputs that overflow to infinity (degenerate digit strings) score as wrong.
- **Decoding:** greedy, 512 new tokens, `repetition_penalty=1.0`, temperature, top-p and top-k cleared.

### Stage 1: SFT (`src/sft.py`)

LoRA (rank 16, alpha 32, dropout 0.05, all attention and MLP projections; 8.8M trainable parameters, 1.75% of the model). Targets are GSM8K's worked solutions with the `<<48/2=24>>` calculator annotations stripped, ending in `#### n` plus Qwen's `<|im_end|>` token so the model learns to stop. Loss is computed on answer tokens only. Learning rate 2e-4 (3% warmup, cosine decay), effective batch 32, 2 epochs (456 steps, about 8 minutes). Validation loss 0.677 to 0.514.

### Stage 2: GRPO (`src/grpo.py`)

Starts from the SFT adapter; the KL reference is the frozen SFT model.

1. Sample 16 questions and 8 answers per question (temperature 1.0, no top-k or top-p filtering, 256-token rollout cap).
2. Reward = 1.0 if the final number is correct, plus 0.1 for a valid `#### n` line; answers cut off at the cap get 0.
3. Advantage = (reward - group mean) / group standard deviation. A group in which every answer scores the same teaches nothing and gets advantage 0.
4. One policy-gradient step per rollout batch, with a per-token KL penalty (weight 0.02). Learning rate 2e-5, 150 steps. Greedy accuracy on the held-out questions is logged every 25 steps and a checkpoint is saved at each.

Differences from PPO: there is no learned critic (the group average is the baseline), and with one update per rollout batch no importance-ratio clipping is needed.

The script also contains two deliberately flawed rewards (`--reward length`, `--reward contains`) for studying reward hacking; those experiments have not been run yet.

### Stage 3: DPO (`src/make_prefs.py`, `src/dpo.py`)

`make_prefs.py` has the SFT model sample 8 answers (temperature 1.0) for each of 2,000 training questions, grades them, and keeps up to 2 pairs per question: a **chosen** answer (correct, finished, valid `#### n` line) and a **rejected** answer (incorrect, finished). The sampling statistics are in `data/prefs_stats.json` (sample accuracy 33.7%, 2,423 pairs).

`dpo.py` minimizes `-log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))])`, where `pi` is the policy (the SFT adapter, kept trainable) and `ref` is the frozen SFT model. Beta 0.1, learning rate 2e-5, 2 epochs, 16 pairs per step.

---

## Repository layout

```
src/
  common.py       prompt, GSM8K loading, answer extraction, rewards
  eval.py         greedy evaluation of a model (optionally with a LoRA adapter)
  sft.py          LoRA SFT loop
  grpo.py         GRPO loop (reward modes: verifiable, length, contains)
  make_prefs.py   builds DPO pairs from the SFT model's own samples
  dpo.py          DPO loop
jobs/             Slurm job scripts (one per stage)
env.sh            environment for jobs on the TAMU HPRC FASTER cluster (edit for your system)
results/          evaluation outputs: <name>.json (scored) and <name>.raw.json (raw generations)
outputs/          training logs (train_log.jsonl) and run settings for every stage; LoRA adapters are not tracked in git
data/             DPO preference pairs and their statistics
logs/             Slurm logs of the runs
docs/             setup and troubleshooting notes for the FASTER cluster environment
```

---

## Reproducing it

### Requirements

- Linux, Python 3.12, one NVIDIA GPU. All runs fit on a 48 GB A40; on a smaller GPU, lower `--micro-batch` (SFT and GRPO), `--micro-pairs` (DPO) or `--prompts-per-batch` (pair generation).
- Versions the results were produced with: `torch==2.7.1` (CUDA 12.6), `numpy==1.26.4`, `transformers==5.22.1`, `peft==0.21.2`, `datasets==4.8.5`, `pyarrow==21.0.0`, `huggingface-hub==1.33.0`. Other combinations are untested. (`trl` and `accelerate` were installed in the original environment but are not used by any script.)

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.7.1
pip install transformers==5.22.1 peft==0.21.2 datasets==4.8.5 pyarrow==21.0.0 huggingface-hub==1.33.0
```

The model and dataset are **not** included in this repository; they download from the Hugging Face Hub on first use. To fetch them ahead of time (useful on clusters whose compute nodes have no internet):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"
```

### Run order

Run everything from the repository root. The scripts import each other from `src/`, so call them as `python src/<script>.py`.

```bash
# 1. baseline evaluation (about 5 minutes)
python src/eval.py --max-new-tokens 512 --out results/base.json

# 2. SFT (about 8 minutes), then evaluate it
python src/sft.py
python src/eval.py --adapter outputs/sft --max-new-tokens 512 --out results/sft.json

# 3. GRPO from the SFT adapter (about 45 minutes), then evaluate the final adapter and a checkpoint
python src/grpo.py
python src/eval.py --adapter outputs/grpo --max-new-tokens 512 --out results/grpo.json
python src/eval.py --adapter outputs/grpo/step_100 --max-new-tokens 512 --out results/grpo_step100.json

# 4. DPO: build the pairs (about 13 minutes), train (about 15 minutes), evaluate
python src/make_prefs.py
python src/dpo.py
python src/eval.py --adapter outputs/dpo --max-new-tokens 512 --out results/dpo.json
```

Every script has a quick smoke-test mode, which is worth running first. For example:

```bash
python src/sft.py --limit 64 --epochs 1 --val 16 --out outputs/sft_smoke
python src/grpo.py --steps 2 --prompts-per-step 4 --group-size 4 --eval-every 2 --val-eval 16 --out outputs/grpo_smoke
python src/make_prefs.py --n-questions 32 --group-size 4 --prompts-per-batch 8 --out data/prefs_smoke.jsonl
python src/dpo.py --pairs data/prefs_smoke.jsonl --epochs 1 --batch-pairs 4 --micro-pairs 2 --val-questions 2 --val-eval 16 --out outputs/dpo_smoke
```

Timings above are from a single A40 and are approximate. Each run writes `train_log.jsonl` and an `*_args.json` with the exact settings into its output folder.

### Running on Slurm (TAMU HPRC FASTER)

The `jobs/*.sbatch` files and `env.sh` are written for FASTER (partition `gpu`, an A40 node, `$SCRATCH`). They source `env.sh`, which loads the cluster's modules, activates the venv and sets offline Hugging Face flags (so the model and dataset must be downloaded beforehand, on the login node). To use them elsewhere, edit the module lines, the paths and the GPU request. Example:

```bash
sbatch jobs/sft.sbatch
sbatch jobs/eval.sbatch --adapter outputs/sft --max-new-tokens 512 --out results/sft.json
sbatch --dependency=afterok:<make_prefs job id> jobs/dpo.sbatch     # DPO needs the pairs file first
```

The FASTER environment is built on top of the cluster's PyTorch module (`GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.7.1-CUDA-12.6.0`) with a venv created with `--system-site-packages`. That combination has several pitfalls (the module's `PYTHONPATH` shadowing newer pip packages, NumPy 1.26 versus newer `pyarrow`, and others), all documented with their fixes in [`docs/setup_and_troubleshooting.md`](docs/setup_and_troubleshooting.md). `requirements-lock.txt` is a raw `pip freeze` of that cluster environment and contains cluster-specific paths, so it is not portable.

---

## References

- Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K), 2021.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Rafailov et al., *Direct Preference Optimization*, 2023.
- Shao et al., *DeepSeekMath* (introduces GRPO), 2024.
- Qwen Team, *Qwen2.5 Technical Report*, 2024.
