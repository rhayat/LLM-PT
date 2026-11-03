# Post-Training Project: Notes and Reference

Written 2026-11-03. Covers the goal, the analysis behind it, the plan, the FASTER (TAMU HPRC) environment setup, and every problem hit with its fix.

---

## 1. Goal

Build a small post-training (RLHF-style) project to close the main gap in the resume for two AMD Summer 2027 PhD internships. The project had to be small enough to run in one night.

### Target jobs (both: Santa Clara, onsite, no visa sponsorship, $91.5K–$137K/yr)

| | **90910: Gen AI and RL Research Intern** | **90950: AI Research Infrastructure, RL Post-Training Intern** |
|---|---|---|
| Focus | Research: algorithms and experiments | Systems: training stack at scale |
| Core asks | RL fundamentals, PyTorch, RLHF/RLAIF/preference optimization, reproducible experiments | RL post-training, distributed training (DP/TP/PP), rollout/reward pipelines, GPU profiling |
| Day-to-day | Reward modeling, policy optimization, credit assignment, reward hacking and instability analysis, evals | Rollout generation, inference, reward computation, policy updates, fault tolerance, checkpointing |
| Preferred | ICML/NeurIPS/ICLR/CVPR papers | Same, plus GPU/cluster experience |

90910 is the better match. 90950 needs a "scaling" angle added on top.

### Resume gap analysis

**Strengths:** current PhD student (TAMU, Jun 2028), strong Python/PyTorch, MS in Computer Engineering, published work, real LLM/VLM pipelines (multimodal RAG, Reflexion-style MCP agent, AWQ quantization of a 15B VLM), CUDA fundamentals.

**Gaps:**
1. No RLHF, RL or preference-optimization work. This is the core requirement of both roles.
2. No distributed or multi-GPU training.
3. Publications are in construction/engineering informatics, not ML venues.
4. LLMs have only been used for inference, never trained or post-trained.

---

## 2. The project plan (one night, small model)

**Model:** Qwen2.5-0.5B-Instruct with LoRA, using Hugging Face TRL. A task with a checkable answer (GSM8K subset).

| Step | Time | What |
|---|---|---|
| 1 | 1 h | Baseline and eval harness on a held-out set |
| 2 | 1 h | SFT warm-start (LoRA) |
| 3 | 1–1.5 h | DPO on correctness-labeled preference pairs |
| 4 | 2–3 h | GRPO with a programmatic reward (correctness plus format bonus) |
| 5 | 1 h (optional) | Small Bradley-Terry reward model, compare to verifiable reward |
| 6 | 1 h | Reward-hacking and failure analysis: use an exploitable reward on purpose (length/format only), show the model games it, then fix it. Log KL, reward curves, length drift |
| 7 | 1 h | README, results table (base / SFT / DPO / GRPO), plots, seeds, configs, W&B logs |

**Deliverables:** GitHub repo, 1–2 page report, eval table with plots, failure-modes section, short "Scaling" section on what changes at 8 GPUs (for 90950).

**Honest caveats:** single-seed gains of a few points on a 0.5B model are noisy, so report that. Expect an hour of library-version trouble. Real multi-GPU work will not fit in one night.

**Resume line once done:** "Post-trained Qwen2.5-0.5B with SFT → DPO → GRPO on verifiable rewards; analyzed reward hacking and KL drift." Also reframe the agent work as "agentic systems with tool use" and the NMPC work as optimization/control for decision-making.

---

## 2b. What it trains on, and is it RLHF?

- **Data:** GSM8K (about 7.5K train / 1.3K test grade-school math word problems, each with a worked solution and a numeric final answer). Correctness is checkable by a program, so no human grading is needed. Qwen reports roughly 50% for this model, so we measure our own baseline.
- **Model:** Qwen2.5-0.5B-Instruct, trained with LoRA (small added weights, fits on one GPU).
- **SFT:** imitate GSM8K worked solutions to learn the output format (steps, then `#### answer`).
- **DPO:** sample several answers per question, grade them by final answer, pair a correct (chosen) with an incorrect (rejected) answer. Preferences come from correctness, not from humans.
- **GRPO:** real RL. Sample a group of answers per question, reward = correct (+ small format bonus). Answers above their group's average are made more likely. A KL penalty keeps the policy near the starting model. No critic model, so it is cheap.
- **Reward hacking experiment:** use a deliberately flawed reward (format or length only), show the model exploiting it (reward, KL, length curves), then fix the reward.

**Is it RLHF?** Strictly no. RLHF = human comparisons → reward model → RL against that reward model. This project is **RL from verifiable rewards (RLVR)** (GRPO) plus **preference optimization** (DPO), with an optional small reward model (RLAIF/RLHF-adjacent). On the resume, write "preference optimization and RL from verifiable rewards", not "RLHF from human feedback". The AMD postings list "verifiable, preference-based, or model-generated feedback", so this matches. Optional closer-to-RLHF track: train a reward model on a real human-preference set (UltraFeedback or Anthropic HH-RLHF), then DPO/GRPO on chat quality. Noisier on a 0.5B model, so only if time remains.

---

## 3. Environment: TAMU HPRC FASTER

### Facts about the cluster (from this session)

- Login node prompt: `<user>@faster2`. A GPU node will show a different hostname, so if `nvidia-smi` fails you are on the login node.
- `$SCRATCH` = `/scratch/user/<user>`. Everything goes on scratch (home quota is small).
- Partition: `gpu`. Walltime limit 7 days. GPU types seen in `sinfo`: A100, A40, A30, A10, T4. Idle at the time: A40, A30, A10. Avoid `drain*` nodes.
- Internet from the login node goes through **WebProxy**, so `module load WebProxy` first.
- Modules are EasyBuild/Lmod and hierarchical. A package may need its toolchain loaded first.
- Existing scratch folders: `llm/`, `ondemand-faster/`, `PIP_CACHE/`. `PIP_CACHE` is reused as the pip cache. `llm/` was an earlier attempt and was left alone.
- Project folder: `$SCRATCH/rl-night`.

### Final working setup (do all of this on the LOGIN node)

```bash
module purge
module load WebProxy
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.7.1-CUDA-12.6.0

export RL=$SCRATCH/rl-night
export HF_HOME=$RL/hf PIP_CACHE_DIR=$SCRATCH/PIP_CACHE TMPDIR=$RL/tmp
mkdir -p $HF_HOME $TMPDIR && cd $RL

python -m venv --system-site-packages venv && source venv/bin/activate

# pin what the module already provides, then preview before installing
pip list --format=freeze | grep -iE "^(torch|numpy|scipy|packaging|pyyaml|filelock|fsspec|jinja2|sympy|networkx|typing[-_]extensions)==" > constraints.txt
pip install --dry-run -c constraints.txt trl peft datasets accelerate transformers
# the dry run must NOT list torch or numpy. If clean:
pip install -c constraints.txt trl peft datasets accelerate transformers

# FIX that was needed afterwards (see problems 9 and 10): the module pins NumPy 1.26.4,
# but the newest pyarrow (26.x) needs NumPy 2, and datasets 5.x needs pyarrow>=24.
# Pin pyarrow and use the datasets 4.x line:
pip install -c constraints.txt "pyarrow==21.0.0" "datasets>=4.0,<5"
# Untested idea for a fresh setup: do this in ONE install so the resolver sees all pins together:
#   pip install -c constraints.txt "pyarrow==21.0.0" "datasets>=4.0,<5" trl peft accelerate transformers
# (the sequence actually run was the two commands above, in that order)

# REQUIRED: venv packages must win over the module's packages
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
```

Versions that resolved: Python 3.12.3, torch 2.7.1 (CUDA 12.6), numpy 1.26.4, trl 1.17.1, peft 0.21.2, transformers 5.22.1, accelerate 1.15.0, **pyarrow 21.0.0, datasets 4.8.5, huggingface-hub 1.33.0** (the last three after the fixes below; the first install had pyarrow 26.0.0, datasets 5.1.0, huggingface-hub 2.4.0).

Pre-download model and data on the login node (compute nodes may not have internet):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
python -c "from datasets import load_dataset; load_dataset('openai/gsm8k','main')"
# (the old `huggingface-cli` command may not exist in huggingface_hub 2.x, so use the Python call)
```

### GPU node (verify and run only, never install)

```bash
srun --partition=gpu --gres=gpu:a40:1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash
# add --account=<acct> if it complains (find it with: myproject)

module purge
module load WebProxy
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.7.1-CUDA-12.6.0
source $SCRATCH/rl-night/venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
export HF_HOME=$SCRATCH/rl-night/hf HF_HUB_OFFLINE=1
nvidia-smi | head -4
python -c "
import torch, numpy, trl, peft, transformers, datasets
print('torch', torch.__version__, '| numpy', numpy.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '| transformers', transformers.__version__)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000, device='cuda'); print((x @ x).sum().item())"
```

Every Slurm script needs the same `module load` line, the venv activation, and the `PYTHONPATH` export. `module purge` clears the module dependencies too.

### Fallback: Path B (self-contained, no module conflicts)

```bash
module purge; module load WebProxy Python/3.12.3   # plus toolchain line from module spider if asked
python -m venv $RL/venv-pip && source $RL/venv-pip/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install trl peft datasets accelerate transformers
```

pip resolves everything in one pass, so nothing from the cluster modules can shadow it. Needs a GPU driver new enough for CUDA 12.6 (the CUDA 12.6 module runs on these nodes, so it should be).

Template Slurm job:

```bash
#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --account=<acct>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

module purge
module load WebProxy
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.7.1-CUDA-12.6.0
source $SCRATCH/rl-night/venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
export HF_HOME=$SCRATCH/rl-night/hf HF_HUB_OFFLINE=1
python train_grpo.py --config configs/grpo.yaml
```

---

## 4. Problems encountered and patches

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | `module load PyTorch/2.7.1-CUDA-12.6.0` failed: "exist but cannot be loaded as requested" | Hierarchical modules: the toolchain must be loaded first | Load the line Lmod suggested: `module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.7.1-CUDA-12.6.0`. Use `module spider <name>/<version>` to see prerequisites |
| 2 | `srun: error: (from job_submit) either --ntasks, or --ntasks-per-node, or --nodes should be provided` | FASTER requires a task count | Add `--ntasks=1` |
| 3 | Same error after adding `--ntasks=1` | The flag was placed after `bash`, so `srun` passed it to bash | Put all options **before** `--pty bash` |
| 4 | `ImportError: regex>=2025.10.22 is required ... found regex==2024.5.15` | The module sets `PYTHONPATH`, which comes before the venv's `site-packages`. The module's old `regex` shadowed the newer one pip installed into the venv | `export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH`. Check with `python -c "import regex; print(regex.__version__, regex.__file__)"`, which should show 2026.9.29 and a path in the venv |
| 5 | `ModuleNotFoundError: No module named 'trl'` | Only the `--dry-run` had been run, which installs nothing | Run the real `pip install` after the dry run |
| 6 | Installed packages while on the GPU node | Setup steps were given in pieces instead of in order | Install and download on the login node only. GPU node is for verification and runs only |
| 7 | `(venv) (venv)` in the prompt | Venv activated twice | Harmless. `deactivate` once to tidy |
| 8 | Unsure whether on a GPU node | Prompt still showed `faster2`, the login-node name | Run `hostname; nvidia-smi -L`. On the login node `nvidia-smi` fails and `torch.cuda.is_available()` is `False`, which is expected there |
| 9 | `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4` when importing `datasets` | pip installed `pyarrow 26.0.0` as a dependency. It needs NumPy 2 at import time, but the PyTorch module ships NumPy 1.26.4. This requirement is not in the package metadata, so neither `--dry-run` nor `pip check` caught it | `pip install -c constraints.txt "pyarrow==21.0.0"`. Do **not** upgrade NumPy: the module's torch, SciPy and pandas are built against NumPy 1.x |
| 10 | After the pyarrow pin: `datasets 5.1.0 requires pyarrow>=24.0.0, but you have pyarrow 21.0.0` | pyarrow was pinned without checking `datasets`' own requirement | Use the datasets 4.x line: `pip install -c constraints.txt "pyarrow==21.0.0" "datasets>=4.0,<5"`. This installed datasets 4.8.5 and **downgraded** huggingface-hub from 2.4.0 to 1.33.0 (plus httpcore 1.0.9, httpx 0.28.1). `pip check` afterwards reported no broken requirements, and `transformers 5.22.1` imports fine with hub 1.33.0 |
| 11 | `sbatch: error: This does not look like a batch script. The first line must start with #!` | The script file on FASTER had mangled line endings or extra bytes (the pasted content showed a blank line after every line), most likely from the terminal paste or the editor. Exact cause not inspected | Upload a clean file via OnDemand, then run `sed -i 's/\r$//; 1s/^\xEF\xBB\xBF//' f.sbatch; sed -i '/^[[:space:]]*$/d' f.sbatch; head -3 f.sbatch \| cat -A`. The first line must be exactly `#!/bin/bash$`. After this the job submitted and ran |
| 12 | No output for minutes while the job ran | Python buffers stdout when writing to a file, so lines appear late or are lost if the job is killed | Use `export PYTHONUNBUFFERED=1` and `python -u`, print timestamps between steps, and follow with `tail -f job-*.out`. Give cold-start jobs a generous `--time` |

### Lessons

- With `--system-site-packages` plus a pip-upgraded package, the module's `PYTHONPATH` entries can shadow the venv. Always put the venv first on `PYTHONPATH`.
- Load only one module stack. Do not load separate `Python`, `CUDA` or `GCC` modules on top of the PyTorch module. Mixing toolchains causes real version conflicts.
- Use `pip install --dry-run -c constraints.txt ...` first. Stop if it wants to install or replace `torch` or `numpy`.
- Timebox path A to about 15 minutes. If it keeps failing, switch to path B instead of fixing one package at a time.
- Order of work: login node for installs and downloads, GPU node for running only.
- Root cause of problems 4, 9 and 10: the PyTorch module is frozen at NumPy 1.26.4 while the newest pip packages increasingly assume NumPy 2. Expect more of these. A pin on one package (pyarrow) can conflict with another package's own pin (datasets), so read pip's "dependency conflicts" message instead of ignoring it.
- `pip check` and `--dry-run` do not catch import-time requirements such as pyarrow's NumPy 2 check. Always run a real import test after any install.
- If the module stack keeps producing conflicts, **Path B** (a second venv with pip's own torch wheel and NumPy 2, in a separate folder such as `$RL/venv-pip`) removes this whole class of problem and leaves the current venv untouched.

---

## 5. Status at time of writing

Done:
- Module stack loads: Python 3.12.3, torch 2.7.1 with CUDA 12.6.
- Dry run was clean (no torch/numpy changes). Packages installed into `$SCRATCH/rl-night/venv`.
- `PYTHONPATH` fix confirmed for `regex`.

- pyarrow pinned to 21.0.0 and datasets downgraded to 4.8.5 (huggingface-hub now 1.33.0). pip reported "Successfully installed".

Confirmed after the datasets 4.x install:
- `pip check` reports "No broken requirements found." (so no `huggingface-hub` complaint from `transformers` or `trl`).
- Full import works. Printed versions: torch 2.7.1, numpy 1.26.4, trl 1.17.1, peft 0.21.2, transformers 5.22.1, datasets 4.8.5, huggingface_hub 1.33.0.

Confirmed by the GPU batch job (2026-11-03, node `fc096`):
- Real GPU node: NVIDIA A40, driver 580.126.20 (CUDA 13.0 runtime on the driver side), `torch.cuda.is_available()` is `True`, GPU matmul works.
- Qwen2.5-0.5B-Instruct loads **offline from the scratch cache** (290 weight tensors in about 4 s) and generates. So the model pre-download worked and `HF_HUB_OFFLINE=1` is fine.
- The raw prompt `2+2=` continued as `4 - solution\n3+3=`. That is just plain text continuation (no chat template), not a problem.
- The environment setup is finished and working end to end.

- `load_dataset('openai/gsm8k', 'main')` loads (user-confirmed; the printed output was not pasted, so the exact split sizes were not seen).

Not yet confirmed:
- Nothing outstanding in the environment. Next: eval harness, then SFT, DPO and GRPO scripts.

## 6. Next steps

1. Finish the verification above on a real GPU node.
2. Pre-download Qwen2.5-0.5B-Instruct and GSM8K on the login node.
3. Write the eval harness and run the baseline.
4. Write SFT, DPO and GRPO scripts plus Slurm files (one short job per stage, each saving a checkpoint).
5. Run the reward-hacking experiment, then write the README and report.
6. Update the resume and add the project line.
