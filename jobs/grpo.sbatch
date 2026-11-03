#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# Usage:  sbatch jobs/grpo.sbatch --steps 2 --prompts-per-step 4 --group-size 4 --eval-every 2 --val-eval 16 --out outputs/grpo_smoke
#         sbatch jobs/grpo.sbatch                                   (full run, starts from outputs/sft)
cd $SCRATCH/rl-night
source env.sh
python src/grpo.py "$@"
