#!/bin/bash
#SBATCH --job-name=dpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Usage:  sbatch jobs/dpo.sbatch --pairs data/prefs_smoke.jsonl --epochs 1 --batch-pairs 4 --micro-pairs 2 --val-questions 2 --val-eval 16 --out outputs/dpo_smoke
#         sbatch jobs/dpo.sbatch                                         (full run, starts from outputs/sft)
cd $SCRATCH/rl-night
source env.sh
python src/dpo.py "$@"
