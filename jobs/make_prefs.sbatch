#!/bin/bash
#SBATCH --job-name=prefs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Usage:  sbatch jobs/make_prefs.sbatch --n-questions 32 --group-size 4 --prompts-per-batch 8 --out data/prefs_smoke.jsonl
#         sbatch jobs/make_prefs.sbatch                                  (full run, samples from outputs/sft)
cd $SCRATCH/rl-night
source env.sh
python src/make_prefs.py "$@"
