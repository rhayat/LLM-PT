#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Usage:  sbatch jobs/sft.sbatch --limit 64 --epochs 1 --val 16 --out outputs/sft_smoke   (smoke test)
#         sbatch jobs/sft.sbatch                                                          (full run)
cd $SCRATCH/rl-night
source env.sh
python src/sft.py "$@"
