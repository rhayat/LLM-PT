#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out

# Usage:  sbatch jobs/eval.sbatch --out results/base.json
#         sbatch jobs/eval.sbatch --adapter outputs/sft --out results/sft.json
cd $SCRATCH/rl-night
source env.sh
python src/eval.py "$@"
