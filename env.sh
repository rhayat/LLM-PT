#!/bin/bash
# Source this in every job: `source $SCRATCH/rl-night/env.sh`
# Never install or download anything from a job. Run only.
module purge
module load WebProxy
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.7.1-CUDA-12.6.0
source $SCRATCH/rl-night/venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
export HF_HOME=$SCRATCH/rl-night/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
