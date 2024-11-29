#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=35G


hostname

# source ./scripts/_setup.sh

python eval.py \
    ckpt=CKPTPATH
