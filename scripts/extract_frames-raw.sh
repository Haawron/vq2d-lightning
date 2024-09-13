#!/bin/bash

#SBATCH --job-name=extract-frames-raw
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=50G
#SBATCH -w ariel-k2

hostname

python -Bm ltvu.preprocess --splits 'val' --short-side 0
