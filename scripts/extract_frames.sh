#!/bin/bash

#SBATCH --job-name=extract-frames-520ss
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=35G

hostname

python -Bm ltvu.preprocess --short-side 520
