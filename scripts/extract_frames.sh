#!/bin/bash

#SBATCH --job-name=extract-frames-320ss
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=35G
#SBATCH -x ariel-v[3,6],ariel-k[1,2],ariel-m1

hostname

python -Bm ltvu.preprocess --short-side 320
