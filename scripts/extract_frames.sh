#!/bin/bash

#SBATCH --job-name=extract-egotracks-frames-320ss
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=35G

hostname

# python -Bm ltvu.preprocess --short-side 320
python -m ltvu.preprocess --task egotracks --split train --whole --world_size 2 --rank 1 --short-side 320
