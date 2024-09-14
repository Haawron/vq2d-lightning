#!/bin/bash

#SBATCH --job-name=lit-train
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G

hostname

source ./scripts/_setup.sh


python run.py dataset.batch_size=3 compile=false \
    backbone_precision=fp32 backbone_fp32_mm_precision=highest \
    dataset.prefetch_factor=4 dataset.num_workers=8

# python run.py dataset.batch_size=3 \
    # dataset.prefetch_factor=4 dataset.num_workers=8
