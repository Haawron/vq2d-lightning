#!/bin/bash

#SBATCH --job-name=vqloc-448-lit-fullpower
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH -x aurora-g7

hostname

source ./scripts/_setup.sh


python run.py
