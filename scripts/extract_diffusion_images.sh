#!/bin/bash

#SBATCH --job-name=diffusion
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --array=0-15

hostname

python notebooks/generate_diffusion_data.py --world_size 16 --rank $SLURM_ARRAY_TASK_ID --prompt_type floor
