#!/bin/bash

#SBATCH --job-name=baseline
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=51G
#SBATCH -x ariel-k[1,2],ariel-m1

hostname

source ./scripts/_setup.sh


# python train.py

# python train.py --config-name train_fast

# python train.py --config-name train_fast \
#     +experiment=repair_baseline augment=affine


python train.py --config-name train_fast batch_size=4 optim.lr_scheduler.warmup_iter=100
