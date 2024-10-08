#!/bin/bash

#SBATCH --job-name=eval_rt-pos-query_mode_both
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=35G
#SBATCH -w ariel-v11


hostname

source ./scripts/_setup.sh

# python eval.py \
#     ckpt='outputs/batch/2024-09-15/124034/epoch\=80-prob_acc\=0.7859.ckpt'
python eval.py --config-name eval_fast \
    ckpt='/data/soyeonhong/vq2d/vq2d-lightning/outputs/batch/2024-10-07/129039/epoch\=49-prob_acc\=0.7889.ckpt'
