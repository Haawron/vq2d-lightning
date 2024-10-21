#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=35G
#SBATCH -x ariel-v[3,6],ariel-k[1,2],ariel-m1


hostname

source ./scripts/_setup.sh

# python eval.py \
#     ckpt='outputs/batch/2024-09-15/124034/epoch\=80-prob_acc\=0.7859.ckpt'
python eval.py \
    ckpt='outputs/ckpts/nvfch3cs-epoch\=61-prob_acc\=0.7739.ckpt'
    # test_submit=True \
