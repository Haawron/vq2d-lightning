#!/bin/bash

#SBATCH --job-name=lit-train-comp
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH -x ariel-v[3,6],ariel-k[1,2],ariel-m1

hostname

source ./scripts/_setup.sh


# python run.py dataset.batch_size=3 compile=false
python run.py dataset.batch_size=3
