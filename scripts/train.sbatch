#!/bin/bash

#SBATCH --job-name=lit-train
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=42G
#SBATCH -x ariel-v[3,6],ariel-k[1,2],ariel-m1

hostname

if [ ! -d "/local_datasets/ego4d_data" ]; then
    if [ -d "/data2/local_datasets" ]; then
        mkdir -p /data2/local_datasets/ego4d_data
        ln -s /data2/local_datasets/ego4d_data /local_datasets/
    fi
fi

tar -xf /data/datasets/tarfiles/vq2d_pos_and_query_frames_520ss.tar -C /local_datasets/

python run.py
