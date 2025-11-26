#!/bin/bash

#SBATCH --job-name=ours-check
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=43G
#SBATCH -x ariel-k[1,2],ariel-m1


hostname

python train.py --config-name train_fast \
    +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide,frame_box_aug,box_penalty\]