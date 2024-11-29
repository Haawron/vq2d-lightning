#!/bin/bash

#SBATCH --job-name=extract-frames-320ss
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=35G
#SBATCH --array=0-3
#SBATCH -x ariel-k[1,2],ariel-m1

hostname

# python -Bm ltvu.preprocess.extract_frames --short-side 320

rank=$SLURM_ARRAY_TASK_ID
python -Bm ltvu.preprocess.extract_frames --task vq2d --split train --short-side 320 \
    --world_size 4 --rank $rank
