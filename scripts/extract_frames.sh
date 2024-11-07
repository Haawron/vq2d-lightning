#!/bin/bash

#SBATCH --job-name=extract-egotracks-frames-320ss
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

# python -Bm ltvu.preprocess --short-side 320

rank=$SLURM_ARRAY_TASK_ID
# python -Bm ltvu.preprocess --task egotracks --split train --whole --short-side 320 \
#     --world_size 4 --rank $rank
python -Bm ltvu.preprocess --task vq2d --split train --whole --short-side 320 \
    --world_size 4 --rank $rank
