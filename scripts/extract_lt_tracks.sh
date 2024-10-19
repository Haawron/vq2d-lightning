#!/bin/bash

#SBATCH --job-name=extract_lt_tracks
#SBATCH --array=0-14
#SBATCH --gres=gpu:0
#SBATCH --output=logs/slurm/%A-%a--%x.out
#SBATCH --error=logs/slurm/%A-%a--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -x ariel-k[1,2],ariel-m1

RANK=$SLURM_ARRAY_TASK_ID
WORLDSIZE=$SLURM_ARRAY_TASK_COUNT

sleep $RANK


python /data/soyeonhong/vq2d/vq2d-lightning/extract_lt_tracks.py --rank $RANK --world-size $WORLDSIZE