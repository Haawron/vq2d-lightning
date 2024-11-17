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

source ./scripts/_setup.sh


# Ours
python train.py --config-name train_fast \
    +experiment=random_pos_query \
    model=match \
    # model.backbone_type='vitl14' \


# CLIP ViT-L/14
python train.py --config-name train_fast \
    +experiment=random_pos_query \
    model=match \
    model.backbone_name='clip-hf' \
    model.backbone_type='vitl14' \
    dataset.segment_size=\[336,336\] \
    dataset.query_size=\[336,336\] \
    model.resolution_transformer=24 \
    model.num_anchor_regions=24
