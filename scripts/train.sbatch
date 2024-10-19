#!/bin/bash

#SBATCH --job-name=pca-guide-to-stx+pca_loss-sing1e-5
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=49G
#SBATCH -x ariel-k[1,2],ariel-m1

hostname

source ./scripts/_setup.sh


# 4개 기본
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score\] \
#     batch_size=4 \
#     model.late_reduce=true \
#     model.num_layers_st_transformer=0 \

# guide to stx
# python train.py --config-name train_fast \
#     +experiment=\[pca_guide\] \
#     +model_adjust=\[no_sttx,no_bottleneck\] \
#     batch_size=4

# python train.py --config-name train_fast \
#     +experiment=\[pca_guide,pca_loss\] \
#     +model_adjust=\[no_sttx,no_bottleneck\] \
#     batch_size=4
python train.py --config-name train_fast \
    +experiment=\[pca_guide,pca_loss\] \
    +model_adjust=\[no_sttx,no_bottleneck\] \
    batch_size=4 \
    model.weight_singular=1e-5
