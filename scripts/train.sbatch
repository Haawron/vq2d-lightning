#!/bin/bash

#SBATCH --job-name=pca-3+6
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=42G
#SBATCH -x ariel-k[1,2],ariel-m1

hostname

source ./scripts/_setup.sh


# 4개 기본
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score\] \
#     batch_size=4 \
#     model.late_reduce=true \
#     model.num_layers_st_transformer=0 \

# stst 1
# python train.py --config-name train_fast \
#     batch_size=4 \
#     model.num_layers_stst_decoder=1 \
#     optim.lr_scheduler.warmup_iter=100 \

# no sttx + no bn
# python train.py --config-name train_fast \
#     batch_size=4 \
#     model.num_layers_st_transformer=0 \
#     model.resolution_transformer=32 model.num_anchor_regions=32 \
#     optim.lr_scheduler.warmup_iter=100 \

# PCA 3+6
python train.py --config-name train_fast \
    +experiment=\[pca_guide,pca_loss\] \
    batch_size=4 \
    model.num_layers_stst_decoder=1 \
    optim.lr_scheduler.warmup_iter=100 \
    trainer.logger.0.group='pca' \

# PCA 3+6 + 4개
# python train.py --config-name train_fast \
#     +experiment=\[pca_guide,pca_loss,rt_pos_query,cls_token_score\] \
#     batch_size=3 \
#     model.num_layers_stst_decoder=1 \
#     model.num_layers_st_transformer=0 \
#     model.resolution_transformer=32 model.num_anchor_regions=32 \
#     model.late_reduce=true \
#     optim.lr_scheduler.warmup_iter=100 \
#     trainer.logger.0.group='pca' \
