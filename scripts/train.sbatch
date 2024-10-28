#!/bin/bash

#SBATCH --job-name=ours_base-nhead_stx=8
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=51G
#SBATCH -x ariel-k[1,2],ariel-m1

hostname

source ./scripts/_setup.sh


# ours_base
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.late_reduce=true \
#     model.cls_norm=true \
#     batch_size=3 \

# ours_base + conv_summary_1d_layers=3 without conv2d
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
#     +model_adjust=\[no_sttx,no_bottleneck\] \
#     +model.conv_summary_1d_layers=3 \
#     model.late_reduce=true \
#     model.cls_norm=true \
#     batch_size=3

# ours_base + nhead_stx=8
python train.py --config-name train_fast \
    +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +model.enable_temporal_shift_conv_summary=true \
    +model.nhead_stx=8 \
    model.late_reduce=true \
    model.cls_norm=true \
    batch_size=3 \
