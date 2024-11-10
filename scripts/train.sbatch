#!/bin/bash

#SBATCH --job-name=final-abl
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
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
#     model.late_reduce=true \
#     +model.weight_entropy=0.1 \
#     +model.ignore_border=false \
#     model.cls_scaling_type='sigmoid_mean' \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     batch_size=3 \

########################################################################################

# - PCA loss
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=0.0 \
    +model.ignore_border=false \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=true \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

########################################################################################

# - Guide without repair
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=0.0 \
    +model.ignore_border=false \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=false \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \
