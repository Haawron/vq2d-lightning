#!/bin/bash

#SBATCH --job-name=tmptmptmp
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

# # VQLoc w HNM 
# python train.py --config-name train_fast \
#     +use_hnm=true \
#     batch_size=8 \

# # VQLoc w HNM and HQQ
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query\] \
#     model.sim_between=positives \
#     +use_hnm=true \
#     batch_size=8 \

###############################################################

# Ours
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=1. \
    model.sim_between=positives \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=true \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

###############################################################

# Ours - PCA loss
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    model.sim_between=positives \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=true \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

# Ours - PCA loss - HQQ
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[cls_token_score,pca_guide\] \
    model.late_reduce=true \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=true \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

# Ours - PCA loss - HQQ - Guide
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    model.late_reduce=true \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

# # Ours - PCA loss - HQQ - Guide - TAC<-vqloc (이미 했음)
# python train.py --config-name train_fast \
#     model.late_reduce=true \
#     batch_size=3 돌리지=마세요 \

###############################################################

# No HQQ
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=1. \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=true \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

# bottom
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=1. \
    model.sim_between=positives \
    rt_pos_query.mode=hard \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=true \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

# Random
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=1. \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=true \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

###############################################################

# No global guide
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[rt_pos_query,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=1. \
    model.sim_between=positives \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \

# Global guide (w/o repair)
python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=1. \
    model.sim_between=positives \
    model.cls_scaling_type='sigmoid_mean' \
    +model.enable_temporal_shift_conv_summary=true \
    batch_size=3 \
