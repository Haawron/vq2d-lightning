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



# Baseline
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     batch_size=4 \

###############################################################

# # ST model: conv
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +model.enable_temporal_shift_conv_summary=true \
#     batch_size=4 \

# # ST model: global sttx
# python train.py --config-name train_fast \
#     model.apply_sttx_mask=false \
#     batch_size=4 \

###############################################################

# # Guide: global (w/o repair)
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[cls_token_score\] \
#     model.cls_norm=false \
#     batch_size=4 \

# # Guide: global (w/ repair)
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[cls_token_score\] \
#     model.cls_norm=false \
#     +model.cls_repair_neighbor=true \
#     batch_size=4 \

# # Guide: Local
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[pca_guide\] \
#     batch_size=4 \

###############################################################

# # HQQ
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[rt_pos_query\] \
#     model.sim_between=positives \
#     batch_size=4 \

# # HQQ: bottom
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[rt_pos_query\] \
#     model.sim_between=positives \
#     rt_pos_query.mode=hard \
#     batch_size=4 \

# # HQQ: Random
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query\] \
#     model.sim_between=positives \
#     batch_size=4 \

###############################################################

# # Ours
# python train.py --config-name train_fast \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
#     model.sim_between=positives \
#     model.cls_norm=false \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     batch_size=4 \
