#!/bin/bash

#SBATCH --job-name=full+tsmattn
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


# 4개 기본
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score\] \
#     batch_size=4 \
#     model.late_reduce=true \
#     model.num_layers_st_transformer=0 \

# full+tsmattn
python train.py --config-name train_fast \
    +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide,pca_loss\] \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +model.enable_temporal_shift_stx=true \
    model.late_reduce=true \
    model.cls_norm=true \
    rt_pos_query.sim_thr=0.6 \
    batch_size=3 \
