#!/bin/bash

#SBATCH --job-name=rt-pos-qeury-thr-curriculum
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=42G
#SBATCH -w ariel-v5

hostname

source ./scripts/_setup.sh


# affine
# python train.py --config-name train_fast \
#     trainer.logger.0.group='augment' \
#     augment=affine \
#     augment.segment.aug_list._args_.0._args_.1.p=0

# # erase
# python train.py --config-name train_fast \
#     model.resolution_transformer=8 \
#     model.num_anchor_regions=64 \
#     batch_size=4 \

# # both
# python train.py --config-name train_fast \
#     trainer.logger.0.group='augment' \
#     augment=affine

# rt_pos_query sort_curriculum
# python train.py --config-name train_fast \
#     "+experiment=[rt_pos_query]" \
#     "rt_pos_query.sort_curriculum=True" \
#     "rt_pos_query.sim_thr=0.5"

# rt_pos_query query jitter
# python train.py --config-name train_fast \
#     "+experiment=[rt_pos_query]" \
#     "rt_pos_query.query_jitter=True"

# rt_pos_query thr curriculum
python train.py --config-name train_fast \
    "+experiment=[rt_pos_query]" \
    rt_pos_query.sim_thr=0.7 \
    rt_pos_query.enable_rt_pq_threshold=True \
    rt_pos_query.thr_curriculum=True

# rt_pos_query box jitter
# python train.py --config-name train_fast \
#     "+experiment=[rt_pos_query]" \
#     rt_pos_query.sim_thr=0.6 \
#     rt_pos_query.enable_rt_pq_threshold=True\
#     "rt_pos_query.rt_pos_query_dir=/data/soyeonhong/vq2d/vq2d-lightning/outputs/rt_pos_queries_jitter/train"
    

# lt_track_query
# python train.py --config-name train_fast \
#     "+experiment=[lt_track_query]"
