#!/bin/bash

#SBATCH --job-name=ours
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=PARTITION
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=43G


hostname

# source ./scripts/_setup.sh

python train.py --config-name train_fast \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=0.1 \
    +model.ignore_border=false \
