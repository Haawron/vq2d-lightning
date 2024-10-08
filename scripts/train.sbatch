#!/bin/bash

#SBATCH --job-name=vqloc-fast-rt-pos-query_hard_0epoch
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=42G
#SBATCH -w ariel-v12

hostname

source ./scripts/_setup.sh

export WANDB_API_KEY=044b832cabd231bf47a089f8ba78e93a641b62b7


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

# rt_pos_query
python train.py --config-name train_fast \
    +experiment=rt_pos_query
