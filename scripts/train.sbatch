#!/bin/bash

#SBATCH --job-name=vqloc+diffusion
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=42G
#SBATCH -x ariel-v[3,6],ariel-k[1,2],ariel-m1

hostname

source ./scripts/_setup.sh


# python train.py --config-name train_fast \
#     model.resolution_transformer=8 \
#     model.num_anchor_regions=64 \
#     batch_size=4 \

# python train.py --config-name train_fast \
#     model.resolution_transformer=16 \
#     model.num_anchor_regions=64 \
#     batch_size=3 \
#     trainer.logger.0.group='bbox_upscale'



python train.py --config-name train_fast \
    +experiment=diffusion_data_as_query dataset.diffusion_data_as_query.data_dir=notebooks/diffusion \
    trainer.logger.0.group='diffusion'
