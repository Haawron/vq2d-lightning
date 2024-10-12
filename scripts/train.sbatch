#!/bin/bash

#SBATCH --job-name=pca-6
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



# python train.py --config-name train_fast \
#     +experiment=pca_guide \
#     trainer.logger.0.group='pca' \

# python train.py --config-name train_fast \
#     +experiment=pca_guide \
#     model.pca_guide_version=6 batch_size=8 \
#     +model.pca_guide_ablation=true \
#     trainer.logger.0.group='pca' \

python train.py --config-name train_fast \
    +experiment=pca_guide \
    model.pca_guide_version=6 batch_size=8 \
    trainer.logger.0.group='pca' \

# python train.py --config-name train_fast \
#     +experiment=pca_guide \
#     model.pca_guide_version=6  batch_size=8 \
#     model.num_layers_st_transformer=0 model.resolution_transformer=16 model.num_anchor_regions=32 \
#     trainer.logger.0.group='pca' \

# python train.py --config-name train_fast \
#     +experiment=pca_guide \
#     model.pca_guide_version=5 \
#     trainer.logger.0.group='pca' \
