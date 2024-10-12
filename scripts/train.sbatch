#!/bin/bash

#SBATCH --job-name=cls
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=51G
#SBATCH -x  aurora-g[1,7]

hostname

source ./scripts/_setup.sh



# python train.py --config-name train_fast \
#     model.num_layers_st_transformer=0 \
#     trainer.logger.0.group='role-of-st'

# python train.py --config-name train_fast \
#     model.num_layers_st_transformer=1 \
#     trainer.logger.0.group='role-of-st'

# python train.py --config-name train_fast \
#     +experiment=cls_token_score \
#     model.late_reduce=false batch_size=8

# python train.py --config-name train_fast \
#     +experiment=cls_token_score \
#     model.num_layers_st_transformer=0 \

python train.py --config-name train_fast \
    +experiment=cls_token_score \
