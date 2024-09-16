#!/bin/bash

#SBATCH --job-name=vqloc-fast-affine2
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


# affine
python run.py --config-name fast \
    trainer.logger.0.group='augment' \
    augment=affine \
    augment.segment.aug_list._args_.0._args_.1.p=0

# # erase
# python run.py --config-name fast \
#     trainer.logger.0.group='augment' \
#     augment=affine \
#     augment.segment.aug_list._args_.0._args_.4.p=0

# # both
# python run.py --config-name fast \
#     trainer.logger.0.group='augment' \
#     augment=affine
