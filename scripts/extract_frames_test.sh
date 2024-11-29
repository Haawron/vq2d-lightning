#!/bin/bash

#SBATCH --job-name=extract-frames-test-320ss
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=50G
#SBATCH -w ariel-k2

hostname

# # raw
# python -Bm ltvu.preprocess.extract_frames --splits 'test_unannotated' --short-side 0

# # 520ss
# python -Bm ltvu.preprocess.extract_frames --splits 'test_unannotated' --short-side 520 --whole

# 320ss
python -Bm ltvu.preprocess.extract_frames --splits 'test_unannotated' --short-side 320 --whole
