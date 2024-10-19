#!/bin/bash

#SBATCH --job-name vq2d_dataprocessing_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=112
#SBATCH --mem=170G
#SBATCH --partition batch
#SBATCH -w  vll5
#SBATCH -t 3-0
#SBATCH -e logs/slurm-%A_%x.err
#SBATCH -o logs/slurm-%A_%x.out

#  [train, val, test_unannotated]
# python ltvu/convert_videos_to_clips.py \
#     --annot-paths /data/dataset/ego4d_temp/ego4d_data/v2/annotations/vq_train.json \
#     --save-root /data/dataset/ego4d_temp/ego4d_data/v2/vq2d_all_clips \
#     --ego4d-videos-root /data/dataset/ego4d_temp/ego4d_data/v2/video_540ss \
#     --num-workers 16 \
#     --video-batch-size 20

# python ltvu/convert_videos_to_clips.py \
#     --annot-paths /data/dataset/ego4d_temp/ego4d_data/v2/annotations/vq_val.json \
#     --save-root /data/dataset/ego4d_temp/ego4d_data/v2/vq2d_all_clips \
#     --ego4d-videos-root /data/dataset/ego4d_temp/ego4d_data/v2/video_540ss \
#     --num-workers 16 \
#     --video-batch-size 20

python ltvu/convert_videos_to_clips.py \
    --annot-paths /data/dataset/ego4d_temp/ego4d_data/v2/annotations/vq_test_unannotated.json \
    --save-root /data/dataset/ego4d_temp/ego4d_data/v2/vq2d_all_clips \
    --ego4d-videos-root /data/dataset/ego4d_temp/ego4d_data/v2/video_540ss \
    --num-workers 16 \
    --video-batch-size 20