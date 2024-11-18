#!/bin/bash

#SBATCH --job-name=final-abl
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


########################################################################################



# Ours
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
#     model.late_reduce=true \
#     +model.weight_entropy=0.1 \
#     +model.ignore_border=false \
#     model.cls_scaling_type='sigmoid_mean' \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     batch_size=3 \

# # Ours Large
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
#     model.late_reduce=true \
#     +model.weight_entropy=0.1 \
#     +model.ignore_border=false \
#     model.cls_scaling_type='sigmoid_mean' \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.backbone_type='vitl14' \
#     batch_size=3 \

# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
#     model.late_reduce=true \
#     +model.weight_entropy=0.1 \
#     +model.ignore_border=false \
#     model.cls_scaling_type='sigmoid_mean' \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.backbone_type='vitg14' \
#     batch_size=1 \

# Ours small
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
#     model.late_reduce=true \
#     +model.weight_entropy=0.1 \
#     +model.ignore_border=false \
#     model.cls_scaling_type='sigmoid_mean' \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.backbone_type='vits14' \
#     batch_size=5 \

# Ours DINOv1
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
#     model.late_reduce=true \
#     +model.weight_entropy=0.1 \
#     +model.ignore_border=false \
#     model.cls_scaling_type='sigmoid_mean' \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.backbone_type='vitb16' \
#     model.backbone_name='dino-hf' \
#     model.resolution_transformer=28 \
#     model.num_anchor_regions=28 \
#     batch_size=5 \
########################################################################################


# Baseline
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     model.late_reduce=true \
#     batch_size=3 \

# # Baseline Large
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     model.late_reduce=true \
#     model.backbone_type='vitl14' \
#     batch_size=3 \

# Baseline Gient
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     model.late_reduce=true \
#     model.backbone_type='vitg14' \
#     batch_size=1 \

# Baseline small
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     model.late_reduce=true \
#     model.backbone_type='vits14' \
#     batch_size=5 \

# Baseline DINOv1
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     model.late_reduce=true \
#     model.backbone_type='vitb16' \
#     model.backbone_name='dino-hf' \
#     model.resolution_transformer=28 \
#     model.num_anchor_regions=28 \
#     batch_size=3 \

########################################################################################



# Ours -slow
python train.py --config-name train_fast_jh \
    +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
    +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
    model.late_reduce=true \
    +model.weight_entropy=0.1 \
    +model.ignore_border=false \
    model.cls_scaling_type='sigmoid_mean' \
    +model.cls_repair_neighbor=true \
    +model.enable_temporal_shift_conv_summary=true \
    +dataset.movement='slow' \
    predict_val=false \
    predict_test=false \
    batch_size=3 \

# # Ours -medium
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
#     model.late_reduce=true \
#     +model.weight_entropy=0.1 \
#     +model.ignore_border=false \
#     model.cls_scaling_type='sigmoid_mean' \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     +dataset.movement='medium' \
#     predict_val=false \
#     predict_test=false \
#     batch_size=3 \

# # Ours -fast
# python train.py --config-name train_fast_jh \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide\] \
#     model.late_reduce=true \
#     +model.weight_entropy=0.1 \
#     +model.ignore_border=false \
#     model.cls_scaling_type='sigmoid_mean' \
#     +model.cls_repair_neighbor=true \
#     +model.enable_temporal_shift_conv_summary=true \
#     +dataset.movement='fast' \
#     predict_val=false \
#     predict_test=false \
#     batch_size=3 \