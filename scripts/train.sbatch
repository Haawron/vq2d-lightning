#!/bin/bash

#SBATCH --job-name=ours-conv2gst
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


# ours_base
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.late_reduce=true \
#     model.cls_norm=true \
#     batch_size=3 \

# ours_base CLS v3
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.late_reduce=true \
#     model.cls_norm=false \
#     model.cls_repair_neighbor=true \
#     batch_size=3 \

# ours_base - temporal shift
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     model.late_reduce=true \
#     model.cls_norm=true \
#     batch_size=3 \
#     +model.type_pe_stx=null

# ours_base -> stst
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
#     +model_adjust=\[no_sttx,no_bottleneck\] \
#     +model.num_layers_short_term_spatio_temporal_transformer=1 \
#     model.late_reduce=true \
#     model.cls_norm=true \
#     batch_size=3 \
#     +model.type_pe_stx=null

# ours_base - CLS
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,pca_guide\] \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.late_reduce=true \
#     batch_size=3 \
#     +model.type_pe_stx=null

# ours_base - PCA
# python train.py --config-name train_fast \
#     +experiment=\[rt_pos_query,cls_token_score\] \
#     +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#     +model.enable_temporal_shift_conv_summary=true \
#     model.late_reduce=true \
#     batch_size=3 \
#     +model.type_pe_stx=null

# ours_base - conv + bn + global st
python train.py --config-name train_fast \
    +experiment=\[rt_pos_query,cls_token_score,pca_guide\] \
    model.apply_sttx_mask=false \
    model.late_reduce=true \
    model.cls_norm=true \
    batch_size=3 \
