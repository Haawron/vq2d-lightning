#!/bin/bash

#SBATCH --job-name=extract_egotracks
#SBATCH --array=0-15
#SBATCH --output=logs/slurm/%A-%a--%x.out
#SBATCH --error=logs/slurm/%A-%a--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --mem=45G
#SBATCH -x ariel-k[1,2],ariel-m1

RANK=$SLURM_ARRAY_TASK_ID
WORLDSIZE=$SLURM_ARRAY_TASK_COUNT

sleep $RANK

# egotracks
python /data/soyeonhong/vq2d/vq2d-lightning/ltvu/extract_rt_pos_query_egotracks.py --rank $RANK --world-size $WORLDSIZE 

# got10k
# python /data/soyeonhong/vq2d/vq2d-lightning/ltvu/extract_rt_pos_query_got10k.py --rank $RANK --world-size $WORLDSIZE

# trackingnet
# python /data/soyeonhong/vq2d/vq2d-lightning/ltvu/extract_rt_pos_query_trackingnet.py --rank $RANK --world-size $WORLDSIZE