#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=43G
#SBATCH -x ariel-k[1,2],ariel-m1

set -e  # exit on error

hostname

sshopt="-o UserKnownHostsFile=/data/$USER/.ssh/known_hosts -i /data/$USER/.ssh/id_rsa"

################# get batch host info and network interfaces #################
batchhost=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
batchhostip=$(getent hosts $batchhost | head -n1 | awk '{print $1}')
batchhostport=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
interfaces=()
for host in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    echo $host
    hostip=$(ssh $sshopt $host hostname -i)
    interfaces+=($(ssh $sshopt $host bash -c "ifconfig | grep -B1 $hostip | head -n1 | awk '{print \$1}' | sed 's/:\$//'"))
done
interfaces=$(echo "${interfaces[@]}" | tr ' ' ',')  # string join

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Batch host: $batchhost"
echo "Batch host IP: $batchhostip"
echo "Batch host port: $batchhostport"
echo "Network interfaces: ${interfaces[@]}"


################# setup #################
for host in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    ssh $sshopt $host bash $(realpath ./scripts/_setup.sh) &
done
echo "Waiting for setup to finish..."
wait
echo "Setup finished."


################# run training #################
# full+tsmconv
# MASTER_ADDR=$batchhostip MASTER_PORT=$batchhostport NCCL_SOCKET_IFNAME=$batchhostinterface \
#     srun -N $SLURM_NNODES --unbuffered --exclusive --open-mode=append --cpus-per-task=8 \
#     python -u train.py --config-name train_fast \
#         +trainer.num_nodes=$SLURM_NNODES trainer.devices=$SLURM_NTASKS_PER_NODE \
#         +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide,pca_loss\] \
#         +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
#         +model.enable_temporal_shift_conv_summary=true \
#         model.late_reduce=true \
#         model.cls_norm=true \
#         rt_pos_query.sim_thr=0.6 \
#         batch_size=3

# full+tsmattn+tsmconv
MASTER_ADDR=$batchhostip MASTER_PORT=$batchhostport NCCL_SOCKET_IFNAME=$interfaces \
    srun -N $SLURM_NNODES --exclusive --open-mode=append --cpus-per-task=8 \
    python train.py --config-name train_fast \
        +trainer.num_nodes=$SLURM_NNODES trainer.devices=$SLURM_NTASKS_PER_NODE \
        +experiment=\[thr_rt_pos_query,cls_token_score,pca_guide,pca_loss\] \
        +model_adjust=\[no_sttx,no_bottleneck,conv_summary,temporal_shift\] \
        model.late_reduce=true \
        model.cls_norm=true \
        rt_pos_query.sim_thr=0.6 \
        batch_size=3

exit $?
