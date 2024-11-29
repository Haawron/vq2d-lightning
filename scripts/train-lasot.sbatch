#!/bin/bash

#SBATCH --job-name=lasot
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=PARTITION
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=43G

set -e  # exit on error

hostname

sshopt="-o UserKnownHostsFile=/data/$USER/.ssh/known_hosts -i /data/$USER/.ssh/id_ed25519"

################# get batch host info and network interfaces #################
batchhost=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
batchhostip=$(getent hosts $batchhost | head -n1 | awk '{print $1}')
batchhostport=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
interfaces=()
for host in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    echo $host
    hostip=$(ssh $sshopt $host hostname -i | awk '{print $1}')
    interfaces+=($(ssh $sshopt $host bash -c "ifconfig | grep -B1 $hostip | head -n1 | awk '{print \$1}' | sed 's/:\$//'"))
done
interfaces=$(echo "${interfaces[@]}" | tr ' ' ',')  # string join

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Batch host: $batchhost"
echo "Batch host IP: $batchhostip"
echo "Batch host port: $batchhostport"
echo "Network interfaces: ${interfaces[@]}"


################# run training #################
# ours
MASTER_ADDR=$batchhostip MASTER_PORT=$batchhostport NCCL_SOCKET_IFNAME=$interfaces \
    srun -N $SLURM_NNODES --exclusive --open-mode=append --cpus-per-task=8 \
    python train.py --config-name train_fast \
        trainer.num_nodes=$SLURM_NNODES trainer.devices=$SLURM_NTASKS_PER_NODE \
        dataset=lasot model.base_sizes=\[16,32,64,128,256\] +trainer.check_val_every_n_epoch=10 trainer.max_epochs=1500 \
        +model_adjust=\[no_sttx,no_bottleneck,conv_summary\] \
        +experiment=\[random_pos_query,cls_token_score,pca_guide\] \
        model.late_reduce=true \
        +model.weight_entropy=0.1 \
        +model.ignore_border=false \
        batch_size=3 \

# # baseline
# MASTER_ADDR=$batchhostip MASTER_PORT=$batchhostport NCCL_SOCKET_IFNAME=$interfaces \
#     srun -N $SLURM_NNODES --exclusive --open-mode=append --cpus-per-task=8 \
#     python train.py --config-name train_fast \
#         trainer.num_nodes=$SLURM_NNODES trainer.devices=$SLURM_NTASKS_PER_NODE \
#         dataset=lasot model.base_sizes=\[16,32,64,128,256\] +trainer.check_val_every_n_epoch=10 trainer.max_epochs=1500 \
#         model.late_reduce=true \
#         batch_size=3 \

exit $?
