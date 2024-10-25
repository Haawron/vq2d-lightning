#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH -N 2
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
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
    hostip=$(ssh $sshopt $host hostname -i | awk '{print $1}')
    interfaces+=($(ssh $sshopt $host bash -c "ifconfig | grep -B1 $hostip | head -n1 | awk '{print \$1}' | sed 's/:\$//'"))
done
interfaces=$(echo "${interfaces[@]}" | tr ' ' ',')  # string join

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Batch host: $batchhost"
echo "Batch host IP: $batchhostip"
echo "Batch host port: $batchhostport"
echo "Network interfaces: ${interfaces[@]}"


################# setup #################
tar_test_frames=outputs/frames/vq2d_pos_and_query_frames_320ss-test_unannotated.tar
for host in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    ssh $sshopt $host "tar -xf $(realpath $tar_test_frames) -C /local_datasets 2>/dev/null" &
done
echo "Waiting for setup to finish..."
wait
echo "Setup finished."


################# run eval #################
MASTER_ADDR=$batchhostip MASTER_PORT=$batchhostport NCCL_SOCKET_IFNAME=$interfaces \
    srun -N $SLURM_NNODES --exclusive --open-mode=append --cpus-per-task=8 \
    python eval.py \
        +trainer.num_nodes=$SLURM_NNODES trainer.devices=$SLURM_NTASKS_PER_NODE \
        ckpt='outputs/batch/2024-10-19/133186/epoch\=54-prob_acc\=0.7952.ckpt' \
        test_submit=true

exit $?
