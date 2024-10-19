#!/bin/bash

#SBATCH --job-name=vq-resize
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-gpu=18
#SBATCH --mem-per-gpu=50G
#SBATCH -w ariel-m1

size=448
outdir=/data/datasets/ego4d_data/v2/vq2d_clips/${size}ss
mkdir -p $outdir

# Number of parallel subprocesses
world_size=8

# Get the list of files to process
files=(/data/datasets/ego4d_data/v2/clips/*.mp4)

# Get the total number of files
num_files=${#files[@]}

# Function to process files based on rank
process_files() {
    local rank=$1
    for (( i=rank; i<num_files; i+=world_size )); do
        filename=${files[$i]}
        clip_uid=$(basename "$filename" .mp4)
        echo "Process $rank resizing $filename ($((i+1))/$num_files)"
        outfile="$outdir/$clip_uid.mp4"
        ffmpeg -y -i "$filename" -vf "scale='if(gt(iw,ih),-2,$size)':'if(gt(iw,ih),$size,-2)',fps=5:start_time=-0.17:round=zero" -fps_mode vfr -hide_banner -loglevel error "$outfile"
    done
}

# Launch subprocesses
for (( rank=0; rank<world_size; rank++ )); do
    process_files $rank &
done

# Wait for all subprocesses to complete
wait

echo "All files have been resized."
