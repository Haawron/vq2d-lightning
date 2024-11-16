#!/bin/bash

#SBATCH --job-name=lasot-resize
#SBATCH --output=logs/slurm/%j--%x.log
#SBATCH --error=logs/slurm/%j--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=50G
#SBATCH -w ariel-k2


size=320
inputdir=/data/datasets/LaSOT/
outdir="/data/datasets/LaSOT-${size}ss/"

mkdir -p "$outdir"

# Number of parallel subprocesses
world_size=8

# Get the list of files to process
# Use find to get all jpg files under inputdir
mapfile -t files < <(find "$inputdir" -type f -name '*.jpg')

# Get the total number of files
num_files=${#files[@]}

# Function to process files based on rank
process_files() {
    local rank=$1
    for (( i=rank; i<num_files; i+=world_size )); do
        filename=${files[$i]}
        # Get the relative path
        relative_path=${filename#$inputdir}
        # Construct output file path
        outfile="$outdir/$relative_path"
        # Create the output directory if it doesn't exist
        mkdir -p "$(dirname "$outfile")"
        echo "Process $rank resizing $filename ($((i+1))/$num_files)"
        # Resize image using ffmpeg, set the short side to $size pixels
        ffmpeg -y -i "$filename" -vf "scale='if(gt(iw,ih),-1,$size)':'if(gt(iw,ih),$size,-1)'" "$outfile"
    done
}

# Launch subprocesses
for (( rank=0; rank<world_size; rank++ )); do
    process_files $rank &
done

# Wait for all subprocesses to complete
wait

echo "All images have been resized."
