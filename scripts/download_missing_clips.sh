set -e

ln -s /data/$USER/.aws ~/.aws

clip_uids=$@

ego4d_dir="/data/datasets/ego4d_data"
clips_dir=$ego4d_dir/v2/clips

ego4d \
    --output_directory=$ego4d_dir \
    --dataset clips \
    --aws_profile_name ego4d \
    --no-metadata \
    --video_uids $clip_uids \
    -y

echo
ls $clips_dir

# vq_clips_448_dir=$ego4d_dir/v2/vq2d_clips_448
# for clip_uid in $clip_uids; do
#     clip_path=$clips_dir/$clip_uid.mp4
#     clip_target_path=$vq_clips_448_dir/$clip_uid.mp4
#     echo "Resizing $clip_path to $clip_target_path"
#     ffmpeg \
#         -i $clip_path \
#         -vf "scale='if(gt(iw,ih),-2,448)':'if(gt(iw,ih),448,-2)',fps=5" \
#         -c:a copy \
#         $clip_target_path
# done
