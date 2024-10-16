if [ ! -d "/local_datasets/ego4d_data" ]; then
  if [ -d "/data2/local_datasets" ]; then
    mkdir -p /data2/local_datasets/ego4d_data
    ln -s /data2/local_datasets/ego4d_data /local_datasets/
  fi
fi

DIR=/local_datasets/ego4d_data/v2/vq2d_frames/320ss
TARFILES=(
  outputs/frames/vq2d_frames_320ss.tar
  outputs/frames/vq2d_frames_320ss-val.tar
)
do_extract=0
for TARFILE in "${TARFILES[@]}"; do
  path_random_image=$(find $DIR -name '*.jpg' -type f -print -quit)
  if [ ! -d "$DIR" ] || [ "$TARFILE" -nt "$path_image" ]; then
    do_extract=1
    break
  fi
done

for TARFILE in "${TARFILES[@]}"; do
  if [ "$do_extract" -eq 1 ]; then
    echo "Tar file is newer, extracting..."
    mkdir -p $DIR
    if [ $? -ne 0 ]; then
      echo "Failed to create directory, exiting..."
      exit 1
    fi
    tar -xf $TARFILE -C /local_datasets/ --overwrite
    find "$DIR" -maxdepth 1 -type d -exec chmod 1777 {} \;
  else
    echo "Directory is up-to-date, skipping extraction."
  fi
done

#   if [ ! -d "$DIR" ] || [ "$TARFILE" -nt "$path_image" ]; then
#     echo "Tar file is newer, extracting..."
#     mkdir -p $DIR
#     if [ $? -ne 0 ]; then
#       echo "Failed to create directory, exiting..."
#       exit 1
#     fi
#     tar -xf $TARFILE -C /local_datasets/ --overwrite
#     find $DIR -maxdepth 1 -type d -exec chmod 1777 {} \;
#   else
#     echo "Directory is up-to-date, skipping extraction."
#   fi
# done
