if [ ! -d "/local_datasets/ego4d_data" ]; then
  if [ -d "/data2/local_datasets" ]; then
    mkdir -p /data2/local_datasets/ego4d_data
    ln -s /data2/local_datasets/ego4d_data /local_datasets/
  fi
fi

DIR=/local_datasets/ego4d_data/v2/vq2d_frames/320ss
TARFILE=/data/datasets/tarfiles/ego4d/vq2d_pos_and_query_frames_320ss.tar
if [ ! -d "$DIR" ] || [ "$TARFILE" -nt "$DIR" ]; then
  echo "Tar file is newer, extracting..."
  mkdir -p $DIR
  if [ $? -ne 0 ]; then
    echo "Failed to create directory, exiting..."
    exit 1
  fi
  tar -xf $TARFILE -C /local_datasets/ --overwrite
  find $DIR -maxdepth 1 -type d -exec chmod 1777 {} \;
else
  echo "Directory is up-to-date, skipping extraction."
fi
