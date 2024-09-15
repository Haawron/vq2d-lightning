if [ ! -d "/local_datasets/ego4d_data" ]; then
  if [ -d "/data2/local_datasets" ]; then
    mkdir -p /data2/local_datasets/ego4d_data
    ln -s /data2/local_datasets/ego4d_data /local_datasets/
  fi
fi

DIR=/local_datasets/ego4d_data/v2/vq2d_frames/520ss
TARFILE=/data/datasets/tarfiles/vq2d_pos_and_query_frames_520ss.tar
if [ ! -d "$DIR" ] || [ "$TARFILE" -nt "$DIR" ]; then
  echo "Tar file is newer, extracting..."
  # check if having permission to write to DIR
  if [ ! -w $DIR ]; then
    echo "No write permission to $DIR"
    exit 1
  fi
  mkdir -p $DIR
  tar -xf $TARFILE -C /local_datasets/ --overwrite
else
  echo "Directory is up-to-date, skipping extraction."
fi
