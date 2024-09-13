

if [ ! -d "/local_datasets/ego4d_data" ]; then
  if [ -d "/data2/local_datasets" ]; then
    mkdir -p /data2/local_datasets/ego4d_data
    ln -s /data2/local_datasets/ego4d_data /local_datasets/
  fi
fi

TARFILE=/data/datasets/tarfiles/vq2d_pos_and_query_frames_520ss.tar
if [ "$TARFILE" -nt "$DIR" ]; then
  echo "Tar file is newer, extracting..."
  tar -xf $TARFILE -C /local_datasets/
else
  echo "Directory is up-to-date, skipping extraction."
fi
