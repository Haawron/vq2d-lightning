# Lightning Implementation of VQ2D Models

<!-- <center>
<img src="resources/overview.png" alt="Repo Overview" style="width:450px;" />
</center> -->

## Quick Start

### Environment Setup

```bash
# conda (recommended)
conda create -n vq2d-lit python=3.12 -y
conda activate vq2d-lit
```

```bash
# install packages
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch

pip install -r requirements.txt
```

### Note

All scripts which end with `.sbatch` are sbatch scripts for Slurm.
So, you can submit the scripts just with `sbatch SCRIPTNAME.sbatch` after replacing the partition name with one belonging to your  Slurm environment.

```bash
#SBATCH --partition=PARTITION
```

### Train

#### Get Frames for Train

```bash
# 1. Extract video frames:
bash scripts/extract_frames.sh
# will generate ./outputs/frames/vq2d_pos_and_query_frames_{SHORTSIDE}ss.tar

# 2. Unpack frames:
tar -xvf TARFILE -C BASEDIR
# frame path template: BASEDIR/ego4d_data/v2/vq2d_frames/{SHORTSIDE}ss/{CLIPUID}/frame_{FRAMEIDX}.jpg
# the index is 1-based

# 3. Extract ground-truth crops
python -m ltvu.extract_rt_pos_query
# crop path template: outputs/rt_pos_queries/vq2d/train/{CLIPUID}/{CLIPUID}_{FRAMEIDX}_{QSET_UUID}.jpg
```

#### Run

```bash
# for the baseline
bash scripts/train-baseline.sbatch

# OR

# for ours
bash scripts/train.sbatch
```

### Evaluation

#### Get Frames for Eval

```bash
# 1. Extract video frames:
bash scripts/extract_frames_val.sh
# will generate ./outputs/frames/vq2d_pos_and_query_frames_{SHORTSIDE}ss-val.tar

# 2. Unpack frames:
tar -xvf TARFILE -C BASEDIR

# 3. Evaluate
# might take an hour with 8 A5000s
python eval.py ckpt=CKPTPATH
```

### Debug

#### Check Hydra Configuration

You can inspect Hydra configurations without running the code:

```bash
python train.py --cfg job            # Print job config
python train.py --cfg hydra          # Print Hydra config (check runtime dir, packages, etc.)
python train.py --cfg job --resolve  # Print resolved config (requires SLURM_JOB_ID)
python train.py -i                   # Print gathered config without running
```

#### Dry-run

```bash
# dry-run 2 steps for each of training and validation
python train.py +debug=base

# dry-run for overfitting 10 batches for 10 epochs
python train.py +debug=overfit
```
