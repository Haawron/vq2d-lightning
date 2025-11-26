# [HERO-VQL: Hierarchical, Egocentric and Robust Visual Query Localization](https://arxiv.org/abs/2509.00385)

[![Conference](https://img.shields.io/badge/BMVC-2025-blue)](https://bmvc2025.bmva.org/)  [![paper](https://img.shields.io/badge/paper-bmvc-red)](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_610/paper.pdf) [![arXiv](https://img.shields.io/badge/arXiv-2509.00385-red)](https://arxiv.org/abs/2509.00385)   [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://joohyunchang.github.io/HERO-VQL/)

This repository is the official implementation of the **"HERO-VQL: Hierarchical, Egocentric and Robust Visual Query Localization"**, accepted as an **Oral** presentation at BMVC 2025🔥.  
[[Project page]](https://joohyunchang.github.io/HERO-VQL/)


# Quick Start

### Environment Setup

```bash
# conda (recommended)
conda create -n hero-vql python=3.12 -y
conda activate hero-vql
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

## Pretrained Checkpoint
We provide a pretrained HERO-VQL checkpoint: [hero-vql.ckpt](https://www.dropbox.com/scl/fi/lijo6t3g6s5fk1qgpznkx/hero-vql.ckpt?rlkey=n9l0w22czzxv87vto3a2tmmvk&dl=1)

## Train

#### 1. Download VQ2D
- Please Follow [VQ2D - Preparing data for training and inference](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D#preparing-data-for-training-and-inference) steps 1, 2, and 4 only.

#### 2. Get Frames for Train

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

#### 3. Run

```bash
# train HERO-VQL
bash scripts/train.sbatch
```

## Evaluation

#### Get Frames for Eval

```bash
# 1. Extract video frames:
bash scripts/extract_frames_val.sh
# will generate ./outputs/frames/vq2d_pos_and_query_frames_{SHORTSIDE}ss-val.tar

# 2. Unpack frames:
tar -xvf TARFILE -C BASEDIR

# 3. Evaluate
# might take an hour with 8 A5000s
python eval.py --config-name eval ckpt=CKPTPATH
```

## Test (for submission to Ego4D challenge on EvalAI)
This procedure is similar to validation, but uses the test_unannotated split.
Set `test_submit=True` to generate the submission file required for the Ego4D VQ2D challenge (https://eval.ai/web/challenges/challenge-page/1843/overview).
```bash
# 1. Extract video frames:
bash scripts/extract_frames_test.sh
# will generate ./outputs/frames/vq2d_pos_and_query_frames_{SHORTSIDE}ss-test_unannotated.tar

# 2. Unpack frames:
tar -xvf TARFILE -C BASEDIR

# 3. Run inference in submission mode
python eval.py --config-name eval ckpt=CKPTPATH test_submit=True
# The submission file will be saved to: vq_test_unannotated.json
````

## Debug

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


## Citation
```
@inproceedings{chang2025herovql,
  title={HERO-VQL: Hierarchical, Egocentric and Robust Visual Query Localization},
  author={Chang, Joohyun and Hong, Soyeon and Lee, Hyogun and Ha, Seong Jong and Lee, Dongho and Kim, Seong Tae and Choi, Jinwoo},
  booktitle={BMVC},
  year={2025},
}
```