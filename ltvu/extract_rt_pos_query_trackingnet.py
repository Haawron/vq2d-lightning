import json
import argparse

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    
    split = 'train'
    p_trackingnet_root = Path('/data/datasets/TrackingNet')
    p_trackingnet = p_trackingnet_root / split.upper()
    p_img = p_trackingnet / 'zips'
    train_dir = sorted([folder.name for folder in p_img.iterdir() if folder.is_dir()])
    train_dir = train_dir[args.rank::args.world_size]
    print(f'rank: {args.rank}, world_size: {args.world_size}, num_anns: {len(train_dir)}')
    p_crop_out_root = Path('/data/soyeonhong/vq2d/vq2d-lightning/outputs/rt_pos_queries/trackingnet') / split
    p_crop_out_root.mkdir(exist_ok=True, parents=True)
    
    
    for folder in tqdm(train_dir):
        p_gt = p_trackingnet / 'anno' / f'{folder}.txt'
        gt_st = pd.read_csv(p_gt, header=None, names=['x', 'y', 'w', 'h'])
        
        p_out = p_crop_out_root / folder
        p_out.mkdir(exist_ok=True, parents=True)
        
        for idx, gt in gt_st.iterrows():
            p_crop_out_dir = p_out / f'{idx}.jpg'
            if p_crop_out_dir.exists():
                continue
            x, y, w, h = gt['x'], gt['y'], gt['w'], gt['h']
            img = Image.open(p_img / folder / f'{idx}.jpg')
            if w <= 0.5 or h <= 0.5:
                continue
            else:
                cropped = img.crop((x, y, x+w, y+h))
            cropped.save(p_crop_out_dir)
            
if __name__ == '__main__':
    main()