import json

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pandas as pd

def main():
    split = 'train'
    p_lasot_rootdir = Path('/data/datasets/LaSOT')
    p_split_csv = p_lasot_rootdir / f'{split}ing_set.txt'
    split_csv = set(pd.read_csv(p_split_csv, header=None).iloc[:, 0].tolist())
    p_crop_out_root = Path('/data/soyeonhong/vq2d/vq2d-lightning/outputs/rt_pos_queries_lasot') / split
    p_crop_out_root.mkdir(exist_ok=True, parents=True)
    
    for data in split_csv:
        class_name = data.split('-')[0]
        class_dir = p_lasot_rootdir / class_name / data
        class_img_dir = class_dir / 'img'
        p_out = p_crop_out_root / class_name / data
        p_out.mkdir(exist_ok=True, parents=True)
        
        gt_st = pd.read_csv(class_dir / 'groundtruth.txt', 
                header=None, names=['x', 'y', 'w', 'h'])
        
        for idx, gt in gt_st.iterrows():
            p_crop_out_dir = p_out / f'{idx+1:08d}.jpg'
            if p_crop_out_dir.exists():
                continue
            x, y, w, h = gt['x'], gt['y'], gt['w'], gt['h']
            img = Image.open(class_img_dir / f'{idx+1:08d}.jpg')
            if w <= 0.5 or h <= 0.5:
                cropped = img
            else:
                cropped = img.crop((x, y, x+w, y+h))
            cropped.save(p_crop_out_dir)
            
            print(f'{class_name}/{data}/{idx+1:08d}.jpg cropped and saved')
            
            
if __name__ == '__main__':
    main()