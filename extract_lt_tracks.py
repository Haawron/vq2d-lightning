import json
import argparse

from pathlib import Path
from PIL import Image
from decord import VideoReader
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    
    args = parser.parse_args()
    
    all_anns = json.load(open('/data/soyeonhong/vq2d/vq2d-lightning/data/egotracks_v1_train.json'))
    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips')
    p_crop_out_dir = Path('/data/soyeonhong/vq2d/vq2d-lightning/outputs/lt_tracks_2') / 'train'
    all_anns = all_anns[args.rank::args.world_size]
    print(f'rank: {args.rank}, world_size: {args.world_size}, all_anns: {len(all_anns)}')

    for aidx, ann in enumerate(all_anns):
        clip_uid = ann['clip_uid']
        annotation_uid = ann['annotation_uid']
        object_title = ann['object_title']
        rt = ann['lt_track']
        ow, oh = ann['original_width'], ann['original_height']
        frame_idxs = [f['fno'] for f in rt]

        p_obj_dir = p_crop_out_dir / clip_uid / object_title
        p_obj_dir.mkdir(exist_ok=True, parents=True)

        p_clip = p_clips_dir / f'{clip_uid}.mp4'
        
        if not p_clip.exists():
            print(f"Clip not found: {p_clip}")
            continue
        
        vr = VideoReader(str(p_clip))
        
        for idx, frame_idx in enumerate(frame_idxs):
            # p_out = p_obj_dir / f'{clip_uid}_{frame_idx}_{annotation_uid}.jpg'
            p_out = p_obj_dir / f'{clip_uid}_{frame_idx}.jpg'
            
            if p_out.exists():
                continue
            
            frame = vr[min(6*frame_idx, len(vr)-1)].asnumpy()
            w, h = frame.shape[1], frame.shape[0]
            x1, y1, x2, y2 = rt[idx]['x'] / ow * w, rt[idx]['y'] / oh * h, (rt[idx]['x'] + rt[idx]['w']) / ow * w, (rt[idx]['y'] + rt[idx]['h']) / oh * h
            
            img = Image.fromarray(frame)
            
            # print(f"clip_uid: {clip_uid}, frame: {frame_idx}, crop: {x1}, {y1}, {x2}, {y2}")
            
            if x2 - x1 <= 0.58 or y2 - y1 <= 0.58 or x1 - x2 <= 0.58 or y1 - y2 <= 0.58:
                print(f"Empty crop: {p_out}")
                # cropped = img
                continue
            else:
                cropped = img.crop((x1, y1, x2, y2))
            
            cropped.save(p_out)

if __name__ == '__main__':
    main()