import json
import argparse

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from decord import VideoReader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    
    all_anns = json.load(open('/data/soyeonhong/vq2d/vq2d-lightning/data/egotracks/egotracks_train_anno.json'))
    all_anns = all_anns[args.rank::args.world_size]
    print(f'rank: {args.rank}, world_size: {args.world_size}, num_anns: {len(all_anns)}')

    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips')
    p_crop_out_dir = Path('/data/soyeonhong/vq2d/vq2d-lightning/outputs/rt_pos_queries/egotracks/train')
    p_crop_out_dir.mkdir(exist_ok=True, parents=True)

    for aidx, ann in enumerate(tqdm(all_anns)):
        if 'clip_uid' in list(ann.keys()):
            clip_uid = ann['clip_uid']
            qset_uuid = f"{clip_uid}_{ann['query_set']}"
            rt = ann['response_track']
            ow, oh = ann['original_width'], ann['original_height']
            frame_idxs = [f['fno'] for f in rt]

            p_obj_dir = p_crop_out_dir / qset_uuid
            p_obj_dir.mkdir(exist_ok=True, parents=True)

            p_clip = p_clips_dir / f'{clip_uid}.mp4'
            vr = VideoReader(str(p_clip))
            
            for idx, frame_idx in enumerate(frame_idxs):
                p_out = p_obj_dir / f'{frame_idx}.jpg'
                
                if p_out.exists():
                    continue
                
                frame = vr[min(6*frame_idx, len(vr)-1)].asnumpy()
                w, h = frame.shape[1], frame.shape[0]
                x1, y1, x2, y2 = rt[idx]['x'] / ow * w, rt[idx]['y'] / oh * h, (rt[idx]['x'] + rt[idx]['w']) / ow * w, (rt[idx]['y'] + rt[idx]['h']) / oh * h
                if x2 - x1 < 0.5 or y2 - y1 < 0.5:
                    continue
                img = Image.fromarray(frame)
                cropped = img.crop((x1, y1, x2, y2))
                
                cropped.save(p_out)
                print(f'{p_out} saved')
            
            
if __name__ == '__main__':
    main()