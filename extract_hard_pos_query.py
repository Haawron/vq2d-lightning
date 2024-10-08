import json

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from decord import VideoReader

def main():
    split = 'train'

    all_anns = json.load(open(f'../data/vq_v2_{split}_anno.json'))
    
    # /local_datasets/ego4d_data/v2/vq2d_frames/520ss
    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips')
    p_crop_out_dir = Path('rt_pos_queries') / split

    for aidx, ann in enumerate(tqdm(all_anns)):
        clip_uid = ann['clip_uid']
        qset_uuid = f"{ann['annotation_uid']}_{ann['query_set']}"
        rt = ann['response_track']
        ow, oh = ann['original_width'], ann['original_height']
        frame_idxs = [f['fno'] for f in rt]

        p_obj_dir = p_crop_out_dir / clip_uid
        p_obj_dir.mkdir(exist_ok=True, parents=True)

        p_clip = p_clips_dir / f'{clip_uid}.mp4'
        vr = VideoReader(str(p_clip))
        
        for idx, frame_idx in enumerate(frame_idxs):
            frame = vr[min(6*frame_idx, len(vr)-1)].asnumpy()
            w, h = frame.shape[1], frame.shape[0]
            x1, y1, x2, y2 = rt[idx]['x'] / ow * w, rt[idx]['y'] / oh * h, (rt[idx]['x'] + rt[idx]['w']) / ow * w, (rt[idx]['y'] + rt[idx]['h']) / oh * h
            img = Image.fromarray(frame)
            cropped = img.crop((x1, y1, x2, y2))
            p_out = p_obj_dir / f'{clip_uid}_{frame_idx}_{qset_uuid}.jpg'
            
            
if __name__ == '__main__':
    main()