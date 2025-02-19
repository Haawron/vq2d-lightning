import json
import argparse

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from decord import VideoReader

def save_cropped_frames(frame_idxs, p_obj_dir, vr, rt, ow, oh):
    """
    Saves cropped frames based on provided indices and bounding boxes.

    Args:
        frame_idxs (list): List of frame indices to process.
        p_obj_dir (Path): Directory path to save cropped frames.
        vr (VideoReader): Video reader object containing video frames.
        rt (list): List of dictionaries with bounding box data for each frame.
        ow (float): Original width of the video.
        oh (float): Original height of the video.
    """
    for idx, frame_idx in enumerate(frame_idxs):
        p_out = Path(p_obj_dir) / f'{frame_idx}.jpg'
        
        if p_out.exists():
            continue

        frame = vr[min(6 * frame_idx, len(vr) - 1)].asnumpy()
        w, h = frame.shape[1], frame.shape[0]

        x1 = rt[idx]['x'] / ow * w
        y1 = rt[idx]['y'] / oh * h
        x2 = (rt[idx]['x'] + rt[idx]['w']) / ow * w
        y2 = (rt[idx]['y'] + rt[idx]['h']) / oh * h

        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue

        img = Image.fromarray(frame)
        cropped = img.crop((x1, y1, x2, y2))

        cropped.save(p_out)
        print(f'{p_out} saved')

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
        if 'clip_uid' in list(ann.keys()) and 'lt_track' in list(ann.keys()):
            clip_uid = ann['clip_uid']
            qset_uuid = f"{clip_uid}_{ann['query_set']}"
            rt = ann['response_track']
            lt = ann['lt_track']
            ow, oh = ann['original_width'], ann['original_height']
            frame_idxs = [f['fno'] for f in rt]
            frame_idxs_lt = [f['fno'] for f in lt]

            p_obj_dir = p_crop_out_dir / qset_uuid
            p_obj_dir.mkdir(exist_ok=True, parents=True)

            p_clip = p_clips_dir / f'{clip_uid}.mp4'
            vr = VideoReader(str(p_clip))
            
            save_cropped_frames(frame_idxs, p_obj_dir, vr, rt, ow, oh)
            save_cropped_frames(frame_idxs_lt, p_obj_dir, vr, lt, ow, oh)
            
            
if __name__ == '__main__':
    main()