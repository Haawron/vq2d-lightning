import json

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from decord import VideoReader


def main():
    split = 'train'

    all_anns = json.load(open(f'data/vq_v2_{split}_anno.json'))

    p_clips_dir = Path('/data/datasets/ego4d_data/v2/clips')
    p_crop_out_dir = Path('outputs/rt_pos_queries/vq2d') / split

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
            p_out = p_obj_dir / f'{clip_uid}_{frame_idx}_{qset_uuid}.jpg'

            if p_out.exists():
                continue

            frame = vr[min(6*frame_idx, len(vr)-1)].asnumpy()
            w, h = frame.shape[1], frame.shape[0]
            x1, y1, x2, y2 = rt[idx]['x'] / ow * w, rt[idx]['y'] / oh * h, (rt[idx]['x'] + rt[idx]['w']) / ow * w, (rt[idx]['y'] + rt[idx]['h']) / oh * h

            img = Image.fromarray(frame)

            if x2 - x1 == 0 or y2 - y1 == 0:
                print(f"Empty crop: {p_out}")
                cropped = img
            else:
                cropped = img.crop((x1, y1, x2, y2))

            cropped.save(p_out)


if __name__ == '__main__':
    main()
