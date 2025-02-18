import json
from pathlib import Path

from omegaconf import DictConfig

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from PIL import Image


class Trek150Dataset(torch.utils.data.Dataset):
    def __init__(self, config: DictConfig, split: str = 'train'):
        torch.utils.data.Dataset.__init__(self)
        self.config = config
        ds_config = config.dataset
        self.anno_files = sorted(glob.glob(os.path.join(ds_config.clips_dir, '*/groundtruth_rect.txt')))
        self.frame_files = sorted(glob.glob(os.path.join(ds_config.clips_dir, '*/frames.txt')))
        self.seq_names = [f.split('/')[-2] for f in self.anno_files] # [''P03-P03_02-558','P03-P03_02-56',]
        self.seq_dirs = [os.path.join(ds_config.clips_dir, n) for n in self.seq_names] # ['/data/dataset/trek150/TREK-150/P03-P03_02-558','/data/dataset/trek150/TREK-150/P03-P03_02-56',]
        self.random_pos_query = False
        self.ope_benchmark = ds_config.get("ope_benchmark")
        
        self.num_frames: int = ds_config.num_frames
        self.frame_interval: int = ds_config.frame_interval
        self.segment_size: tuple[int] = tuple(ds_config.segment_size)  # H, W, desired
        self.query_size: tuple[int] = tuple(ds_config.query_size)  # H, W, desired
        self.query_square: bool = ds_config.query_square
        self.query_padding: bool = ds_config.query_padding
        if ds_config.padding_value == 'mean':
            self.padding_value = .5
        elif ds_config.padding_value == 'zero':
            self.padding_value = 0.

        self.split = split

        self.anns = []
        for class_name, clip_dir, gt_dir, frame_dir in zip(self.seq_names, self.seq_dirs, self.anno_files, self.frame_files):
            gt_st = pd.read_csv(gt_dir, header=None, names=['x', 'y', 'w', 'h'])
            # gt_st = pd.read_csv(gt_dir, header=None, names=['x1', 'y1', 'x2', 'y2'])
            with open(frame_dir, 'r') as f:
                frames = [line.strip() for line in f.readlines()]
            self.anns.append({
                'class_name': class_name,
                'p_clip': Path(clip_dir),
                'gt_st': gt_st,
                'clip_frames': frames,
            })
        # for p_class_dir in sorted(self.p_lasot_rootdir.glob('*')):
        #     if not p_class_dir.is_dir():
        #         continue
        #     if 'cache' in p_class_dir.stem:
        #         continue
        #     class_name = p_class_dir.stem
        #     for p_clip in sorted(p_class_dir.glob('*'), key=lambda p: int(p.stem.split('-')[-1])):
        #         clip_uid = p_clip.stem
        #         if clip_uid not in self.split_csv:
        #             continue
        #         clip_idx = int(clip_uid.split('-')[-1])
        #         gt_st = pd.read_csv(p_clip / 'groundtruth.txt', header=None, names=['x', 'y', 'w', 'h'])
        #         self.anns.append({
        #             'class_name': class_name,
        #             'clip_idx': clip_idx,
        #             'p_clip': p_clip,
        #             'gt_st': gt_st,
        #         })

    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, idx):
        ann = self.anns[idx]
        p_clip = ann['p_clip']
        clip_uid = p_clip.stem
        clip_len = len(ann['gt_st'])

        # get inputs
        required_len = (self.num_frames - 1) * self.frame_interval + 1
        start_idx = int(ann['clip_frames'][0])
        start = np.random.randint(start_idx, start_idx + clip_len - required_len)
        frame_idxs = np.arange(start, start + required_len, self.frame_interval).clip(0, start_idx + clip_len - 1)

        segment = self.get_segment_frames(ann, frame_idxs)  # [t, c, h, w]
        oh, ow = segment.shape[-2:]
        gt_stt, gt_mask = self.get_st_track(ann, frame_idxs - start_idx, oh, ow)  # prob as a binary mask

        query = self.get_query(segment, gt_stt)
        segment, gt_stt = self.pad_and_resize(segment, gt_stt)  # [t, c, s, s], [t, 4]

        sample = {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # GT
            'gt_bboxes': gt_stt.astype(np.float32),  # [t, 4], yxyx, normalized
            'gt_probs': gt_mask.astype(np.float32),  # [t], GT prob
            'before_query_mask': torch.tensor(gt_mask).bool(),  # [t], the key name is misleading due to the legacy code

            # for logging
            'video_uid': '',  # str
            'clip_uid': clip_uid,  # str
            'annotation_uid': '',
            'seg_idxs': frame_idxs,  # np.ndarray
            'query_set': '',  # str (of a single digit)
            'clip_fps': 60,  # float
            'query_frame': 999999,  # int
            # 'visual_crop': vc,  # dict
            'object_title': ann['class_name'],  # str
        }

        return sample


    def get_segment_frames(self, ann, frame_idxs):
        p_clip = ann['p_clip']
        num_clip_frames = len(ann['gt_st'])

        # load - normalize - permute
        p_frames = [p_clip / f'img/frame_{str(idx).zfill(10)}.jpg' for idx in frame_idxs]
        frames = [Image.open(p) for p in p_frames]
        frames = torch.stack([TF.pil_to_tensor(f) for f in frames])  # [t, c, h, w]
        frames = frames.float() / 255.

        return frames

    def get_st_track(self, ann, frame_idxs, oh, ow):
        gt_st: pd.DataFrame = ann['gt_st']
        gt_st['x2'] = gt_st['x'] + gt_st['w']
        gt_st['y2'] = gt_st['y'] + gt_st['h']
        gt_st = gt_st[['y', 'x', 'y2', 'x2']].iloc[frame_idxs].values.astype(np.float32)
        gt_st /= [oh, ow, oh, ow]
        return gt_st, np.ones(len(frame_idxs))

    def get_query(self, segment, gt_stt):
        if self.split == 'train' and self.random_pos_query and np.random.rand() < .5:
            idx = np.random.randint(0, len(segment))
        else:
            idx = 0

        query = segment[idx]
        oh, ow = segment.shape[-2:]
        y1, x1, y2, x2 = gt_stt[idx] * [oh, ow, oh, ow]
        w, h = x2 - x1, y2 - y1
        x, y = x1, y1
        l, s = max(w, h), min(w, h)  # large, short

        if self.query_square:  # but don't have to be strictly square, will be resized at the end of this function
            cx, cy, s = x + w / 2, y + h / 2, np.clip(l, a_min=10, a_max=min(oh, ow)-1).item()
            cx, cy = np.clip(cx, s / 2, ow - s / 2 - 1).item(), np.clip(cy, s / 2, oh - s / 2 - 1).item()
            x, y, w, h = cx - s / 2, cy - s / 2, s, s
            assert 0 <= x < ow and 0 <= y < oh and 0 < x + w < ow and 0 < y + h < oh, \
                f'Invalid visual crop: {x=}, {y=}, {h=}, {w=}, {oh=}, {ow=}'
            x, y, w, h = map(lambda a: int(round(a)), (x, y, w, h))

        # crop - permute - normalize
        query: torch.Tensor = TF.crop(query, y, x, h, w)  # [c, h, w]

        # permute - pad - resize
        if self.query_padding:
            pad_size = (l - s) // 2
            if h > w:
                pad = (pad_size, l - s - pad_size, 0, 0)   # Left, Right, Top, Bottom
            else:
                pad = (0, 0, pad_size, l - s - pad_size)   # Left, Right, Top, Bottom
            query = F.pad(query, pad, value=0)
        query = F.interpolate(query[None], size=self.query_size, mode='bilinear', align_corners=True, antialias=True)
        return query.squeeze(0)  # [c, h, w]

    def pad_and_resize(self, frames: torch.Tensor, bboxes: np.ndarray):
        # frames: [t, c, h, w]
        # bboxes: [t, 4], yxyx, normalized
        t, c, h, w = frames.shape
        bboxes *= [h, w, h, w]  # de-normalize

        # pad
        pad_size: int = abs(w - h) // 2
        if w > h:
            pad_top, pad_bot = pad_size, w - h - pad_size
            pad = (0, 0, pad_top, pad_bot)   # Left, Right, Top, Bottom
            frames = F.pad(frames, pad, value=self.padding_value)
            bboxes[:, [0, 2]] += float(pad_top)
        else:
            pad_left, pad_right = pad_size, h - w - pad_size
            pad = (pad_left, pad_right, 0, 0)
            frames = F.pad(frames, pad, value=self.padding_value)
            bboxes[:, [1, 3]] += float(pad_left)
        # verify padding
        _, _, h_pad, w_pad = frames.shape
        assert h_pad == w_pad, f'Padded frames should be square, got {frames.shape}'
        hw_pad = h_pad

        # resize
        frames = F.interpolate(frames, size=self.segment_size, mode='bilinear', align_corners=True, antialias=True)

        # normalize
        bboxes /= hw_pad

        return frames, bboxes



class Trek150FitDataset(Trek150Dataset):
    def __getitem__(self, idx):
        ann = self.anns[idx]
        p_clip = ann['p_clip']
        clip_uid = p_clip.stem
        clip_len = len(ann['gt_st'])

        # get inputs
        required_len = (self.num_frames - 1) * self.frame_interval + 1
        start_idx = int(ann['clip_frames'][0])
        start = np.random.randint(start_idx, start_idx + clip_len - required_len)
        frame_idxs = np.arange(start, start + required_len, self.frame_interval).clip(0, start_idx + clip_len - 1)

        segment = self.get_segment_frames(ann, frame_idxs)  # [t, c, h, w]
        oh, ow = segment.shape[-2:]
        gt_stt, gt_mask = self.get_st_track(ann, frame_idxs - start_idx, oh, ow)  # prob as a binary mask

        query = self.get_query(segment, gt_stt)
        segment, gt_stt = self.pad_and_resize(segment, gt_stt)  # [t, c, s, s], [t, 4]

        sample = {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # GT
            'gt_bboxes': gt_stt.astype(np.float32),  # [t, 4], yxyx, normalized
            'gt_probs': gt_mask.astype(np.float32),  # [t], GT prob
            'before_query_mask': torch.tensor(gt_mask).bool(),  # [t], the key name is misleading due to the legacy code

            # for logging
            'video_uid': '',  # str
            'clip_uid': clip_uid,  # str
            'annotation_uid': '',
            'seg_idxs': frame_idxs,  # np.ndarray
            'query_set': '',  # str (of a single digit)
            'clip_fps': 60,  # float
            'query_frame': 999999,  # int
            # 'visual_crop': vc,  # dict
            'object_title': ann['class_name'],  # str
        }

        return sample


class Trek150EvalDataset(Trek150Dataset):
    def __init__(self, config, split = 'val'):
        super().__init__(config, split)
        self.num_frames_per_segment = self.num_frames
        self.segment_length = self.frame_interval * self.num_frames_per_segment  # trailing stride is considered as occupied
        # self.test_submit = split == 'challenge_test_unannotated'
        del self.num_frames  # to avoid confusion

        self.all_segments = []
        for ann_idx, ann in enumerate(self.anns):
            p_clip = ann['p_clip']
            num_frames_clip = len(ann['gt_st'])
            num_segments = np.ceil(num_frames_clip / self.segment_length).astype(int).item()
            seg_uuids = [f'{p_clip.stem}_{seg_idx}' for seg_idx in range(num_segments)]
            for seg_idx in range(num_segments):
                self.all_segments.append({
                    'ann_idx': ann_idx,
                    'seg_idx': seg_idx,

                    'seg_uuid': seg_uuids[seg_idx],
                    'qset_uuid': p_clip.stem,
                    'num_segments': num_segments,
                })

    def __len__(self):
        return len(self.all_segments)

    def __getitem__(self, idx):
        seg_info = self.all_segments[idx]
        ann_idx, seg_idx = seg_info['ann_idx'], seg_info['seg_idx']
        ann = self.anns[ann_idx]
        p_clip = ann['p_clip']
        clip_uid = p_clip.stem
        num_frames_clip = len(ann['gt_st'])
        t = self.num_frames_per_segment
        start_idx = int(ann['clip_frames'][0])
        frame_idxs = np.arange(seg_idx * t, (seg_idx + 1) * t, self.frame_interval)
        frame_idxs[frame_idxs >= num_frames_clip] = num_frames_clip - 1  # repeat

        segment = self.get_segment_frames(ann, frame_idxs + start_idx)  # [t, c, h, w]
        oh, ow = segment.shape[-2:]
        gt_stt, gt_mask = self.get_st_track(ann, frame_idxs, oh, ow)  # prob as a binary mask
        if self.ope_benchmark:
            query_segment = self.get_segment_frames(ann, np.array([start_idx]))  # [1, c, h, w]
            query_stt, _ = self.get_st_track(ann, np.array([0]), oh, ow)  # prob as a binary mask
            query = self.get_query(query_segment, query_stt)
        else:
            query = self.get_query(segment, gt_stt)
        segment, gt_stt = self.pad_and_resize(segment, gt_stt)  # [t, c, s, s], [t, 4]

        return {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # # GT
            'gt_bboxes': gt_stt.astype(np.float32),  # [t, 4], yxyx, normalized
            'gt_probs': gt_mask.astype(np.float32),  # [t], GT prob
            'before_query_mask': torch.tensor(gt_mask).bool(),  # [t]

            # info
            'clip_uid': clip_uid,
            'seg_uuid': seg_info['seg_uuid'],
            'qset_uuid': seg_info['qset_uuid'],
            'seg_idx': seg_info['seg_idx'],
            'num_segments': seg_info['num_segments'],
            'original_height': oh,
            'original_width': ow,
            'frame_idxs': frame_idxs,
        }


if __name__ == '__main__':
    # python -Bm ltvu.dataset.trek150
    import hydra
    hydra.initialize(config_path='../../config', version_base='1.3')
    config = hydra.compose(config_name='train', overrides=['dataset=trek150'])
    import lightning as L
    # L.seed_everything(42)
    # ds = Trek150Dataset(config, split='train')
    ds = Trek150EvalDataset(config, split='val')
    from imgcat import imgcat
    import matplotlib.pyplot as plt
    import io
    # idx = 565  # portrait
    idx = np.random.randint(0, len(ds))
    idx = 0  # landscape
    sample = ds[idx]
    segment = sample['segment']
    gt_bboxes = sample['gt_bboxes']
    T = len(segment)

    for t in range(0, T, T // 10):
        image = plt.imshow(segment[t].permute(1, 2, 0).cpu().numpy())
        y1, x1, y2, x2 = gt_bboxes[t] * (segment.shape[-2:] * 2)
        ax = plt.gca()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=2))
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        plt.close()
        imgcat(img_io.getvalue())
        print()

    print(sample['clip_uid'])
    image = sample['query']
    img_io = io.BytesIO()
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.savefig(img_io, format='png')
    imgcat(img_io.getvalue())
    print()
