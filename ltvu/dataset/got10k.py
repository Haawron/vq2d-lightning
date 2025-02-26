import json
from pathlib import Path

from omegaconf import DictConfig

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from PIL import Image
import imghdr

class GOT10KDataset(torch.utils.data.Dataset):
    def __init__(self, config: DictConfig, split: str = 'train'):
        torch.utils.data.Dataset.__init__(self)
        self.config = config
        ds_config = config.dataset
        self.p_got10k_rootdir = Path(ds_config.clips_dir)  # {PATH}/CLASSNAME/CLASSNAME-IDX/img/08d.jpg
        self.num_frames: int = ds_config.num_frames
        self.frame_interval: int = ds_config.frame_interval
        self.segment_size: tuple[int] = tuple(ds_config.segment_size)  # H, W, desired
        self.query_size: tuple[int] = tuple(ds_config.query_size)  # H, W, desired
        self.query_square: bool = ds_config.query_square
        self.query_padding: bool = ds_config.query_padding
        self.rt_pos_query = config.get('rt_pos_query')
        if ds_config.padding_value == 'mean':
            self.padding_value = .5
        elif ds_config.padding_value == 'zero':
            self.padding_value = 0.
            
        if self.rt_pos_query is not None:
            self.p_rt_pos_query = Path(self.rt_pos_query.rt_pos_query_dir)
            self.num_rt_pos_quey = ds_config.num_rt_pos_query

        if split == 'val':
            split = 'test'
        self.split = split
        
        self.p_clip_dir = self.p_got10k_rootdir / split
        self.anns = []
        for clip_dir in sorted(self.p_clip_dir.glob('*')):
            if not clip_dir.is_dir():
                continue
            if 'cache' in clip_dir.stem:
                continue
            if 'GOT-10k_Train_000996' in clip_dir.stem:
                continue
        
            clip_idx = int(clip_dir.stem.split('_')[-1])
            gt_st = pd.read_csv(clip_dir / 'groundtruth.txt', header=None, names=['x', 'y', 'w', 'h'])
            
            if len(gt_st) <= (self.num_frames - 1) * self.frame_interval + 1:
                continue
            
            self.anns.append({
                'clip_idx': clip_idx,
                'p_clip': clip_dir,
                'gt_st': gt_st,
                'class_name': clip_dir.stem,
            })

    def __len__(self):
        return len(self.anns)

    def get_segment_frames(self, ann, frame_idxs):
        p_clip = ann['p_clip']
        num_clip_frames = len(ann['gt_st'])
        frame_idxs = frame_idxs.clip(0, num_clip_frames - 1)

        # load - normalize - permute
        p_frames = [p_clip / f'{idx+1:08d}.jpg' for idx in frame_idxs]
        first_frame = Image.open(p_frames[0])
        frames = []
        for p_frame in p_frames:
            frame = Image.open(p_frame)
            if frame.size != first_frame.size:
                frame = frame.resize(first_frame.size, Image.BICUBIC)
            frame = TF.pil_to_tensor(frame)
            frames.append(frame)
        frames = torch.stack(frames)  # [t, c, h, w]
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
            pad = tuple(map(lambda a: int(round(a)), pad))
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
    
    def get_rt_pos_query(self, ann, frame_idxs):
        class_name = ann['class_name']
        gt_st = ann['gt_st']
        frame_idxs = [np.random.randint(0, len(gt_st)) for _ in range(self.num_rt_pos_quey)]
        rt_pos_queries, rt_pos_idx = [], []

        for frame_idx in frame_idxs:
            p_pos_frame = self.p_rt_pos_query / class_name / f'{frame_idx+1:08d}.jpg'
            if p_pos_frame.exists() and imghdr.what(p_pos_frame) is not None:
                frame = Image.open(p_pos_frame)
                frame = TF.pil_to_tensor(frame)
                frame = frame.float() / 255.
                if self.query_padding:
                    bbox_h, bbox_w = gt_st.iloc[frame_idx]['h'], gt_st.iloc[frame_idx]['w']
                    l, s = max(bbox_h, bbox_w), min(bbox_h, bbox_w)
                    pad_size = (l - s) // 2
                    if bbox_h > bbox_w:
                        pad = (pad_size, l - s - pad_size, 0, 0)
                    else:
                        pad = (0, 0, pad_size, l - s - pad_size)
                    pad = tuple(map(lambda a: int(round(a)), pad))
                    frame = F.pad(frame, pad, value=0)
                frame = F.interpolate(frame[None], size=self.query_size, mode='bilinear', align_corners=True, antialias=True)
            else:
                frame = torch.zeros(3, self.query_size[0], self.query_size[1])
                frame_idx = -1
            rt_pos_idx.append(frame_idx)
            rt_pos_queries.append(frame.squeeze(0))
        rt_pos_queries = torch.stack(rt_pos_queries)

        return rt_pos_queries, rt_pos_idx


class GOT10KFitDataset(GOT10KDataset):
    def __getitem__(self, idx):
        ann = self.anns[idx]
        p_clip = ann['p_clip']
        clip_uid = p_clip.stem
        clip_len = len(ann['gt_st'])

        # get inputs
        required_len = (self.num_frames - 1) * self.frame_interval + 1
        try:
            start = np.random.randint(0, clip_len - required_len)
        except:
            raise ValueError(f'{p_clip}, {clip_uid}, {clip_len}')
        frame_idxs = np.arange(start, start + required_len, self.frame_interval)

        segment = self.get_segment_frames(ann, frame_idxs)  # [t, c, h, w]
        oh, ow = segment.shape[-2:]
        gt_stt, gt_mask = self.get_st_track(ann, frame_idxs, oh, ow)  # prob as a binary mask

        query = self.get_query(segment, gt_stt)
        segment, gt_stt = self.pad_and_resize(segment, gt_stt)  # [t, c, s, s], [t, 4]
        
        if self.rt_pos_query is not None and self.split == 'train':
            rt_pos_queries, rt_pos_idx = self.get_rt_pos_query(ann, frame_idxs)

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
            'clip_fps': 30,  # float
            'query_frame': 999999,  # int
            'visual_crop': {"fno": 0, "x": 0, "y": 0, "w": 0, "h": 0},  # dict
            'object_title': ann['class_name'],  # str
        }
        
        if self.rt_pos_query is not None and self.split == 'train':
            (sample
                .setdefault('experiment', {})
                .setdefault('multi_query', {})
                .setdefault('rt_pos_queries', rt_pos_queries))
            sample['experiment']['multi_query']['rt_pos_idx'] = np.array(rt_pos_idx)

        return sample


class GOT10KEvalDataset(GOT10KDataset):
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
        frame_idxs = np.arange(seg_idx * t, (seg_idx + 1) * t, self.frame_interval)
        frame_idxs[frame_idxs >= num_frames_clip] = num_frames_clip - 1  # repeat

        segment = self.get_segment_frames(ann, frame_idxs)  # [t, c, h, w]
        oh, ow = segment.shape[-2:]
        gt_stt, gt_mask = self.get_st_track(ann, frame_idxs, oh, ow)  # prob as a binary mask
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
    # python -Bm ltvu.dataset.got10k
    import hydra
    hydra.initialize(config_path='../../config', version_base='1.3')
    config = hydra.compose(config_name='train', overrides=['dataset=got10k'])
    config.dataset.clips_dir = '/data/datasets/GOT10K'
    import lightning as L
    # L.seed_everything(42)
    ds = GOT10KFitDataset(config, split='train')
    from imgcat import imgcat
    import matplotlib.pyplot as plt
    import io
    # idx = 0  # landscape
    # idx = 565  # portrait
    idx = np.random.randint(0, len(ds))
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

    image = sample['query']
    img_io = io.BytesIO()
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.savefig(img_io, format='png')
    imgcat(img_io.getvalue())
    print()
