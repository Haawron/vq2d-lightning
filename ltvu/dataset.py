import json
import random
from pathlib import Path
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from omegaconf import DictConfig

import decord

import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
from einops import rearrange

import lightning as L


decord.bridge.set_bridge("torch")
P_CLIPS_FOR_CHECKING_VALIDITY = None


def sample_nearby_gt_frames(
    gt_interval: list[int],  # both inclusive
    num_frames: int = 30,
    frame_interval: int = 1,
) -> np.ndarray:
    required_len = (num_frames - 1) * frame_interval + 1

    # extend the GT interval if it is shorter than required
    raw_gt_len = gt_interval[1] - gt_interval[0] + 1
    if raw_gt_len < required_len:  # extend the GT interval
        len_short = required_len - raw_gt_len  # shortage of length
        ext_left = max(gt_interval[0], np.random.randint(len_short + 1))  # left extension
        ext_right = len_short - ext_left
        gt_ = [gt_interval[0] - ext_left, gt_interval[1] + ext_right]
        assert gt_[0] >= 0
    else:
        gt_ = gt_interval[:]  # deep copy

    # get num_frames + 1 temporal anchors from the extended GT interval, only left border is inclusive
    gt_len = gt_[1] - gt_[0] + 1
    assert gt_len >= required_len
    in_gt_offset = np.random.randint(gt_len - required_len + 1)  # ex) 1 if they are equal
    t_anchors = gt_[0] + in_gt_offset + np.linspace(0, required_len, num_frames + 1).astype(int)  # both inclusive

    # sample a frame idxs from each interval
    frame_idxs = np.array([np.random.randint(s, e) for s, e in zip(t_anchors, t_anchors[1:])])
    assert frame_idxs.shape[0] == num_frames
    assert (frame_idxs >= 0).all()
    # TODO: Add the clip length assertion
    return frame_idxs


class VQ2DFitDataset(torch.utils.data.Dataset):
    ann_keys = [
        'video_uid',
        'clip_uid',
        'annotation_uid',
        'query_set',                    # str, annotation enumeration, 1-based
        'clip_fps',
        'query_frame',  # int, the end of the extent of the input clip (nothing to do with the query)
        'object_title',                 # str, object name (e.g. 'sellotape'), don't use this for training
        'visual_crop',                  # actual query
        'response_track_valid_range',   # both inclusive
        'response_track']

    def __init__(self, config: DictConfig, split: str = 'train'):
        super().__init__()
        self.config = config
        ds_config = config.dataset
        self.num_frames: int = ds_config.num_frames
        self.frame_interval: int = ds_config.frame_interval
        self.segment_size: list[int] = ds_config.segment_size  # H, W
        self.query_size: list[int] = ds_config.query_size  # H, W
        self.query_square: bool = ds_config.query_square
        self.query_padding: bool = ds_config.query_padding
        if ds_config.padding_value == 'mean':
            self.padding_value = .5
        elif ds_config.padding_value == 'zero':
            self.padding_value = 0.

        self.split = split
        self.p_clips_dir = Path('/data/datasets/ego4d_data/v2/vq2d_clips')
        self.p_anns_dir = Path('./data/')
        self.p_ann = self.p_anns_dir / f'vq_v2_{split}_anno.json'
        self.all_anns = json.load(self.p_ann.open())

    def __len__(self):
        return len(self.all_anns)

    def __getitem__(self, idx):
        # setup
        ann = self.all_anns[idx]
        clip_uid = ann['clip_uid']
        p_clip = self.p_clips_dir / f'{clip_uid}.mp4'
        vr = decord.VideoReader(str(p_clip), num_threads=1)

        # get inputs
        segment, seg_idxs = self.get_segment(ann, vr)
        bboxes, seg_with_gt = self.get_bboxes(ann, seg_idxs)
        query = self.get_query(ann, vr)

        return {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # GT
            'gt_bboxes': bboxes,  # [t, 4], yxyx, normalized
            'gt_prob': seg_with_gt,  # [t], GT prob

            # infos
            'video_uid': ann['video_uid'],
            'clip_uid': clip_uid,
            'annotation_uid': ann['annotation_uid'],
            'query_set': ann['query_set'],
            'clip_fps': ann['clip_fps'],
            'query_frame': ann['query_frame'],
            'object_title': ann['object_title'],
        }

    def get_segment(self, ann, vr, frame_idxs = None):
        """
        4 steps to get an input clip:
            1. sample
            2. load - normalize - permute
            3. pad or crop - resize
            (4. augment -> not here, GPU accelerated by Kornia)
        """
        # sample
        vlen = len(vr)
        origin_fps = int(vr.get_avg_fps())
        gt_fps = int(ann['clip_fps'])
        down_rate = origin_fps // gt_fps
        frame_idxs = frame_idxs or sample_nearby_gt_frames(
            ann['response_track_valid_range'], self.num_frames, self.frame_interval)
        frame_idxs_origin = np.minimum(down_rate * frame_idxs, vlen - 1)  # idxs FPS considered

        # load - normalize - permute
        frames = vr.get_batch(frame_idxs_origin)
        frames = frames.float() / 255.
        frames = rearrange(frames, 't h w c -> t c h w')
        t, c, h, w = frames.shape
        assert h <= w  # Check: if Ego4D clips are all in landscape mode

        # pad - resize
        pad_size = (h - w) // 2
        pad = (0, 0, pad_size, h - w - pad_size)   # Left, Right, Top, Bottom
        frames = F.pad(frames, pad, value=self.padding_value)
        _, _, h_pad, w_pad = frames.shape
        assert h_pad == w_pad
        frames = F.interpolate(frames, self.segment_size, mode='bilinear')

        return frames, frame_idxs

    def get_query(self, ann, vr):
        vc = ann['visual_crop']
        fno = vc['fno']
        oh, ow = vc['oh'], vc['ow']
        y, x, h, w = vc['y'], vc['x'], vc['h'], vc['w']

        if self.query_square:
            cx, cy, s = x + w / 2, y + h / 2, max(h, w)
            x, y, h, w = cx - s / 2, cy - s / 2, s, s
            x, y = max(0, x), max(0, y)
            h, w = min(oh - y, h), min(ow - x, w)  # don't have to be strictly square

        y, x, h, w = int(y), int(x), int(h), int(w)
        query: torch.Tensor = vr[fno]  # [c, oh, ow]
        query = query[..., y:y+h, x:x+w]

        if self.query_padding:
            pad_size = (h - w) // 2
            pad = (0, 0, pad_size, h - w - pad_size)   # Left, Right, Top, Bottom
            query = F.pad(query, pad, value=0)

        query = F.interpolate(query, self.query_size, mode='bilinear')
        query = query / 255.
        return query

    def get_bboxes(self, ann, seg_idxs: np.ndarray):
        rt = ann['response_track']
        rt_valid_range = ann['response_track_valid_range']

        # initialize bboxes with default values
        seg_with_gt = (seg_idxs <= rt_valid_range[1]) & (seg_idxs >= rt_valid_range[0])
        default_bbox = [0, 0, 1e-6, 1e-6]  # yxyx, normalized
        bboxes = np.array([default_bbox] * self.num_frames)
        valid_idx_start: int = np.where(seg_with_gt)[0][0]
        gt_idxs = np.array([res['fno'] for res in rt])
        assert len(set(gt_idxs) - set(seg_idxs)), 'All GT frames should be in the sampled frames.'

        # update bboxes with GT values
        ow, oh = rt[0]['ow'], rt[0]['oh']
        _bboxes = [[res['y'], res['x'], res['y'] + res['h'], res['x'] + res['w']] for res in rt]  # yxyx
        _bboxes = np.array(_bboxes) / [oh, ow, oh, ow]  # normalize
        bboxes[valid_idx_start:] = _bboxes
        bboxes = bboxes.astype(np.float32).clip(0, 1)

        return torch.tensor(bboxes), torch.tensor(seg_with_gt.float())


class VQ2DTestDatasetSeparated(VQ2DFitDataset):
    """An item is defined as a tuple `(qset_uuid or ann_idx, seg_idx)`."""

    def __init__(self, config, split = 'val'):
        super().__init__(config, split)
        # self.all_clip_uids = sorted(set([ann['clip_uid'] for ann in self.all_anns]))
        segment_length = (self.num_frames - 1) * self.frame_interval + 1
        self.all_seg_uids = []
        for ann_idx, ann in enumerate(self.all_anns):
            clip_uid = ann['clip_uid']
            num_frames_clip = ann['query_frame']
            num_segments = np.ceil(num_frames_clip / segment_length).astype(int).item()
            self.all_seg_uids.extend([(ann_idx, seg_idx, clip_uid) for seg_idx in range(num_segments)])

    def __len__(self):
        return len(self.all_seg_uids)

    def __getitem__(self, idx):
        ann_idx, seg_idx, clip_uid = self.all_clip_uids[idx]
        ann = self.all_anns[ann_idx]
        qset_uuid = f"{ann['annotation_uid']}_{ann['query_set']}"
        frame_idxs = np.arange(seg_idx * self.num_frames, (seg_idx + 1) * self.num_frames, self.frame_interval)
        assert frame_idxs.shape[0] == self.num_frames
        p_clip = self.p_clips_dir / f'{clip_uid}.mp4'
        vr = decord.VideoReader(str(p_clip), num_threads=1)

        segment, _ = self.get_segment(ann, vr)
        query = self.get_query(ann, vr)

        if segment.shape[0] < self.num_frames:
            pad_size = self.num_frames - segment.shape[0]
            pad = (pad_size, 0, 0, 0)
            segment = F.pad(segment, pad, value=0)
            assert segment.shape[0] == self.num_frames
            seg_mask = torch.tensor([1] * segment.shape[0] + [0] * pad_size)
        else:
            seg_mask = torch.ones(self.num_frames)

        return {
            'segment': segment,  # [t,c,h,w], normalized
            'query': query,  # [c,h,w], normalized
            'qset_uuid': qset_uuid,
            'seg_idx': seg_idx,
            'seg_mask': seg_mask,
        }


class LitDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.p_raw_anns_dir = Path('/data/datasets/ego4d_data/v2/annotations/')
        self.p_anns_dir = Path('./data/')
        self.p_clips_dir = Path('/data/datasets/ego4d_data/v2/vq2d_clips')
        global P_CLIPS_FOR_CHECKING_VALIDITY
        P_CLIPS_FOR_CHECKING_VALIDITY = self.p_clips_dir

    def prepare_data(self):
        print('Preparing data...')
        all_num_anns = [13607, 4504]
        video_uids, clip_uids = set(), set()
        for split, num_anns in zip(['train', 'val'], all_num_anns):
            p_raw_ann = self.p_raw_anns_dir / f'vq_{split}.json'
            all_anns = json.load(p_raw_ann.open())
            flat_anns = generate_flat_annotations(all_anns)
            assert len(flat_anns) == num_anns, f'Split {split} has {len(flat_anns)} annotations, expected {num_anns}'
            p_ann = self.p_anns_dir / f'vq_v2_{split}_anno.json'
            json.dump(flat_anns, p_ann.open('w'))
            for ann in flat_anns:
                video_uids.add(ann['video_uid'])
                clip_uids.add(ann['clip_uid'])
        assert len(clip_uids) == 4690, f'Expected 4690 clips, got {len(clip_uids)}'
        p_ann.with_name('video_uids.txt').write_text(' '.join(sorted(video_uids)))
        p_ann.with_name('clip_uids.txt').write_text(' '.join(sorted(clip_uids)))
        print('Data preparation done.')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            VQ2DFitDataset(self.config, split='train'),
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            VQ2DFitDataset(self.config, split='val'),
            batch_size=self.config.test.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            pin_memory=True,
            drop_last=False,
        )


def generate_flat_annotations(all_anns):
    assert P_CLIPS_FOR_CHECKING_VALIDITY is not None, 'P_CLIPS_FOR_CHECKING_VALIDITY is not set.'

    def polish_bbox_dict(bbox: dict):
        key_map = {
            'frame_number': 'fno',
            'x': 'x', 'y': 'y', 'width': 'w', 'height': 'h',
            'original_width': 'ow', 'original_height': 'oh'}
        return {key_map[k]: v for k, v in bbox.items() if k in key_map}

    flat_anns = []
    count_invalids = 0
    for ann_video in all_anns['videos']:
        video_uid = ann_video['video_uid']
        for ann_clip in ann_video['clips']:
            clip_uid = ann_clip['clip_uid']
            if clip_uid is None:
                continue
            clip_duration = ann_clip['video_end_sec'] - ann_clip['video_start_sec']
            clip_fps = ann_clip['clip_fps']
            _p_clip = P_CLIPS_FOR_CHECKING_VALIDITY / f'{clip_uid}.mp4'
            assert _p_clip.exists(), f'Clip {clip_uid} does not exist in {_p_clip.parent}.'
            for ann_annots in ann_clip['annotations']:
                annotation_uid = ann_annots['annotation_uid']
                for qset_id, qset in ann_annots['query_sets'].items():
                    if not qset['is_valid']:
                        count_invalids += 1
                        continue
                    rt = [polish_bbox_dict(bbox) for bbox in qset['response_track']]
                    sample = {
                        'video_uid': video_uid,
                        'clip_uid': clip_uid,
                        'annotation_uid': annotation_uid,
                        'query_set': qset_id,
                        'clip_fps': clip_fps,
                        'clip_duration': clip_duration,
                        'query_frame': qset['query_frame'],
                        'object_title': qset['object_title'],
                        'visual_crop': polish_bbox_dict(qset['visual_crop']),
                        'response_track_valid_range': [rt[0]['fno'], rt[-1]['fno']],
                        'response_track': rt,
                    }
                    flat_anns.append(sample)
    return flat_anns


if __name__ == '__main__':
    config = {}
    dm = LitDataModule(config)
    dm.prepare_data()
    print('Data preparation done.')
