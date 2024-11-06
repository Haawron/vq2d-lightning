# built-in + hydra
import json
from pathlib import Path
from omegaconf import DictConfig

# torch
import torch
import torch.utils.data
from torch.nn import functional as F
import torchvision.transforms.functional as TF

# lightning
# import lightning as L

# others
import decord
import numpy as np
from PIL import Image

# local (ours)

import random

decord.bridge.set_bridge("torch")


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
        self.p_clips_dir = Path(ds_config.clips_dir)  # ./{clip_uid}.mp4 or ./{clip_uid}/frame_{idx+1:07d}.jpg
        self.p_anns_dir = Path(ds_config.flat_anns_dir)
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

        self.rt_pos_query = config.get('rt_pos_query')
        if self.rt_pos_query is not None:
            self.p_rt_pos_query = Path(self.rt_pos_query.rt_pos_query_dir)
        self.split = split
        self.p_ann = self.p_anns_dir / f'vq_v2_{split}_anno.json'
        self.all_anns = json.load(self.p_ann.open())
        self.all_anns = self.subsample_anns(self.all_anns)

    def __len__(self):
        return len(self.all_anns)

    def __getitem__(self, idx):
        # setup
        ann: dict = self.all_anns[idx]
        clip_uid = ann['clip_uid']
        clip_len = ann['query_frame']
        vc = ann['visual_crop']
        vc_idx = vc['fno']
        p_clip_dir = self.p_clips_dir / clip_uid
        idxs_avail = set(int(p.stem.split('_')[-1]) - 1 for p in p_clip_dir.glob('*.jpg'))

        # get inputs
        gt_ext: list[int] = ann['response_track_valid_range']  # s, e both inclusive
        frame_idxs = self.sample_frame_idxs(self.num_frames, self.frame_interval, clip_len, gt_ext)
        idxs_required = set([*frame_idxs.tolist(), vc_idx])

        assert idxs_required.issubset(idxs_avail), \
            f'{clip_uid} does not have all required frames in {p_clip_dir}: {idxs_required - idxs_avail}'

        segment = self.get_segment_frames(ann, frame_idxs)  # [t, c, h, w]
        gt_rt, gt_prob = self.get_response_track(ann, frame_idxs)  # prob as a binary mask

        if self.rt_pos_query is not None and self.split == 'train':
            rt_pos_queries, rt_pos_idx = self.get_rt_pos_query(ann, frame_idxs)

        segment, gt_rt = self.pad_and_resize(segment, gt_rt)  # [t, c, s, s], [t, 4]
        query = self.get_query(ann)

        sample = {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # GT
            'gt_bboxes': gt_rt.astype(np.float32),  # [t, 4], yxyx, normalized
            'gt_probs': gt_prob.astype(np.float32),  # [t], GT prob
            'before_query_mask': torch.tensor(frame_idxs < ann['query_frame']).bool(),  # [t], whether before the query frame, used for loss masking(?)

            # for logging
            'video_uid': ann['video_uid'],  # str
            'clip_uid': clip_uid,  # str
            'annotation_uid': ann['annotation_uid'],  # str
            'seg_idxs': frame_idxs,  # np.ndarray
            'query_set': ann['query_set'],  # str (of a single digit)
            'clip_fps': ann['clip_fps'],  # float
            'query_frame': ann['query_frame'],  # int
            'visual_crop': vc,  # dict
            'object_title': ann['object_title'],  # str
        }

        if self.rt_pos_query is not None and self.split == 'train':
            (sample
                .setdefault('experiment', {})
                .setdefault('multi_query', {})
                .setdefault('rt_pos_queries', rt_pos_queries))
            sample['experiment']['multi_query']['rt_pos_idx'] = np.array(rt_pos_idx)

        return sample

    def subsample_anns(self, anns):  # interface
        return anns

    def sample_frame_idxs(self, num_frames: int, frame_interval: int, clip_len: int, gt_ext = None):
        frame_idxs = sample_nearby_gt_frames(gt_ext, num_frames, frame_interval)
        frame_idxs = shift_indices_to_clip_range(frame_idxs, clip_len)
        return frame_idxs

    def get_segment_frames(self, ann, frame_idxs):
        """
        4 steps to get an input clip:
            1. sample
            2. load - normalize - permute
            3. pad or crop - resize
            (4. augment -> not here, GPU accelerated by kornia, done in the training loop, the lit data module)
        """
        p_clip = self.p_clips_dir / ann['clip_uid']

        # load - normalize - permute
        frames = [Image.open(p_clip / f'frame_{idx+1:07d}.jpg') for idx in frame_idxs]
        frames = torch.stack([TF.pil_to_tensor(f) for f in frames])  # [t, c, h, w]
        frames = frames.float() / 255.
        t, c, h, w = frames.shape
        assert h <= w

        return frames

    def pad_and_resize(self, frames: torch.Tensor, bboxes: np.ndarray):
        # frames: [t, c, h, w]
        # bboxes: [t, 4], yxyx, normalized
        t, c, h, w = frames.shape
        bboxes *= [h, w, h, w]  # de-normalize

        # pad
        assert w > h, f'All the videos in Ego4D are landscape, got {frames.shape}'
        pad_size: int = (w - h) // 2
        pad_top, pad_bot = pad_size, w - h - pad_size
        pad = (0, 0, pad_top, pad_bot)   # Left, Right, Top, Bottom
        frames = F.pad(frames, pad, value=self.padding_value)
        bboxes[:, [0, 2]] += float(pad_top)
        # verify padding
        _, _, h_pad, w_pad = frames.shape
        assert h_pad == w_pad, f'Padded frames should be square, got {frames.shape}'
        hw_pad = h_pad

        # resize
        frames = F.interpolate(frames, size=self.segment_size, mode='bilinear', align_corners=True, antialias=True)

        # normalize
        bboxes /= hw_pad

        return frames, bboxes

    def get_query(self, ann):
        vc = ann['visual_crop']
        oh, ow = ann['original_height'], ann['original_width']
        num_clip_frames = int(ann['clip_fps'] * ann['clip_duration'])
        fno = min(vc['fno'], num_clip_frames - 1)
        p_frame = self.p_clips_dir / ann['clip_uid'] / f'frame_{fno+1:07d}.jpg'
        x, y, w, h = vc['x'], vc['y'], vc['w'], vc['h']
        l, s = max(w, h), min(w, h)  # large, short

        if self.query_square:  # but don't have to be strictly square, will be resized at the end of this function
            cx, cy, s = x + w / 2, y + h / 2, np.clip(l, a_min=10, a_max=min(oh, ow)-1).item()
            cx, cy = np.clip(cx, s / 2, ow - s / 2 - 1).item(), np.clip(cy, s / 2, oh - s / 2 - 1).item()
            x, y, w, h = cx - s / 2, cy - s / 2, s, s
            assert 0 <= x < ow and 0 <= y < oh and 0 < x + w < ow and 0 < y + h < oh, \
                f'Invalid visual crop: {x=}, {y=}, {h=}, {w=}, {oh=}, {ow=}'

        # load
        query = Image.open(p_frame)

        # crop - permute - normalize
        oow, ooh = query.size  # might be pre-pre-processed already
        rho = (oh / ooh + ow / oow) / 2
        x, y, w, h = x / rho, y / rho, h / rho, w / rho
        query = query.crop((x, y, x + w, y + h))  # [y:y+h, x:x+w]  # [h, w, c]
        query = TF.pil_to_tensor(query)  # [c, h, w]
        query = query.float() / 255.

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

    def get_rt_pos_query(self, ann, frame_idxs):
        clip_uid = ann['clip_uid']
        query_set = ann['query_set']
        annotation_uid = ann['annotation_uid']

        rt_ann = {}
        for rt in ann['response_track']:
            rt_ann[rt['fno']] = {
                'w': rt['w'],
                'h': rt['h'],
            }

        rt_pos_queries, rt_pos_idx = [], []

        for frame_idx in frame_idxs:
            if frame_idx in list(rt_ann.keys()):
                frame = Image.open(self.p_rt_pos_query / clip_uid / f'{clip_uid}_{frame_idx}_{annotation_uid}_{query_set}.jpg')
                frame = TF.pil_to_tensor(frame)
                frame = frame.float() / 255.
                if self.query_padding:
                    bbox_h, bbox_w = rt_ann[frame_idx]['h'], rt_ann[frame_idx]['w']
                    l, s = max(bbox_h, bbox_w), min(bbox_h, bbox_w)
                    pad_size = (l - s) // 2
                    if bbox_h > bbox_w:
                        pad = (pad_size, l - s - pad_size, 0, 0)
                    else:
                        pad = (0, 0, pad_size, l - s - pad_size)
                    frame = F.pad(frame, pad, value=0)
                frame = F.interpolate(frame[None], size=self.query_size, mode='bilinear', align_corners=True, antialias=True)
            else:
                frame = torch.zeros(3, self.query_size[0], self.query_size[1])
                frame_idx = -1
            rt_pos_idx.append(frame_idx)
            rt_pos_queries.append(frame.squeeze(0))

        rt_pos_queries = torch.stack(rt_pos_queries)

        return rt_pos_queries, rt_pos_idx

    def get_response_track(self, ann: dict, frame_idxs: np.ndarray):
        """_summary_

        Parameters
        ----------
        ann : dict
            _description_
        frame_idxs : np.ndarray
            Frame indices of the segment (a part of the clip).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            bboxes: [t, 4], yxyx, normalized
            seg_with_gt: [t], np.float32, 1. if the segment contains GT bbox else 0.
        """
        oh, ow = ann['original_height'], ann['original_width']
        gt_rt = ann['response_track']
        gt_ext = ann['response_track_valid_range']
        assert gt_ext[1] - gt_ext[0] + 1 == len(gt_rt)

        # initialize bboxes with default values
        seg_with_gt = (frame_idxs <= gt_ext[1]) & (frame_idxs >= gt_ext[0])

        # update bboxes with GT values
        bboxes = []
        for frame_idx, with_gt in zip(frame_idxs, seg_with_gt):
            if with_gt:
                res = gt_rt[frame_idx - gt_ext[0]]
                assert frame_idx == res['fno']
                bbox = [res['y'], res['x'], res['y'] + res['h'], res['x'] + res['w']]  # yxyx
                bboxes.append(bbox)
            else:
                bboxes.append([0, 0, 1e-5, 1e-5])

        # normalize
        bboxes = np.array(bboxes, dtype=np.float32) / [oh, ow, oh, ow]
        bboxes = bboxes.clip(0, 1)

        return bboxes, seg_with_gt.astype(np.float32)


def sample_nearby_gt_frames(
    gt_ext: list[int],  # both inclusive
    num_frames: int = 30,
    frame_interval: int = 1,
) -> np.ndarray:
    """Sample frame indices from the GT interval.
    Extend the GT interval if it is shorter than required.
    N.B. Does not ensure sampled indices are in the clip range.

    Parameters
    ----------
    gt_ext : list[int]
        Start and end frame indices of the GT interval, both inclusive.
    num_frames : int
        Number of frames to sample, by default 30.
    frame_interval : int
        Stride of sampled indices, by default 1.

    Returns
    -------
    np.ndarray
        Frame indices sampled from the GT interval.
    """
    required_len = (num_frames - 1) * frame_interval + 1

    # extend the GT interval if it is shorter than required
    raw_gt_len = gt_ext[1] - gt_ext[0] + 1
    if raw_gt_len < required_len:  # extend the GT interval
        len_short = required_len - raw_gt_len  # shortage of length
        ext_left = min(gt_ext[0], np.random.randint(len_short + 1))  # left extension
        ext_right = len_short - ext_left
        gt_ = [gt_ext[0] - ext_left, gt_ext[1] + ext_right]
        assert gt_[0] >= 0
    else:
        gt_ = gt_ext[:]  # deep copy

    # get num_frames + 1 temporal anchors from the extended GT interval, only left border is inclusive
    gt_len = gt_[1] - gt_[0] + 1
    assert gt_len >= required_len
    in_gt_offset = np.random.randint(gt_len - required_len + 1)  # ex) 1 if they are equal
    t_anchors = gt_[0] + in_gt_offset + np.linspace(0, required_len, num_frames + 1).astype(int)  # both inclusive

    # sample a frame idxs from each interval
    frame_idxs = np.array([np.random.randint(s, e) for s, e in zip(t_anchors, t_anchors[1:])])
    assert frame_idxs.shape[0] == num_frames
    assert (frame_idxs >= 0).all()
    assert ((gt_ext[0] <= frame_idxs) & (frame_idxs <= gt_ext[1])).any(), 'At least one frame should be in the GT interval.'
    # TODO: Add the clip length assertion
    return frame_idxs


def shift_indices_to_clip_range(
    frame_idxs: np.ndarray,
    clip_len: int,
):
    if isinstance(frame_idxs, list | tuple):
        frame_idxs = np.array(frame_idxs)
    # assert len(frame_idxs) < clip_len // 2, \
    #     f'The number of frames should be less than half of the clip length. {clip_len=} {len(frame_idxs)=}'  # half: chosen arbitrarily
    lmost = frame_idxs.min()
    rmost = frame_idxs.max()
    if clip_len < len(frame_idxs):
        frame_idxs = np.linspace(0, clip_len - 1, len(frame_idxs)).astype(int)
    else:
        if lmost < 0:
            frame_idxs = frame_idxs - lmost
        elif rmost >= clip_len:
            frame_idxs = frame_idxs - (rmost - clip_len + 1)

        lmost, rmost = frame_idxs.min(), frame_idxs.max()
        if lmost < 0 or rmost >= clip_len:
            frame_idxs = np.linspace(0, clip_len - 1, len(frame_idxs)).astype(int)

    assert (0 <= frame_idxs).all(), f'Negative frame indices: {frame_idxs}'
    assert (frame_idxs < clip_len).all(), f'Frame indices out of clip range: {frame_idxs}, {lmost=} {rmost=} {clip_len=}'
    # assert (0 <= frame_idxs).all() and (frame_idxs < clip_len).all()
    return frame_idxs


class VQ2DEvalDataset(VQ2DFitDataset):
    def __init__(self, config, split = 'val'):
        super().__init__(config, split)
        self.num_frames_per_segment = self.num_frames
        self.segment_length = self.frame_interval * self.num_frames_per_segment  # trailing stride is considered as occupied
        self.test_submit = split == 'test_unannotated'
        del self.num_frames  # to avoid confusion

        self.all_segments = []
        for ann_idx, ann in enumerate(self.all_anns):
            annotation_uid = ann['annotation_uid']
            query_set: str = ann['query_set']
            qset_uuid = f"{annotation_uid}_{query_set}"
            num_frames_clip = ann['query_frame']  # exclusive
            num_segments = np.ceil(num_frames_clip / self.segment_length).astype(int).item()
            seg_uuids = [f'{qset_uuid}_{seg_idx}' for seg_idx in range(num_segments)]
            for seg_idx in range(num_segments):
                self.all_segments.append({
                    'ann_idx': ann_idx,
                    'seg_idx': seg_idx,

                    'seg_uuid': seg_uuids[seg_idx],
                    'qset_uuid': qset_uuid,
                    'num_segments': num_segments,
                })

    def __len__(self):
        return len(self.all_segments)

    def __getitem__(self, idx):
        seg_info = self.all_segments[idx]
        ann_idx, seg_idx = seg_info['ann_idx'], seg_info['seg_idx']
        ann = self.all_anns[ann_idx]
        num_frames_clip = ann['query_frame']
        t = self.num_frames_per_segment
        frame_idxs = np.arange(seg_idx * t, (seg_idx + 1) * t, self.frame_interval)
        frame_idxs[frame_idxs >= num_frames_clip] = num_frames_clip - 1  # repeat

        segment = self.get_segment_frames(ann, frame_idxs)  # [t, c, h, w]
        query = self.get_query(ann)
        if self.test_submit:
            gt_rt, gt_prob = np.random.randn(t, 4), np.random.randn(t)
        else:
            gt_rt, gt_prob = self.get_response_track(ann, frame_idxs)  # prob as a binary mask
        segment, gt_rt = self.pad_and_resize(segment, gt_rt)  # [t, c, s, s], [t, 4]

        return {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # GT
            'gt_bboxes': gt_rt.astype(np.float32),  # [t, 4], yxyx, normalized
            'gt_probs': gt_prob.astype(np.float32),  # [t], GT prob
            'before_query_mask': torch.tensor(frame_idxs < ann['query_frame']).bool(),  # [t]

            # info
            'clip_uid': ann['clip_uid'],
            'seg_uuid': seg_info['seg_uuid'],
            'qset_uuid': seg_info['qset_uuid'],
            'seg_idx': seg_info['seg_idx'],
            'num_segments': seg_info['num_segments'],
            'original_height': ann['original_height'],
            'original_width': ann['original_width'],
            'frame_idxs': frame_idxs,
        }

    @staticmethod
    def testme():
        from omegaconf import OmegaConf
        config = OmegaConf.load('config/eval.yaml')
        config.dataset.clips_dir = '/local_datasets/ego4d_data/v2/vq2d_frames/raw'
        ds = VQ2DEvalDataset(config)
        torch.set_printoptions(linewidth=1000, precision=3, sci_mode=False)
        np.set_printoptions(linewidth=1000, precision=3, suppress=True)
        print(ds[0]['segment'].shape)
        print()
        print(ds[0]['query'].shape)
        print()
        print(ds[0]['gt_bboxes'].shape)
        print(ds[0]['gt_bboxes'])
        print()
        print(ds[0]['gt_probs'].shape)
        print(ds[0]['gt_probs'])
        print()
        print(ds[0]['before_query_mask'].shape)
        print(ds[0]['before_query_mask'])
        print()


if __name__ == '__main__':
    # python -m ltvu.dataset
    VQ2DEvalDataset.testme()
