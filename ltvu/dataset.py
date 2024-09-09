# built-in + hydra
import json
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
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
from einops import rearrange
from PIL import Image
from tqdm import tqdm

# local (ours)


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
        self.segment_size: tuple[int] = tuple(ds_config.segment_size)  # H, W
        self.query_size: tuple[int] = tuple(ds_config.query_size)  # H, W
        self.query_square: bool = ds_config.query_square
        self.query_padding: bool = ds_config.query_padding
        if ds_config.padding_value == 'mean':
            self.padding_value = .5
        elif ds_config.padding_value == 'zero':
            self.padding_value = 0.

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
        vc_idx = ann['visual_crop']['fno']
        p_clip_dir = self.p_clips_dir / clip_uid
        idxs_avail = set(int(p.stem.split('_')[-1]) - 1 for p in p_clip_dir.glob('*.jpg'))

        # get inputs
        gt_ext: list[int] = ann['response_track_valid_range']  # s, e both inclusive
        frame_idxs = self.sample_frame_idxs(self.num_frames, self.frame_interval, clip_len, gt_ext)
        idxs_required = set([*frame_idxs.tolist(), vc_idx])

        assert idxs_required.issubset(idxs_avail), \
            f'{clip_uid} does not have all required frames in {p_clip_dir}: {idxs_required - idxs_avail}'

        gt_rt, gt_prob = self.get_response_track(ann, frame_idxs)  # prob as a binary mask
        segment = self.get_segment_frames(ann, frame_idxs)
        segment, gt_rt = self.pad_and_resize(ann, segment, gt_rt)
        query = self.get_query(ann)

        return {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # GT
            'gt_bboxes': gt_rt,  # [t, 4], xyxy, normalized
            'gt_probs': gt_prob,  # [t], GT prob
            'before_query_mask': torch.tensor(frame_idxs < ann['query_frame']).bool(),  # [t], whether before the query frame, used for loss masking(?)

            # for logging
            'video_uid': ann['video_uid'],  # str
            'clip_uid': clip_uid,  # str
            'annotation_uid': ann['annotation_uid'],  # str
            'seg_idxs': frame_idxs,  # np.ndarray
            'query_set': ann['query_set'],  # str (of a single digit)
            'clip_fps': ann['clip_fps'],  # float
            'query_frame': ann['query_frame'],  # int
            'object_title': ann['object_title'],  # str
        }

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

    def pad_and_resize(self, ann: dict, frames: torch.Tensor, bboxes: np.ndarray):
        # frames: [t, c, h, w]
        # bboxes: [t, 4], xyxy, normalized
        t, c, h, w = frames.shape
        ow, oh = ann['original_width'], ann['original_height']
        bboxes *= [ow, oh, ow, oh]  # de-normalize

        # pad
        pad_size = (w - h) // 2
        pad_top, pad_bot = pad_size, w - h - pad_size
        pad = (0, 0, pad_top, pad_bot)   # Left, Right, Top, Bottom
        frames = F.pad(frames, pad, value=self.padding_value)
        bboxes[:, [1, 3]] += pad_top
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
        ow, oh = ann['original_height'], ann['original_width']
        fno = vc['fno']
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
        ooh, oow = query.size  # might be pre-pre-processed already
        rho = (oh / ooh + ow / oow) / 2
        x, y, w, h = x / rho, y / rho, h / rho, w / rho
        query = query.crop((x, y, x + w, y + h)) # [y:y+h, x:x+w]  # [h, w, c]
        query = TF.pil_to_tensor(query)  # [c, h, w]
        query = query.float() / 255.

        # permute - pad - resize
        # query = rearrange(query, 'h w c -> c h w').contiguous()
        if self.query_padding:
            pad_size = (l - s) // 2
            if h > w:
                pad = (pad_size, l - s - pad_size, 0, 0)   # Left, Right, Top, Bottom
            else:
                pad = (0, 0, pad_size, l - s - pad_size)   # Left, Right, Top, Bottom
            query = F.pad(query, pad, value=0)
        query = F.interpolate(query[None], size=self.query_size, mode='bilinear', align_corners=True, antialias=True)
        return query.squeeze(0)  # [c, h, w]

    def get_response_track(self, ann: dict, seg_idxs: np.ndarray):
        """_summary_

        Parameters
        ----------
        ann : dict
            _description_
        seg_idxs : np.ndarray
            Frame indices of the segment (a part of the clip).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            bboxes: [t, 4], xyxy, normalized
            seg_with_gt: [t], bool, whether the segment contains GT bbox
        """
        ow, oh = ann['original_width'], ann['original_height']
        gt_rt = ann['response_track']
        gt_ext = ann['response_track_valid_range']

        # initialize bboxes with default values
        seg_with_gt = (seg_idxs <= gt_ext[1]) & (seg_idxs >= gt_ext[0])
        default_bbox = [0, 0, 1e-6, 1e-6]  # xyxy, normalized
        bboxes = np.array([default_bbox] * self.num_frames)  # [t, 4]

        # update bboxes with GT values
        valid_bboxes = []
        for res in gt_rt:
            if res['fno'] in seg_idxs:
                num_repeat = (res['fno'] == seg_idxs).sum()  # segment frames have duplicates whenever clip_len < num_frames
                x, y, w, h = res['x'], res['y'], res['w'], res['h']
                x1, y1, x2, y2 = x, y, x + w, y + h
                for _ in range(num_repeat):
                    valid_bboxes.append([x1, y1, x2, y2])

        assert len(valid_bboxes) == seg_with_gt.sum(), f'GT bbox length mismatch: {len(valid_bboxes)=} vs {seg_with_gt.sum()=}'
        bboxes[seg_with_gt] = np.array(valid_bboxes)
        bboxes = np.array(bboxes) / [ow, oh, ow, oh]  # normalize
        bboxes = bboxes.astype(np.float32).clip(0, 1)

        return bboxes, seg_with_gt.astype(np.float32)


class VQ2DTestDatasetSeparated(VQ2DFitDataset):
    """An item is defined as a tuple `(qset_uuid, seg_idx)`."""

    def __init__(self, config, split = 'val'):
        super().__init__(config, split)
        # self.all_clip_uids = sorted(set([ann['clip_uid'] for ann in self.all_anns]))
        segment_length = (self.num_frames - 1) * self.frame_interval + 1
        self.all_seg_uids = []
        for ann in self.all_anns:
            clip_uid = ann['clip_uid']
            annotation_uid = ann['annotation_uid']
            query_set: str = ann['query_set']
            qset_uuid = f"{annotation_uid}_{query_set}"
            num_frames_clip = ann['query_frame']
            num_segments = np.ceil(num_frames_clip / segment_length).astype(int).item()
            # (qset_uuid, seg_idx) tuple for indexing, clip_uid for loggging
            self.all_seg_uids.extend([(qset_uuid, seg_idx, clip_uid) for seg_idx in range(num_segments)])

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

        segment = self.get_segment_frames(ann, vr)
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
    if clip_len < len(frame_idxs):
        frame_idxs = np.linspace(0, clip_len - 1, len(frame_idxs)).astype(int)
    else:
        lmost = frame_idxs.min()
        rmost = frame_idxs.max()
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
