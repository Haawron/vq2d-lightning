# built-in + hydra
import json
from pathlib import Path
from omegaconf import DictConfig

# torch
import torch
import torch.utils.data
from torch.nn import functional as F

# lightning
# import lightning as L

# others
import decord
import numpy as np
from einops import rearrange

# local (ours)


decord.bridge.set_bridge("torch")
P_CLIPS_DIR_FOR_CHECKING_VALIDITY = Path('/data/datasets/ego4d_data/v2/vq2d_clips_448')
REPORTED_INVALID_CLIP_UIDS = {
    'b6061527-3ae2-46b4-b050-15eddc1967bb',  # vq2d
}


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
        self.p_clips_dir = Path(ds_config.clips_dir)
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
        self.p_anns_dir = Path('./data/')
        self.p_ann = self.p_anns_dir / f'vq_v2_{split}_anno.json'
        self.all_anns = json.load(self.p_ann.open())
        self.all_anns = self.subsample_anns(self.all_anns)

    def __len__(self):
        return len(self.all_anns)

    def __getitem__(self, idx):
        # setup
        ann: dict = self.all_anns[idx]
        clip_uid = ann['clip_uid']
        p_clip = self.p_clips_dir / f'{clip_uid}.mp4'
        vr = decord.VideoReader(str(p_clip), num_threads=1)

        # get inputs
        gt_ext: list[int] = ann['response_track_valid_range']  # s, e both inclusive
        frame_idxs = self.sample_frame_idxs(self.num_frames, self.frame_interval, gt_ext)
        segment, seg_idxs = self.get_segment_frames(ann, vr, frame_idxs)
        gt_rt, gt_prob = self.get_response_track(ann, seg_idxs)  # prob as a binary mask
        query = self.get_query(ann, vr)

        return {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # GT
            'gt_bboxes': gt_rt,  # [t, 4], xyxy, normalized
            'gt_probs': gt_prob,  # [t], GT prob
            'before_query_mask': torch.tensor(seg_idxs < ann['query_frame']).bool(),  # [t], whether before the query frame, used for loss masking(?)

            # for logging
            'video_uid': ann['video_uid'],  # str
            'clip_uid': clip_uid,  # str
            'annotation_uid': ann['annotation_uid'],  # str
            'seg_idxs': seg_idxs,  # np.ndarray
            'query_set': ann['query_set'],  # str (of a single digit)
            'clip_fps': ann['clip_fps'],  # float
            'query_frame': ann['query_frame'],  # int
            'object_title': ann['object_title'],  # str
        }

    def subsample_anns(self, anns):  # interface
        return anns

    def sample_frame_idxs(self, num_frames: int, frame_interval: int, gt_ext = None):
        return sample_nearby_gt_frames(gt_ext, num_frames, frame_interval)

    def get_segment_frames(self, ann, vr, frame_idxs):
        """
        4 steps to get an input clip:
            1. sample
            2. load - normalize - permute
            3. pad or crop - resize
            (4. augment -> not here, GPU accelerated by kornia)
        """
        # sample
        vlen = len(vr)
        origin_fps = int(vr.get_avg_fps())
        gt_fps = int(ann['clip_fps'])
        down_rate = origin_fps // gt_fps
        frame_idxs_origin = np.minimum(down_rate * frame_idxs, vlen - 1)  # mp4 file's FPS considered

        # load - normalize - permute
        frames = vr.get_batch(frame_idxs_origin)
        frames = frames.float() / 255.
        frames = rearrange(frames, 't h w c -> t c h w')
        t, c, h, w = frames.shape
        assert h <= w  # Check: if Ego4D clips are all in landscape mode

        # pad - resize
        pad_size = (w - h) // 2
        pad = (0, 0, pad_size, w - h - pad_size)   # Left, Right, Top, Bottom
        frames = F.pad(frames, pad, value=self.padding_value)
        _, _, h_pad, w_pad = frames.shape
        assert h_pad == w_pad, f'Padded frames should be square, got {frames.shape}'
        frames = F.interpolate(frames, size=self.segment_size, mode='bilinear', align_corners=True, antialias=True)

        return frames, frame_idxs

    def get_query(self, ann, vr: decord.VideoReader):
        vc = ann['visual_crop']
        ow, oh = ann['original_height'], ann['original_width']
        fno = vc['fno']
        x, y, w, h = vc['x'], vc['y'], vc['w'], vc['h']
        l, s = max(w, h), min(w, h)  # large short

        if self.query_square:  # but don't have to be strictly square, will be resized at the end of this function
            cx, cy, s = x + w / 2, y + h / 2, np.clip(l, a_min=10, a_max=min(oh, ow)-1).item()
            cx, cy = np.clip(cx, s / 2, ow - s / 2 - 1).item(), np.clip(cy, s / 2, oh - s / 2 - 1).item()
            x, y, w, h = cx - s / 2, cy - s / 2, s, s
            assert 0 <= x < ow and 0 <= y < oh and 0 < x + w < ow and 0 < y + h < oh, \
                f'Invalid visual crop: {x=}, {y=}, {h=}, {w=}, {oh=}, {ow=}'

        ooh, oow, _ = vr[0].shape  # might be pre-pre-processed already
        rho = (oh / ooh + ow / oow) / 2
        x, y, w, h = int(x / rho), int(y / rho), int(h / rho), int(w / rho)

        query: torch.Tensor = vr[fno]

        query = query.float() / 255.
        query = query[y:y+h, x:x+w]  # [h, w, c]
        query = rearrange(query, 'h w c -> c h w').contiguous()

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
        bboxes = np.array([default_bbox] * self.num_frames)

        # update bboxes with GT values
        valid_bboxes = [
            [res['x'], res['y'], res['x'] + res['w'], res['y'] + res['h']]
            for res in gt_rt if res['fno'] in seg_idxs
        ]  # xyxy
        assert len(valid_bboxes) == seg_with_gt.sum(), f'GT bbox length mismatch: {len(valid_bboxes)=} vs {seg_with_gt.sum()=}'
        bboxes[seg_with_gt] = np.array(valid_bboxes)
        bboxes = np.array(bboxes) / [ow, oh, ow, oh]  # normalize
        bboxes = bboxes.astype(np.float32).clip(0, 1)

        return torch.tensor(bboxes), torch.tensor(seg_with_gt).float()


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

        segment, _ = self.get_segment_frames(ann, vr)
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
    gt_interval: list[int],  # both inclusive
    num_frames: int = 30,
    frame_interval: int = 1,
) -> np.ndarray:
    required_len = (num_frames - 1) * frame_interval + 1

    # extend the GT interval if it is shorter than required
    raw_gt_len = gt_interval[1] - gt_interval[0] + 1
    if raw_gt_len < required_len:  # extend the GT interval
        len_short = required_len - raw_gt_len  # shortage of length
        ext_left = min(gt_interval[0], np.random.randint(len_short + 1))  # left extension
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
    assert ((gt_interval[0] <= frame_idxs) & (frame_idxs <= gt_interval[1])).any(), 'At least one frame should be in the GT interval.'
    # TODO: Add the clip length assertion
    return frame_idxs


def generate_flat_annotations_vq2d(all_anns):
    assert P_CLIPS_DIR_FOR_CHECKING_VALIDITY is not None, 'P_CLIPS_FOR_CHECKING_VALIDITY is not set.'

    def polish_bbox_dict(bbox: dict):
        key_map = {
            'frame_number': 'fno',
            'x': 'x', 'y': 'y', 'width': 'w', 'height': 'h',
            'original_width': None, 'original_height': None}
        return {key_map[k]: v for k, v in bbox.items() if key_map.get(k) is not None}

    flat_anns = []
    count_invalids = 0
    for ann_video in all_anns['videos']:
        video_uid = ann_video['video_uid']
        for ann_clip in ann_video['clips']:
            clip_uid = ann_clip['clip_uid']
            if clip_uid is None or clip_uid in REPORTED_INVALID_CLIP_UIDS:
                continue
            clip_duration = ann_clip['video_end_sec'] - ann_clip['video_start_sec']
            clip_fps = ann_clip['clip_fps']
            _p_clip = P_CLIPS_DIR_FOR_CHECKING_VALIDITY / f'{clip_uid}.mp4'
            is_invalid_clip = all(not qset['is_valid'] for ann_annots in ann_clip['annotations'] for qset in ann_annots['query_sets'].values())
            assert _p_clip.exists() or is_invalid_clip, f'Clip {clip_uid} is a valid clip but does not exist in {_p_clip.parent}.'
            for ann_annots in ann_clip['annotations']:
                annotation_uid = ann_annots['annotation_uid']
                for qset_id, qset in ann_annots['query_sets'].items():
                    if not qset['is_valid']:
                        count_invalids += 1
                        continue
                    oh, ow = qset['response_track'][0]['original_height'], qset['response_track'][0]['original_width']
                    rt = [polish_bbox_dict(bbox) for bbox in qset['response_track']]
                    sample = {
                        'video_uid': video_uid,
                        'clip_uid': clip_uid,
                        'annotation_uid': annotation_uid,
                        'query_set': qset_id,
                        'clip_fps': clip_fps,
                        'clip_duration': clip_duration,
                        'original_width': ow,
                        'original_height': oh,
                        'query_frame': qset['query_frame'],
                        'object_title': qset['object_title'],
                        'visual_crop': polish_bbox_dict(qset['visual_crop']),
                        'response_track_valid_range': [rt[0]['fno'], rt[-1]['fno']],
                        'response_track': rt,
                    }
                    flat_anns.append(sample)
    return flat_anns
