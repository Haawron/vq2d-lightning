import json
from pathlib import Path

from omegaconf import DictConfig

import numpy as np
import torch

from ltvu.dataset import VQ2DFitDataset, shift_indices_to_clip_range



class EgoTracksDataset(VQ2DFitDataset):
    ann_keys = [
        'video_uid',
        'clip_uid',
        'query_set',                    # str, annotation enumeration, 1-based
        'clip_fps',
        'object_title',                 # str, object name (e.g. 'sellotape'), don't use this for training
        'visual_crop',                  # actual query
        'uuid_ltt',
    ]
    def __init__(self, config: DictConfig, split: str = 'train'):
        torch.utils.data.Dataset.__init__(self)
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
        if split == 'test':
            split = 'challenge_test_unannotated'
        self.split = split
        self.p_ann = self.p_anns_dir / f'egotracks_{split}_anno.json'
        self.all_anns = json.load(self.p_ann.open())
        self.all_anns = self.subsample_anns(self.all_anns)

    def subsample_anns(self, anns):
        anns = [ann for ann in anns if 'clip_uid' in ann]
        return anns

    def get_lt_track(self, ann, frame_idxs):
        oh, ow = ann['original_height'], ann['original_width']
        lt_track = ann['lt_track']

        fno2bbox = {bbox['fno']: bbox for bbox in lt_track}

        fnos = {bbox['fno'] for bbox in lt_track}
        bboxes = []
        for fno in frame_idxs:
            if fno in fnos:
                bbox = fno2bbox[fno]
                bbox = [bbox['y'], bbox['x'], bbox['y'] + bbox['h'], bbox['x'] + bbox['w']]
                bboxes.append(bbox)
            else:
                bboxes.append([0, 0, 1e-5, 1e-5])

        # normalize
        bboxes = np.array(bboxes, dtype=np.float32) / [oh, ow, oh, ow]
        bboxes = np.clip(bboxes, 0, 1)

        return bboxes



class EgoTracksFitDataset(EgoTracksDataset):
    ann_keys = [
        *EgoTracksDataset.ann_keys,
        'query_frame',                  # int, the end of the extent of the input clip (nothing to do with the query)
        'response_track_valid_range',   # both inclusive
        'response_track',
        'visual_clip',
        'lt_track'
    ]

    def __getitem__(self, idx):
        ann = self.all_anns[idx]
        if 'lt_track' not in ann:
            assert 'response_track' in ann
            ann['annotation_uid'] = ''  # for compatibility
            return super().__getitem__(idx)
        else:
            clip_uid = ann['clip_uid']
            lt_track = ann['lt_track']
            clip_len = int(ann['clip_duration'] * ann['clip_fps'])
            vc = ann['visual_crop']
            vc_idx = vc['fno']
            p_clip_dir = self.p_clips_dir / clip_uid
            idxs_avail = set(int(p.stem.split('_')[-1]) - 1 for p in p_clip_dir.glob('*.jpg'))

            # get inputs
            required_len = (self.num_frames - 1) * self.frame_interval + 1
            lt_track_frame_idxs = np.array([bbox['fno'] for bbox in lt_track])
            gt_mask = np.zeros(clip_len)
            gt_mask[lt_track_frame_idxs] = 1
            num_forward_gt_frames = np.convolve(gt_mask, np.ones(required_len), mode='valid')
            start = np.random.choice(np.where(num_forward_gt_frames > 0)[0])

            frame_idxs = np.arange(start, start + required_len, self.frame_interval)
            gt_mask = gt_mask[frame_idxs]
            assert gt_mask.any(), f'Sampled frames do not contain GT frames: {frame_idxs}'

            idxs_required = set([*frame_idxs.tolist(), vc_idx])
            assert idxs_required.issubset(idxs_avail), \
                f'{clip_uid} does not have all required frames in {p_clip_dir}: {idxs_required - idxs_avail}'

            segment = self.get_segment_frames(ann, frame_idxs)  # [t, c, h, w]
            gt_ltt = self.get_lt_track(ann, frame_idxs)  # prob as a binary mask

            if self.rt_pos_query is not None and self.split == 'train':
                rt_pos_queries, rt_pos_idx = self.get_rt_pos_query(ann, frame_idxs)

            segment, gt_ltt = self.pad_and_resize(segment, gt_ltt)  # [t, c, s, s], [t, 4]
            query = self.get_query(ann)

            sample = {
                # inputs
                'segment': segment,  # [t, c, h, w], normalized
                'query': query,  # [c, h, w], normalized

                # GT
                'gt_bboxes': gt_ltt.astype(np.float32),  # [t, 4], yxyx, normalized
                'gt_probs': gt_mask.astype(np.float32),  # [t], GT prob
                'before_query_mask': torch.tensor(gt_mask).bool(),  # [t], the key name is misleading due to the legacy code

                # for logging
                'video_uid': ann['video_uid'],  # str
                'clip_uid': clip_uid,  # str
                'annotation_uid': '',
                'seg_idxs': frame_idxs,  # np.ndarray
                'query_set': ann['query_set'],  # str (of a single digit)
                'clip_fps': ann['clip_fps'],  # float
                'query_frame': 999999,  # int
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


class EgoTracksEvalDataset(EgoTracksDataset):
    def __init__(self, config, split = 'val'):
        super().__init__(config, split)
        self.num_frames_per_segment = self.num_frames
        self.segment_length = self.frame_interval * self.num_frames_per_segment  # trailing stride is considered as occupied
        self.test_submit = split == 'challenge_test_unannotated'
        del self.num_frames  # to avoid confusion

        self.all_segments = []
        for ann_idx, ann in enumerate(self.all_anns):
            ltt_uuid = ann['uuid_ltt']
            num_frames_clip = int(ann['clip_duration'] * ann['clip_fps'])
            num_segments = np.ceil(num_frames_clip / self.segment_length).astype(int).item()
            seg_uuids = [f'{ltt_uuid}_{seg_idx}' for seg_idx in range(num_segments)]
            for seg_idx in range(num_segments):
                self.all_segments.append({
                    'ann_idx': ann_idx,
                    'seg_idx': seg_idx,

                    'seg_uuid': seg_uuids[seg_idx],
                    'qset_uuid': ltt_uuid,
                    'num_segments': num_segments,
                })

    def __len__(self):
        return len(self.all_segments)

    def __getitem__(self, idx):
        seg_info = self.all_segments[idx]
        ann_idx, seg_idx = seg_info['ann_idx'], seg_info['seg_idx']
        ann = self.all_anns[ann_idx]
        num_frames_clip = int(ann['clip_duration'] * ann['clip_fps'])
        t = self.num_frames_per_segment
        frame_idxs = np.arange(seg_idx * t, (seg_idx + 1) * t, self.frame_interval)
        frame_idxs[frame_idxs >= num_frames_clip] = num_frames_clip - 1  # repeat

        segment = self.get_segment_frames(ann, frame_idxs)  # [t, c, h, w]
        query = self.get_query(ann)
        if self.test_submit:
            gt_ltt = np.array([[0, 0, 1e-5, 1e-5]] * t, dtype=np.float32)
            gt_prob = np.zeros(t, dtype=np.float32)
        else:
            gt_ltt = self.get_lt_track(ann, frame_idxs)
            gt_prob = None
            raise NotImplementedError
        segment, gt_ltt = self.pad_and_resize(segment, gt_ltt)  # [t, c, s, s], [t, 4]

        return {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # # GT
            'gt_bboxes': gt_ltt.astype(np.float32),  # [t, 4], yxyx, normalized
            'gt_probs': gt_prob.astype(np.float32),  # [t], GT prob
            'before_query_mask': torch.tensor(gt_prob).bool(),  # [t]

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
