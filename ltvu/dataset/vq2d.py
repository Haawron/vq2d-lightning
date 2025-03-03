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
# import decord
import numpy as np
from PIL import Image
import random

# local (ours)

# decord.bridge.set_bridge("torch")


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

    def __init__(self, config: DictConfig, split: str = 'train', movement: str = ""):
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
        self.frame_dash_aug = self.config.dataset.get('frame_dash_aug')
        self.frame_dash_rate = self.config.dataset.get('frame_dash_rate')
        self.frame_stride = self.config.dataset.get('frame_stride')
        self.frame_incremental = self.config.dataset.get('frame_incremental')
        self.frame_incremental_level = 0
        self.frame_box_aug = ds_config.get('frame_box_aug')
        self.frame_random = ds_config.get('frame_random', False)
        self.aug_based_time = ds_config.get('aug_based_time', False)
        self.frame_neighbor = ds_config.get('frame_neighbor', False)
        self.frame_shift = ds_config.get('frame_shift', False)
        self.gt_consider = ds_config.get('gt_consider', False)
        self.box_aug = ds_config.get('box_aug', False)
        self.aug_time = ds_config.get('aug_time', False)
        self.compare_clip_penalty = ds_config.get('compare_clip_penalty', False)
        self.reverse_clip_penalty = ds_config.get('reverse_clip_penalty', False)
        self.box_aug_mode = None
        self.split = split
        self.movement = movement
        if ds_config.get('box_aug'):
            self.frame_box_aug = True
        if movement != "":
            assert movement in ['slow', 'medium', 'fast', 'slow2', 'medium2', 'fast2'], f'Invalid movement: {movement}'
            self.p_ann = self.p_anns_dir / f'vq_v2_{split}_{movement}_anno.json'
        else:
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

        query = self.get_query(ann)
        segment, gt_rt, gt_rt_ori = self.pad_and_resize(segment, gt_rt)  # [t, c, s, s], [t, 4]
        
        reorder_idxs = np.arange(0, self.num_frames)
        if self.split == 'train' and (self.frame_dash_aug or self.frame_box_aug or self.compare_clip_penalty):
            segment, gt_rt, gt_prob, before_delta, after_delta, reorder_idxs = self.frame_aug(segment, gt_rt, gt_rt_ori, gt_prob)

        sample = {
            # inputs
            'segment': segment,  # [t, c, h, w], normalized
            'query': query,  # [c, h, w], normalized

            # GT
            'gt_bboxes': gt_rt.astype(np.float32),  # [t, 4], yxyx, normalized
            'gt_probs': gt_prob.astype(np.float32),  # [t], GT prob
            'before_query_mask': torch.tensor(frame_idxs < ann['query_frame']).bool(),  # [t], whether before the query frame, used for loss masking(?)
            'reorder_idxs': torch.tensor(reorder_idxs),  # [t], normalized

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
            
        if self.split == 'train' and (self.frame_dash_aug or self.frame_box_aug):
            (sample
                .setdefault('experiment', {})
                .setdefault('frame_aug', {}))
            sample['experiment']['frame_aug']['before_delta'] = before_delta
            sample['experiment']['frame_aug']['after_delta'] = after_delta
            sample['experiment']['frame_aug']['difference'] = after_delta - before_delta

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
        num_clip_frames = int(ann['clip_fps'] * int(ann['clip_duration']))
        frame_idxs = frame_idxs.clip(0, num_clip_frames - 1)

        # load - normalize - permute
        last_idx = max([int(p.stem.split('_')[-1]) for p in p_clip.glob('*.jpg')])
        p_frames = [
            p if (p := p_clip / f'frame_{idx+1:07d}.jpg').exists() else p_clip / f'frame_{last_idx:07d}.jpg'
            for idx in frame_idxs
        ]
        frames = [Image.open(p) for p in p_frames]
        frames = torch.stack([TF.pil_to_tensor(f) for f in frames])  # [t, c, h, w]
        frames = frames.float() / 255.
        t, c, h, w = frames.shape
        assert h <= w, f'All the videos in Ego4D are landscape, got {ann["clip_uid"]}, {frames.shape=}'

        return frames
    
    def frame_aug(self, segment, gt_rt, gt_rt_ori, gt_prob):
        reorder_idxs = np.arange(0, self.num_frames)
        if self.split == 'train':
            dash_rate, box_rate, reverse_rate = 0, 0, 0
            before_delta, after_delta = 0, 0
            if self.compare_clip_penalty:
                if self.frame_box_aug and self.frame_dash_aug and not self.reverse_clip_penalty:
                    box_rate, dash_rate, reverse_rate = 0.6, 0.4, 0.
                elif self.frame_box_aug and not self.frame_dash_aug and self.reverse_clip_penalty:
                    box_rate, dash_rate, reverse_rate = 0.6, 0., 0.4
                elif self.frame_box_aug and not self.frame_dash_aug and not self.reverse_clip_penalty:
                    box_rate = 1
                elif not self.frame_box_aug and self.frame_dash_aug and not self.reverse_clip_penalty:
                    dash_rate = 1
                elif not self.frame_box_aug and not self.frame_dash_aug and self.reverse_clip_penalty:
                    reverse_rate = 0
                else:
                    assert False, 'Invalid frame augmentation configuration.'
                
                random_rate = random.random()
                if random_rate <= dash_rate:
                    if self.frame_incremental_level >= 2:
                        frame_stride = 2 if random.random() < 0.7 else 3
                    else:
                        frame_stride = 2
                    segment, gt_rt, gt_prob, reorder_idxs = self.frame_dash(segment, gt_rt, gt_prob, frame_stride)
                elif random_rate <= dash_rate + box_rate:
                    segment, gt_rt, before_delta, after_delta, gt_prob, reorder_idxs = self.frame_box(segment, gt_rt, gt_rt_ori, gt_prob)
                else:
                    reorder_idxs = np.array(reorder_idxs[::-1])
                    segment = segment[reorder_idxs]
                    gt_rt = gt_rt[reorder_idxs]
                    gt_prob = gt_prob[reorder_idxs]
                    
            elif not self.frame_incremental:
                if self.frame_dash_aug and random.random() < self.frame_dash_rate:
                    segment, gt_rt, gt_prob, reorder_idxs = self.frame_dash(segment, gt_rt, gt_prob, self.frame_stride)
                if self.frame_box_aug and random.random() < 0.5:
                    if self.frame_random:
                        gt_idx = np.where(gt_prob == 1)[0]
                        gt_idx_shuffled = np.random.permutation(gt_idx)
                        segment[gt_idx] = segment[gt_idx_shuffled]
                        gt_rt[gt_idx] = gt_rt[gt_idx_shuffled]
                        reorder_idxs[gt_idx] = reorder_idxs[gt_idx_shuffled]
                    else:
                        segment, gt_rt, before_delta, after_delta, gt_prob, re_reorder_idxs = self.frame_box(segment, gt_rt, gt_rt_ori, gt_prob)
                        reorder_idxs = reorder_idxs[re_reorder_idxs]
            elif self.frame_incremental:
                if self.frame_incremental_level == 0:
                    total_rate = 0.2
                    frame_stride = 2
                elif self.frame_incremental_level == 1:
                    total_rate = 0.4
                    frame_stride = 2
                elif self.frame_incremental_level == 2:
                    total_rate = 0.6
                    frame_stride = 2 if random.random() < 0.7 else 3
                elif self.frame_incremental_level == 3:
                    total_rate = 0.6
                    frame_stride = 2 if random.random() < 0.5 else 3
                    
                if self.frame_dash_aug and self.frame_box_aug:
                    dash_rate = total_rate / 2
                    box_rate = total_rate / 2 + 0.1
                elif self.frame_box_aug:
                    box_rate = total_rate
                elif self.frame_dash_aug:
                    dash_rate = total_rate
                    
                random_rate = random.random()
                if random_rate <= dash_rate:
                    segment, gt_rt, gt_prob, reorder_idxs = self.frame_dash(segment, gt_rt, gt_prob, frame_stride)
                elif random_rate <= dash_rate + box_rate:
                    segment, gt_rt, before_delta, after_delta, gt_prob, reorder_idxs = self.frame_box(segment, gt_rt, gt_rt_ori, gt_prob)
                    
        return segment, gt_rt, gt_prob, before_delta, after_delta, reorder_idxs
    
    def frame_dash(self, segment, gt_rt, gt_prob, frame_stride):
        num_frames = self.num_frames 
        new_order = np.arange(0, num_frames)

        if frame_stride == 2:
            forward = np.arange(0, num_frames, frame_stride)
            remaining = np.setdiff1d(np.arange(num_frames), forward, assume_unique=True)
            backward = remaining[::-1]

            new_order = np.concatenate([forward, backward])

        elif frame_stride == 3:
            forward = np.arange(0, num_frames, frame_stride)
            remaining = np.setdiff1d(np.arange(num_frames), forward, assume_unique=True)

            backward = remaining[::-1][::frame_stride - 1]
            remaining = np.setdiff1d(remaining, backward, assume_unique=True)

            third_pass = remaining

            new_order = np.concatenate([forward, backward, third_pass])

        else:
            raise ValueError("frame_stride must be either 2 or 3.")

        segment = segment[new_order]
        gt_rt = gt_rt[new_order]
        gt_prob = gt_prob[new_order]

        return segment, gt_rt, gt_prob, new_order
    
    def frame_box(self, segment, gt_rt, gt_rt_ori, gt_prob):
        reorder_idxs = np.arange(0, self.num_frames)
        box_aug_data = self.get_box_aug(segment, gt_rt, gt_rt_ori, gt_prob)
        before_delta = box_aug_data['before_delta']
        if self.box_aug_mode == 'diff':
            segment, gt_rt = box_aug_data['aug_segment_diff'], box_aug_data['aug_gt_rt_diff']
            after_delta = box_aug_data['after_diff_delta']
            reorder_idxs = box_aug_data['reorder_idxs_diff']
            gt_prob = box_aug_data['aug_gt_prob_diff']
        elif self.box_aug_mode == 'easy':
            segment, gt_rt = box_aug_data['aug_segment_easy'], box_aug_data['aug_gt_rt_easy']
            after_delta = box_aug_data['after_easy_delta']
            reorder_idxs = box_aug_data['reorder_idxs_easy']
            gt_prob = box_aug_data['aug_gt_prob_easy']
        elif self.box_aug_mode == None:
            after_delta = 0
            pass
        return segment, gt_rt, before_delta, after_delta, gt_prob, reorder_idxs


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

        gt_rt_ori = bboxes.copy()
        # normalize
        bboxes /= hw_pad

        return frames, bboxes, gt_rt_ori

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
   
    def get_box_aug(self, segment, gt_rt, gt_rt_ori, gt_prob):
        if self.gt_consider:
            gt_idx = np.arange(0, self.num_frames)
            gt_idx = gt_idx.astype(int)
        else:
            gt_idx = np.where(gt_prob == 1)[0]
        if self.aug_based_time and len(gt_idx) > self.aug_time:
            gt_idx_rand = np.random.choice(gt_idx, self.aug_time, replace=False)
        else:
            gt_idx_rand = gt_idx

        aug_segment_diff = segment.clone()
        aug_gt_rt_diff = gt_rt.copy()
        aug_segment_easy = segment.clone()
        aug_gt_rt_easy = gt_rt.copy()
        gt_rt_original = gt_rt_ori.copy()
        gt_box = gt_rt_original[gt_idx]
        num_boxes = len(gt_box)
        before_delta, after_diff_delta, after_easy_delta = 0, 0, 0
        aug_gt_prob_diff = gt_prob.copy()
        aug_gt_prob_easy = gt_prob.copy()
        
        reorder_idxs_easy = np.arange(0, self.num_frames)
        reorder_idxs_diff = np.arange(0, self.num_frames)
        
        if not (len(gt_box) <=2):
            # Compute total change (delta)
            delta_total = compute_bbox_deltas(gt_box)
            original_delta_total = delta_total.copy()
            
            def reorder_boxes(mode="diff"):
                """
                Reorders boxes based on either the most different (farthest) or most similar (nearest).
                
                Args:
                    delta_matrix (np.ndarray): The distance or difference matrix.
                    mode (str): "diff" for most different ordering, "easy" for most similar ordering.
                
                Returns:
                    list: New ordering indices.
                """
                new_order = [0]  # Start with the first box
                remaining_indices = list(range(1, num_boxes))

                for _ in range(1, num_boxes):
                    last_idx = new_order[-1]
                    if mode == "diff":
                        selected_idx = remaining_indices[np.argmax(delta_total[last_idx, remaining_indices])]
                    else:  # mode == "easy"
                        selected_idx = remaining_indices[np.argmin(delta_total[last_idx, remaining_indices])]
                    
                    new_order.append(selected_idx)
                    remaining_indices.remove(selected_idx)

                return new_order
            
            def reorder_boxes_for_time(mode="diff"):
                new_order = []
                used_idx = set()
                in_gt_idx_rand = np.where(np.isin(gt_idx, gt_idx_rand))[0]
                for i, gt_i in enumerate(gt_idx):
                    if i in in_gt_idx_rand:
                        excluded_indices = np.concatenate((list(used_idx), np.where(np.isin(gt_idx, gt_idx_rand))[0]))  
                        valid_indices = np.setdiff1d(np.arange(len(delta_total[i])), excluded_indices)
                        
                        if len(valid_indices) > 0:
                            if mode == "diff":
                                best_match = valid_indices[np.argmax(delta_total[i][valid_indices])]
                            else:  # mode == "easy"
                                best_match = valid_indices[np.argmin(delta_total[i][valid_indices])]
                        else:
                            best_match = -1  # 선택할 값이 없으면 -1
                            
                        new_order.append(i)
                        if best_match != -1:
                            new_order.append(best_match)
                            used_idx.add(best_match)
                    elif i in in_gt_idx_rand + 1:
                        continue
                    else:
                        new_order.append(i)
                return new_order
            
            def reorder_boxes_for_time_neighbor(gt_idx_rand, mode="diff"):
                
                new_order = [i for i in range(len(gt_idx))]
                delta = delta_total[gt_idx_rand]
                
                if mode == "diff":
                    best_match = np.argmax(delta)
                elif mode == "easy":
                    best_match = np.argmin(delta)
                
                next_idx = gt_idx_rand + 1 if gt_idx_rand + 1 < len(gt_idx) and gt_idx_rand + 2 < len(gt_idx) else None

                if next_idx and best_match != next_idx + 1 and best_match != gt_idx_rand - 1:
                    original_future_change = delta_total[next_idx][next_idx + 1]
                    new_future_change = delta_total[best_match][next_idx + 1]
                    if new_future_change < original_future_change * 0.5:
                        new_order = np.insert(new_order, next_idx, best_match)
                        new_order = np.delete(new_order, best_match + 1)
                    else:
                        if best_match - 1 >= 0:
                            original_change2 = delta_total[best_match][best_match - 1] 
                            new_future_change2 = delta_total[gt_idx_rand][best_match - 1]
                            
                            tmp = new_order[next_idx]
                            new_order[next_idx] = best_match
                            if new_future_change2 >= original_change2 * 0.5:
                                new_order[best_match] = tmp           
                return new_order
            
            def reorder_boxes_shift(rand_idx, gt_idx, gt_rt_original, mode="diff"):
                new_order = []
                delta_list = []
                for num in gt_idx:
                    if num == gt_idx[rand_idx]:
                        continue
                    tmp_gt_idx = gt_idx.copy()
                    tmp_gt_idx = np.insert(tmp_gt_idx, rand_idx, num)
                    valid_idx = np.where(tmp_gt_idx == num)[0]
                    valid_idx = valid_idx[valid_idx != rand_idx]
                    tmp_gt_idx = np.delete(tmp_gt_idx, valid_idx)
                    new_order.append(tmp_gt_idx - tmp_gt_idx.min())
                    
                    tmp_gt_rt_ori = gt_rt_original.copy()
                    delta = compute_bbox_deltas(tmp_gt_rt_ori[tmp_gt_idx])
                    delta_list.append(np.mean(np.diagonal(delta, offset=1)))

                max_idx = np.argmax(delta_list).tolist()
                
                return new_order[max_idx]
            
            # Compute new orderings
            if self.aug_based_time:
                if self.frame_neighbor:
                    for idx in range(self.aug_time):
                        rand_idx = np.random.randint(0, len(gt_idx))
                        new_order_diff = reorder_boxes_for_time_neighbor(rand_idx, mode="diff")
                        new_order_easy = reorder_boxes_for_time_neighbor(rand_idx, mode="easy")
                        
                        if idx < self.aug_time - 1:
                            tmp_idx_diff = gt_idx[new_order_diff]
                            tmp_idx_easy = gt_idx[new_order_easy]
                            if self.box_aug_mode == 'diff':
                                delta_total = compute_bbox_deltas(gt_rt_original[tmp_idx_diff])
                            else:
                                delta_total = compute_bbox_deltas(gt_rt_original[tmp_idx_easy])
                elif self.frame_shift:
                    selected_idx = []
                    for idx in range(self.aug_time):
                        if idx == 0:
                            rand_idx = np.random.randint(0, len(gt_idx))
                        else:
                            valid_choices = np.setdiff1d(np.arange(len(gt_idx)), selected_idx)
                            rand_idx = np.random.choice(valid_choices)
                        selected_idx.append(rand_idx)
                        new_order_diff = reorder_boxes_shift(rand_idx, gt_idx, gt_rt_original, mode="diff")
                        new_order_easy = reorder_boxes_shift(rand_idx, gt_idx, gt_rt_original, mode="easy")
                else:
                    new_order_diff = reorder_boxes_for_time(mode="diff")
                    new_order_easy = reorder_boxes_for_time(mode="easy")
            else:
                new_order_diff = reorder_boxes(mode="diff")
                new_order_easy = reorder_boxes(mode="easy")
                
            gt_idx_new_diff = gt_idx[new_order_diff]
            gt_idx_new_easy = gt_idx[new_order_easy]

            aug_gt_rt_diff[gt_idx] = aug_gt_rt_diff[gt_idx_new_diff]
            aug_segment_diff[gt_idx] = aug_segment_diff[gt_idx_new_diff]

            aug_gt_rt_easy[gt_idx] = aug_gt_rt_easy[gt_idx_new_easy]
            aug_segment_easy[gt_idx] = aug_segment_easy[gt_idx_new_easy]
            
            reorder_idxs_easy[gt_idx] = reorder_idxs_easy[gt_idx_new_easy]
            reorder_idxs_diff[gt_idx] = reorder_idxs_diff[gt_idx_new_diff]
            
            before_delta = np.mean(np.diagonal(original_delta_total, offset=1))
            aug_diff_delta = compute_bbox_deltas(gt_box[new_order_diff])
            after_diff_delta = np.mean(np.diagonal(aug_diff_delta, offset=1))
            aug_diff_delta = compute_bbox_deltas(gt_box[new_order_easy])
            after_easy_delta = np.mean(np.diagonal(aug_diff_delta, offset=1))  
            
            if self.gt_consider:
                aug_gt_prob_diff[gt_idx] = aug_gt_prob_diff[gt_idx_new_diff]
                aug_gt_prob_easy[gt_idx] = aug_gt_prob_easy[gt_idx_new_easy]
            
        data = {
            'aug_segment_diff': aug_segment_diff,
            'aug_gt_rt_diff': aug_gt_rt_diff,
            'aug_segment_easy': aug_segment_easy,
            'aug_gt_rt_easy': aug_gt_rt_easy,
            'before_delta': before_delta,
            'after_diff_delta': after_diff_delta,
            'after_easy_delta': after_easy_delta,
            'reorder_idxs_easy': reorder_idxs_easy,
            'reorder_idxs_diff': reorder_idxs_diff,
            'aug_gt_prob_diff': aug_gt_prob_diff,
            'aug_gt_prob_easy': aug_gt_prob_easy
        }          
        
        
        return data


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

def compute_bbox_deltas(gt_box):
    """
    Compute center points, width, height, pairwise distances,
    width/height differences, scale ratios, and total change (delta) for bounding boxes.
    
    Parameters:
        gt_box (numpy.ndarray): Bounding box coordinates of shape (N, 4),
                                where each row is [y1, x1, y2, x2].

    Returns:
        tuple: distance (numpy.ndarray), delta_w (numpy.ndarray), delta_h (numpy.ndarray),
               scale_w (numpy.ndarray), scale_h (numpy.ndarray), delta_total (numpy.ndarray)
    """
    # Compute center points, width, and height
    cx = (gt_box[:, 1] + gt_box[:, 3]) / 2
    cy = (gt_box[:, 0] + gt_box[:, 2]) / 2
    w = gt_box[:, 3] - gt_box[:, 1]
    h = gt_box[:, 2] - gt_box[:, 0]
    
    # Compute pairwise distances
    cx_diff = cx[:, None] - cx[None, :]
    cy_diff = cy[:, None] - cy[None, :]
    distance = np.sqrt(cx_diff**2 + cy_diff**2)
    
    # Compute width and height differences
    delta_w = w[:, None] - w[None, :]
    delta_h = h[:, None] - h[None, :]
    
    # Compute scale ratios safely
    valid_w = (w[:, None] > 0) & (w[None, :] > 0)
    valid_h = (h[:, None] > 0) & (h[None, :] > 0)
    scale_w = np.where(valid_w, np.log(w[:, None] / w[None, :]), 0)
    scale_h = np.where(valid_h, np.log(h[:, None] / h[None, :]), 0)

    # Compute total change (delta)
    delta_total = distance + np.abs(delta_w) + np.abs(delta_h) + np.abs(scale_w) + np.abs(scale_h)
    
    return delta_total


class VQ2DEvalDataset(VQ2DFitDataset):
    def __init__(self, config, split = 'val', movement = ""):
        super().__init__(config, split, movement)
        self.num_frames_per_segment = self.num_frames
        self.segment_length = self.frame_interval * self.num_frames_per_segment  # trailing stride is considered as occupied
        self.test_submit = split == 'test_unannotated'
        del self.num_frames  # to avoid confusion

        self.eval_experiemnt = config.dataset.get('eval_experiment')

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

            if self.eval_experiemnt is not None:
                if self.eval_experiemnt == 'hide_objects_random5':
                    idxs = np.where(gt_prob > 0.5)[0]
                    if len(idxs) >= 10:
                        idxs_chosen = np.random.choice(idxs, 5, replace=False)
                        h, w = segment.shape[-2:]
                        bboxes = (gt_rt * [h, w, h, w]).astype(int)
                        for ii in idxs_chosen:
                            bbox = bboxes[ii]
                            segment[ii, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] = torch.rand(3, bbox[2] - bbox[0], bbox[3] - bbox[1])


        segment, gt_rt, _ = self.pad_and_resize(segment, gt_rt)  # [t, c, s, s], [t, 4]

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
    # VQ2DEvalDataset.testme()
    import hydra
    hydra.initialize(config_path='../../config', version_base='1.3')
    # config = hydra.compose(config_name='train', overrides=['dataset=vq2d'])
    config = hydra.compose(config_name='train', overrides=['dataset=vq2d', '+experiment=frame_box_aug'])
    # config = hydra.compose(config_name='train', overrides=['dataset=vq2d', '+experiment=frame_dash'])
    # config.dataset.clips_dir = '/data/datasets/LaSOT'
    ds_config = config.dataset
    import lightning as L
    import argparse
    from pathlib import Path
    import json
    # L.seed_everything(42)
    
    ds = VQ2DFitDataset(config, split='train')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    ds.box_aug_mode='diff'
    
    # ver1
    ds.frame_box_aug = True
    dir_name = 'ver1'
    
    # ver2-t-2
    # ds.aug_based_time=True
    # ds.aug_time=2
    # dir_name = 'ver2-t-2'
    
    # ver2-t=3
    ds.aug_based_time=True
    dir_name = 'ver2-t-3'
    
    # # ver3-t-2
    # ds.frame_neighbor=True
    # ds.aug_time=2
    # dir_name = 'ver3-t-2'
    
    # # # # ver3-t-3
    # ds.frame_neighbor=True
    # ds.aug_time=3
    # dir_name = 'ver3-t-3'
    
    # # # # ver4
    ds.frme_shift=True
    ds.aug_time=2
    dir_name = 'ver4'
    
    p_out_root = Path(f'/data/soyeonhong/vq2d/vq2d-lightning/outputs/{dir_name}')
    p_out_root.mkdir(parents=True, exist_ok=True)
    
    # from imgcat import imgcat
    # import matplotlib.pyplot as plt
    # import io
    # idx = 0  # landscape
    # idx = 565  # portrait
    
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=ds_config.batch_size,
        shuffle=True,
        pin_memory=ds_config.pin_memory,
        prefetch_factor=ds_config.prefetch_factor,
        persistent_workers=ds_config.persistent_workers,
        num_workers=ds_config.num_workers,
        drop_last=True,
        )

    diffrence_all = []
    for i, sample in enumerate(dl):
        
        p_out = p_out_root / f'diff_{i}.json'
        
        if p_out.exists():
            continue
        difference = sample['experiment']['frame_aug']['difference']
        
        json.dump(difference.tolist(), open(p_out, 'w'))
        
        print(f'{p_out} saved.')
        
    
    # for i in range(1000):
    #     idx = np.random.randint(0, len(ds))
    #     sample = ds[idx]
    #     print(sample['seg_idxs'])
    # segment = sample['segment']
    # gt_bboxes = sample['gt_bboxes']
    # T = len(segment)

    # for t in range(0, T, T // 10):
    #     image = plt.imshow(segment[t].permute(1, 2, 0).cpu().numpy())
    #     y1, x1, y2, x2 = gt_bboxes[t] * (segment.shape[-2:] * 2)
    #     ax = plt.gca()
    #     ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=2))
    #     img_io = io.BytesIO()
    #     plt.savefig(img_io, format='png')
    #     plt.close()
    #     imgcat(img_io.getvalue())
    #     print()

    # image = sample['query']
    # img_io = io.BytesIO()
    # plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    # plt.savefig(img_io, format='png')
    # imgcat(img_io.getvalue())
    # print()