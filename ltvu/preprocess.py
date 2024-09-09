import io
import os
import tarfile
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.utils.data
import torch.utils.data.distributed

import cv2
import numpy as np
from decord import VideoReader
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from ltvu.dataset import shift_indices_to_clip_range


REPORTED_INVALID_CLIP_UIDS = {
    'b6061527-3ae2-46b4-b050-15eddc1967bb',  # vq2d
}


def generate_flat_annotations_vq2d(all_anns):

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


class FrameExtractAndSaveAsTarfileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        short_side: int,
        p_raw_clips_dir = Path('/data/datasets/ego4d_data/v2/clips'),
        p_tarfiles_dir = Path('./outputs/frames'),
    ):
        super().__init__()
        self.short_side = short_side
        self.p_raw_clips_dir = p_raw_clips_dir  # FPS: any, resolution: Any

        p_anns = [Path(f'./data/vq_v2_{split}_anno.json') for split in ['train', 'val']]
        all_anns = sum([json.load(p_ann.open()) for p_ann in p_anns], [])
        clip2anns = defaultdict(list)
        for ann in all_anns:
            clip_uid = ann['clip_uid']
            clip2anns[clip_uid].append(ann)

        self.all_anns = all_anns
        self.clip_uids = sorted(clip2anns.keys())
        self.clip2anns = clip2anns
        self.p_tarfiles_dir = p_tarfiles_dir
        self.p_tarfile = None  # per worker
        # self.tar: tarfile.TarFile = None  # per worker

    def __len__(self):
        return len(self.clip_uids)

    def __getitem__(self, data_idx):
        clip_uid = self.clip_uids[data_idx]
        anns = self.clip2anns[clip_uid]
        p_raw_clip = self.p_raw_clips_dir / f'{clip_uid}.mp4'
        ss = self.short_side

        # get video info
        vr = VideoReader(str(p_raw_clip))
        fps_raw = vr.get_avg_fps()  # might be 30
        fps_ann = anns[0]['clip_fps']  # might be 5
        stride = round(fps_raw / fps_ann)  # 6

        # get frame idxs to be extracted
        ranges, vc_idxs, clip_len = [], [], 0
        for ann in anns:
            ranges.append(ann['response_track_valid_range'])  # inclusive
            vc_idxs.append(ann['visual_crop']['fno'])
            clip_len = max(clip_len, ann['query_frame'])

        # extend ranges as well as vc_idxs
        ranges_extended = []
        for s, e in ranges:  # extend 3 times
            c, r = (s + e) // 2, (e - s + 1) // 2
            r *= 3
            s, e = min(c - r, s - 32), max(c + r, e + 32)  # 32: minimum clip length
            s, e = shift_indices_to_clip_range((s, e), clip_len)  # adjust to clip range
            ranges_extended.append((s, e))
        for vc_idx in vc_idxs:
            if vc_idx * stride == len(vr):  # last frame
                vc_idx -= 1
            s, e = vc_idx - 6, vc_idx + 6
            s, e = shift_indices_to_clip_range((s, e), np.ceil(len(vr) / stride).astype(int).item())
            assert stride * s <= stride * vc_idx <= stride * e < len(vr), f'{s=}, {vc_idx=}, {e=}, {len(vr)=}'
            ranges_extended.append((s, e))

        # gather all frame idxs to be extracted
        all_frame_idxs = set()
        for s, e in ranges_extended:
            all_frame_idxs.update(range(s, e + 1))
        all_frame_idxs = np.array(sorted(all_frame_idxs) + [np.ceil(len(vr) / stride).astype(int)])

        # extract frames and resize!
        raw_idxs = stride * all_frame_idxs
        raw_idxs[-1] = len(vr) - 1
        chunk_size = 128
        tar = tarfile.open(self.p_tarfile, 'a')
        for chidx in range(0, len(raw_idxs), chunk_size):
            frame_idxs = all_frame_idxs[chidx: chidx + chunk_size]
            frames: torch.Tensor = vr.get_batch(raw_idxs[chidx: chidx + chunk_size])    # [N, H, W, c]
            frames = frames.numpy()
            frames = self.resize(frames)  # [N, h, w, c]
            frames = [Image.fromarray(frame) for frame in frames]
            for frame_idx, frame in zip(frame_idxs, frames):
                frame_io = io.BytesIO()
                frame.save(frame_io, format='JPEG')
                frame_io.seek(0)
                tarinfo = tarfile.TarInfo(name=f'ego4d_data/v2/vq2d_frames/{ss}ss/{clip_uid}/frame_{1+frame_idx:07d}.jpg')  # 1-based, following ffmpeg
                tarinfo.size = frame_io.getbuffer().nbytes
                tar.addfile(tarinfo, fileobj=frame_io)
        tar.close()

        return clip_uid, all_frame_idxs, clip_len, torch.utils.data.get_worker_info().id, self.p_tarfile

    def resize(self, frames: np.ndarray):
        _, h, w, _ = frames.shape
        target_size = round(self.short_side * w / h), self.short_side  # w, h (reverse to the input order)
        frames = [cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC) for frame in frames]
        return frames

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        p_tarfiles_dir = dataset.p_tarfiles_dir
        p_tarfile = p_tarfiles_dir / f'worker_{worker_id}.tar'
        with tarfile.open(p_tarfile, 'w'):  # create tarfile
            pass
        dataset.p_tarfile = p_tarfile


def main(short_side = 520, world_size = 1, rank = 0):
    num_workers = os.cpu_count() // 4
    print(f'{num_workers=}, {world_size=}, {rank=}')
    ds = FrameExtractAndSaveAsTarfileDataset(short_side=short_side)
    length = len(ds)
    sampler = torch.utils.data.distributed.DistributedSampler(
        list(range(length)), num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=num_workers,
        sampler=sampler,
        worker_init_fn=FrameExtractAndSaveAsTarfileDataset.worker_init_fn,
        collate_fn=lambda x: x[0])

    p_tarfiles_dir = Path('./outputs/frames')
    p_tarfiles_dir.mkdir(exist_ok=True, parents=True)

    pbar = tqdm(dl, total=len(ds), desc=f'Rank {rank}/{world_size}', leave=True)
    frames_processed, total_frame_count = 0, 0
    for bidx, (clip_uid, frame_idxs, clip_len, worker_id, p_tarfile) in enumerate(pbar):
        frames_processed += len(frame_idxs)
        total_frame_count += clip_len
        ratio_extracted = frames_processed / total_frame_count
        tqdm.write(f'Worker {worker_id}: {clip_uid} {len(frame_idxs)} frames out of {clip_len} -> {p_tarfile}')
        pbar.set_description(f'rank={rank}/{world_size} {ratio_extracted:.1%} = {frames_processed}/{total_frame_count}')

    del ds, dl
    print('Waiting for all workers to close tarfiles...')
    time.sleep(10)

    if rank == 0:
        print('Rank 0: gathering tarfiles...')
        p_tarfile_gathered = p_tarfiles_dir / f'vq2d_pos_and_query_frames_{short_side}ss.tar'
        with tarfile.open(p_tarfile_gathered, 'w') as outtar:
            seen_files = set()
            for p_tarfile in tqdm(list(p_tarfiles_dir.glob('worker_*.tar'))):
                with tarfile.open(p_tarfile, 'r') as tar:
                    for member in tar.getmembers():
                        if not member.isfile(): continue
                        if member.name in seen_files: continue
                        file = tar.extractfile(member)
                        if file is None: continue
                        file_io = file.read()
                        tarinfo = tarfile.TarInfo(name=member.name)
                        tarinfo.size = len(file_io)
                        tarinfo.uid = 30013
                        tarinfo.gid = 30013
                        outtar.addfile(tarinfo, io.BytesIO(file_io))
                        seen_files.add(member.name)
                p_tarfile.unlink()
        print(f'Rank 0: gathered tarfiles -> {p_tarfile_gathered}')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
