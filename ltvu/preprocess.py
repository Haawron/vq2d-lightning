import io
import os
import pwd
import tarfile
import json
import time
from typing import override
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
import decord
decord.bridge.set_bridge("torch")

from ltvu.dataset import shift_indices_to_clip_range


REPORTED_INVALID_CLIP_UIDS = {
    'b6061527-3ae2-46b4-b050-15eddc1967bb',  # vq2d
}

REPORTED_INVALID_VIDEO_UIDS = {
    # vq2d test
    '26d0d4bb-df7e-4459-805e-5db87b170e11',
    'fdebc0fb-0f2e-42c8-b2f3-c697b4fc1f8a',
    '67ae160b-f173-4340-b048-6aebf4027a9d',
    '0b2e10d7-6d96-4a4d-a7f9-69656f3ac20b',
    '56723ca2-d092-4a6d-aa2a-25669f6644f7',
    '196e0e8c-f29f-48de-8e1e-ce52c2e76641',
}


def generate_flat_annotations_vq2d(p_official_ann: Path):
    """
    Usage
    -----
    
    Basic usage:
    
        from ltvu.preprocess import generate_flat_annotations_vq2d
        p_official_ann = Path('SOMEPATH/vq_val.json')
        flat_anns = generate_flat_annotations_vq2d(p_official_ann)
    """

    def polish_bbox_dict_keys(bbox: dict):
        key_map = {
            'frame_number': 'fno',
            'x': 'x', 'y': 'y', 'width': 'w', 'height': 'h',
            'original_width': None, 'original_height': None}
        return {key_map[k]: v for k, v in bbox.items() if key_map.get(k) is not None}

    all_anns = json.load(p_official_ann.open())
    is_annotated = 'unannotated' not in p_official_ann.stem
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

                    oh, ow = qset['visual_crop']['original_height'], qset['visual_crop']['original_width']
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
                        'visual_crop': polish_bbox_dict_keys(qset['visual_crop']),
                    }

                    if is_annotated:
                        rt = [polish_bbox_dict_keys(bbox) for bbox in qset['response_track']]
                        sample.update({
                            'response_track_valid_range': [rt[0]['fno'], rt[-1]['fno']],
                            'response_track': rt,
                        })
                    flat_anns.append(sample)
    return flat_anns


def generate_flat_annotations_egotracks(p_official_ann: Path):
    """
    Usage
    -----
    
    Basic usage:
    
        from ltvu.preprocess import generate_flat_annotations_egotracks
        p_official_ann = Path('SOMEPATH/vq_val.json')
        flat_anns = generate_flat_annotations_egotracks(p_official_ann)
    """

    REPORTED_INVALID_CLIP_UIDS = {
        # egotracks
        # https://discuss.ego4d-data.org/t/egotracks-dataset-download-failure/218/28?page=2
        '59daca91-5433-48a4-92fc-422b406b551f',
        'db211359-c259-4515-9d6c-be521711b6d0',
        '87b52dc5-3ac3-47e7-9648-1b719049732f',
        'b7fc5f98-e5d5-405d-8561-68cbefa75106'
    }

    def polish_bbox_dict(bbox: dict):
        key_map = {
            'frame_number': 'fno',
            'x': 'x', 'y': 'y', 'width': 'w', 'height': 'h',
            'original_width': None, 'original_height': None}
        bbox = {kk: v for k, v in bbox.items() if (kk:=key_map.get(k)) is not None}
        bbox = {k: round(v, 2) if isinstance(v, float) else v for k, v in bbox.items()}
        return bbox

    p_official_ann = Path(p_official_ann)
    all_anns = json.load(p_official_ann.open())
    is_annotated = 'unannotated' not in p_official_ann.stem
    flat_anns = []
    count_invalids = 0
    for ann_video in all_anns['videos']:
        video_uid = ann_video['video_uid']
        if len(ann_video['clips']) == 0:
            flat_anns.append({'video_uid': video_uid})
            continue
        for ann_clip in ann_video['clips']:
            clip_uid = ann_clip['clip_uid']
            if clip_uid is None or clip_uid in REPORTED_INVALID_CLIP_UIDS:
                continue
            clip_duration = ann_clip['video_end_sec'] - ann_clip['video_start_sec']
            clip_fps = ann_clip['clip_fps']
            for ann_annots in ann_clip['annotations']:
                for qset_id, qset in ann_annots['query_sets'].items():
                    if 'is_valid' in qset and not qset['is_valid']:
                        count_invalids += 1
                        continue

                    oh, ow = qset['visual_crop']['original_height'], qset['visual_crop']['original_width']
                    sample = {
                        'video_uid': video_uid,
                        'clip_uid': clip_uid,
                        'query_set': qset_id,
                        'clip_fps': clip_fps,
                        'clip_duration': clip_duration,
                        'original_width': ow,
                        'original_height': oh,
                        'query_frame': qset.get('query_frame', 1000000),
                        'object_title': qset['object_title'],
                        'visual_crop': polish_bbox_dict(qset['visual_crop']),
                    }

                    if sample['query_frame'] == 1000000:  # egotracks test
                        del sample['query_frame']

                    sample['uuid_ltt'] = f'{clip_uid}_{qset_id}_{qset["object_title"]}'

                    if is_annotated:  # at most 2 samples will be added
                        rt = [polish_bbox_dict(bbox) for bbox in qset['response_track']]
                        flat_anns.append({
                            **sample,
                            'response_track_valid_range': [rt[0]['fno'], rt[-1]['fno']],
                            'response_track': rt,
                        })

                        if 'lt_track' in qset:
                            ltt = sorted(
                                [polish_bbox_dict(bbox) for bbox in qset['lt_track']],
                                key=lambda x: x['fno'])
                            if 'visual_clip' in qset:
                                vcl = sorted(
                                    [polish_bbox_dict(bbox) for bbox in qset['visual_clip']],
                                    key=lambda x: x['fno'])
                                assert len(vcl) == vcl[-1]['fno'] - vcl[0]['fno'] + 1
                            else:
                                vcl = []

                            flat_anns.append({
                                **sample,
                                'response_track_valid_range': [rt[0]['fno'], rt[-1]['fno']],
                                'response_track': rt,
                                'lt_track': ltt,
                                'visual_clip': vcl,
                            })
                    else:
                        flat_anns.append(sample)
    return flat_anns


class FrameExtractAndSaveAsTarfileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        short_side: int,
        task: str = 'vq2d',
        split: str = 'train',
        p_raw_clips_dir = Path('/data/datasets/ego4d_data/v2/clips'),
        p_ego4d_dir = Path('/data/datasets/ego4d_data/v2'),
        p_tarfiles_dir = Path('./outputs/frames'),
        clip_uids: list[str] = [],
    ):
        super().__init__()
        self.short_side = short_side
        self.p_raw_clips_dir = p_raw_clips_dir  # FPS: any, resolution: Any

        if task == 'vq2d':
            if split == 'test':
                split = 'test_unannotated'
            p_official_ann = p_ego4d_dir / 'annotations' / f'vq_{split}.json'
            all_anns = generate_flat_annotations_vq2d(p_official_ann)
        elif task == 'egotracks':
            if split == 'test':
                split = 'challenge_test_unannotated'
            p_official_ann = p_ego4d_dir / 'egotracks' / f'egotracks_{split}.json'
            all_anns = generate_flat_annotations_egotracks(p_official_ann)

        clip2anns = defaultdict(list)
        for ann in all_anns:
            if 'clip_uid' not in ann:
                continue
            clip_uid = ann['clip_uid']
            clip2anns[clip_uid].append(ann)

        # self.all_anns = all_anns
        self.clip_fps = all_anns[0]['clip_fps']
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
        ssdir = f'{ss}ss' if ss > 0 else 'raw'

        # get video info
        vr = VideoReader(str(p_raw_clip))
        fps_raw = vr.get_avg_fps()  # might be 30
        stride = round(fps_raw / self.clip_fps)  # 6

        # get frame idxs to be extracted
        all_frame_idxs, clip_len = self.get_frame_idxs(anns, len(vr), stride)

        # extract frames and resize!
        raw_idxs = stride * all_frame_idxs
        raw_idxs[-1] = len(vr) - 1
        chunk_size = 128
        tar = tarfile.open(self.p_tarfile, 'a')
        for chidx in range(0, len(raw_idxs), chunk_size):
            frame_idxs = all_frame_idxs[chidx: chidx + chunk_size]
            frames: torch.Tensor = vr.get_batch(raw_idxs[chidx: chidx + chunk_size])    # [N, H, W, c]
            frames: np.ndarray = frames.numpy()
            frames = self.resize(frames)  # [N, h, w, c]
            frames = [Image.fromarray(frame) for frame in frames]
            for frame_idx, frame in zip(frame_idxs, frames):
                frame_io = io.BytesIO()
                frame.save(frame_io, format='JPEG')
                frame_io.seek(0)
                tarinfo = tarfile.TarInfo(name=f'ego4d_data/v2/vq2d_frames/{ssdir}/{clip_uid}/frame_{1+frame_idx:07d}.jpg')  # 1-based, following ffmpeg
                tarinfo.mtime = time.time()
                tarinfo.size = frame_io.getbuffer().nbytes
                tar.addfile(tarinfo, fileobj=frame_io)
        tar.close()

        return clip_uid, all_frame_idxs, clip_len, torch.utils.data.get_worker_info().id, self.p_tarfile

    def get_frame_idxs(self, anns, clip_len_raw, frame_interval=6):
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
            if vc_idx * frame_interval == clip_len_raw:  # last frame
                vc_idx -= 1
            s, e = vc_idx - 6, vc_idx + 6
            s, e = shift_indices_to_clip_range((s, e), np.ceil(clip_len_raw / frame_interval).astype(int).item())
            assert frame_interval * s <= frame_interval * vc_idx <= frame_interval * e < clip_len_raw, f'{s=}, {vc_idx=}, {e=}, {clip_len_raw=}'
            ranges_extended.append((s, e))

        # gather all frame idxs to be extracted
        all_frame_idxs = set()
        for s, e in ranges_extended:
            all_frame_idxs.update(range(s, e + 1))
        all_frame_idxs = np.array(sorted(all_frame_idxs) + [np.ceil(clip_len_raw / frame_interval).astype(int)])

        return all_frame_idxs, clip_len

    def resize(self, frames: np.ndarray):
        if self.short_side > 0:
            _, h, w, _ = frames.shape
            target_size = round(self.short_side * w / h), self.short_side  # w, h (reverse to the input order)
            frames = [cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC) for frame in frames]
        return frames

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        dataset: FrameExtractAndSaveAsTarfileDataset = worker_info.dataset
        p_tarfiles_dir = dataset.p_tarfiles_dir
        p_tarfile = p_tarfiles_dir / f'worker_{worker_id}.tar'
        with tarfile.open(p_tarfile, 'w'):  # create tarfile
            pass
        dataset.p_tarfile = p_tarfile


class FrameExtractAndSaveAsTarfileDatasetWholeClip(FrameExtractAndSaveAsTarfileDataset):
    @override
    def get_frame_idxs(self, anns, clip_len_raw, frame_interval=6):
        return np.arange(np.ceil(clip_len_raw / frame_interval).astype(int)), clip_len_raw


class FrameExtractAndSaveAsTarfileDatasetFewClipsAndWhole(FrameExtractAndSaveAsTarfileDataset):
    def __init__(
        self,
        *args,
        clip_uids: list[str] = [],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clip_uids = clip_uids
        self.clip2anns = {clip_uid: None for clip_uid in self.clip_uids}

    @override
    def get_frame_idxs(self, anns, clip_len_raw, frame_interval=6):
        return np.arange(np.ceil(clip_len_raw / frame_interval).astype(int)), clip_len_raw


def main(
    short_side: int = 320,
    task: str = 'vq2d',
    split: str = 'train',
    whole = False, world_size = 1, rank = 0,
    only_gather: bool = False,
    clip_uids: list[str] = [],
):
    """
    Usage
    -----

    Run preprocessing for training split:

        python -m ltvu.preprocess  # will process both train and val split

    Run preprocessing in parallel for 4 ranks:

        for rank in {0..3}; do
            python -m ltvu.preprocess --world_size 4 --rank $rank &
        done

    Run preprocessing for validation split for evaluation:

        python -m ltvu.preprocess --split val --whole

    Run preprocessing for egotracks train:
    
        python -m ltvu.preprocess --task egotracks --split train --whole

    Run only gathering tarfiles:

        python -m ltvu.preprocess --task egotracks --split train --whole --only_gather

    Run preprocessing for few clips and whole clip:

        python -Bm ltvu.preprocess --task egotracks --whole --clip_uids '["17b73c0a-afda-4944-b2ed-450c9ef97849", "622c1b29-76c6-4845-95df-7e54792687d4", "d1419b9b-2944-421b-ba6f-0ddac32d5521", "ae8727ba-fe6f-4411-b277-48a8b7326a2a", "74130fd9-3e7b-482a-9627-0f53ac672f57", "59daca91-5433-48a4-92fc-422b406b551f", "72f95d60-cf26-4821-8d79-4ec72c748031", "87b52dc5-3ac3-47e7-9648-1b719049732f", "b7fc5f98-e5d5-405d-8561-68cbefa75106", "db211359-c259-4515-9d6c-be521711b6d0", "78a01e40-6ab7-4c4f-b596-b8908eff923"]'
    """

    assert split in ('train', 'val', 'test_unannotated', 'challenge_test_unannotated')

    print(f'Preprocessing {"VQ2D" if task == 'vq2d' else 'EgoTracks' if task == 'egotracks' else '???'} frames with short_side={short_side}, split={split}')
    p_tarfiles_dir = Path(f'./outputs/frames')
    p_tarfiles_tmpdir = p_tarfiles_dir / f'tmp/{rank}/'

    # determine the frame extraction coverage
    if not whole:
        ds_class = FrameExtractAndSaveAsTarfileDataset
    else:
        if clip_uids:
            ds_class = FrameExtractAndSaveAsTarfileDatasetFewClipsAndWhole
        else:
            ds_class = FrameExtractAndSaveAsTarfileDatasetWholeClip

    # determine the filename
    ssdir = f'{short_side}ss' if short_side > 0 else 'raw'
    if clip_uids:
        filename = f'subsamples-{task}_pos_and_query_frames_{ssdir}-{split}'
    else:
        filename = f'{task}_pos_and_query_frames_{ssdir}-{split}'
    p_tarfile_gathered = p_tarfiles_dir / f'{filename}.tar'
    print(p_tarfile_gathered)

    if clip_uids:
        print(f'Preprocessing {len(clip_uids)} clips: {clip_uids}')

    if not only_gather:
        num_workers = int(os.environ.get('SLURM_CPUS_ON_NODE', 1)) // 4
        if clip_uids:
            num_workers = min(num_workers, len(clip_uids))
        print(f'{num_workers=}, {world_size=}, {rank=}')

        if p_tarfiles_tmpdir.exists():
            if rank == 0:
                for p_tarfile in p_tarfiles_tmpdir.parent.glob('**/worker_*.tar'):
                    p_tarfile.unlink()
        else:
            p_tarfiles_tmpdir.mkdir(exist_ok=True, parents=True)
        ds = ds_class(short_side=short_side, task=task, split=split, p_tarfiles_dir=p_tarfiles_tmpdir, clip_uids=clip_uids)

        length = len(ds)
        sampler = torch.utils.data.distributed.DistributedSampler(
            list(range(length)), num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=num_workers,
            sampler=sampler,
            worker_init_fn=FrameExtractAndSaveAsTarfileDataset.worker_init_fn,
            collate_fn=lambda x: x[0])

        pbar = tqdm(dl, total=len(sampler), desc=f'Rank {rank}/{world_size}', leave=True)
        frames_processed, total_frame_count = 0, 0
        for bidx, (clip_uid, frame_idxs, clip_len, worker_id, p_tarfile) in enumerate(pbar):
            frames_processed += len(frame_idxs)
            total_frame_count += clip_len
            ratio_extracted = frames_processed / total_frame_count
            tqdm.write(f'Worker {worker_id}: {clip_uid} {len(frame_idxs)} frames out of {clip_len} -> {p_tarfile}')
            pbar.set_description(f'rank={rank}/{world_size} {ratio_extracted:.1%} = {frames_processed}/{total_frame_count}')

        del ds, dl
        print('Waiting for all workers to close tarfiles...')
        time.sleep(0 if clip_uids else 600)

    if rank == 0:
        print('Rank 0: gathering tarfiles...')
        gather_tarfiles(p_tarfiles_tmpdir, p_tarfile_gathered)
        print(f'Rank 0: gathered tarfiles -> {p_tarfile_gathered}')


def gather_tarfiles(
    p_tarfiles_tmpdir: Path,
    p_tarfile_gathered: Path,
):
    with tarfile.open(p_tarfile_gathered, 'w') as outtar:
        seen_files = set()
        for p_tarfile_tmp in tqdm(list(p_tarfiles_tmpdir.parent.glob('**/worker_*.tar'))):
            with tarfile.open(p_tarfile_tmp, 'r') as tar:
                for member in tar.getmembers():
                    if not member.isfile(): continue
                    if member.name in seen_files: continue
                    file = tar.extractfile(member)
                    if file is None: continue
                    file_io = file.read()
                    tarinfo = tarfile.TarInfo(name=member.name)
                    tarinfo.size = len(file_io)
                    tarinfo.uid = tarinfo.gid = os.getuid()
                    tarinfo.uname = tarinfo.gname = pwd.getpwuid(os.getuid()).pw_name  # user name
                    tarinfo.mtime = member.mtime
                    outtar.addfile(tarinfo, io.BytesIO(file_io))
                    seen_files.add(member.name)
            # p_tarfile_tmp.unlink()


if __name__ == '__main__':
    import fire
    fire.Fire(main)
