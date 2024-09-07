import json
import shutil
from collections import defaultdict
from pathlib import Path
import subprocess

import numpy as np
from tqdm import tqdm

from .dataset import shift_indices_to_clip_range


REPORTED_INVALID_CLIP_UIDS = {
    'b6061527-3ae2-46b4-b050-15eddc1967bb',  # vq2d
}


def get_clip_fps(p_clip) -> int:
    output = subprocess.check_output([
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries',
        'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(p_clip)]).decode().strip()
    numer, denom = map(int, output.split('/'))
    fps = np.round(numer / denom).astype(int).item()
    return fps


def get_frame_count(p_clip) -> int:
    frame_count = int(subprocess.check_output([
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries',
        'stream=nb_frames',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(p_clip)]).decode().strip())
    return frame_count


def generate_flat_annotations_vq2d(all_anns):
    P_CLIPS_DIR_FOR_CHECKING_VALIDITY = Path('/data/datasets/ego4d_data/v2/vq2d_clips_448')

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


def extract_and_resize_frames(
    p_clip, p_outdir_this_clip, s: int, e: int, short_side: int = 320, fps_orig: int = 5.
) -> subprocess.Popen[bytes]:
    """
    Extracts frames from a video between specified indices, resizes them to a short-side of 320 pixels, 
    and saves them in a specified directory with filenames following a zero-padded format.

    Parameters
    ----------
    p_clip : Path
        Path to the input video file (e.g., MP4).
    s : int
        Starting frame index (0-based).
    e : int
        Ending frame index (0-based), inclusive.
    p_outdir : Path
        Directory where the output frames will be saved. The directory name must be a valid UUID.

    Raises
    ------
    AssertionError
        If the `p_outdir` stem (basename) is not a valid UUID (length != 36).
    CalledProcessError
        If the `ffmpeg` subprocess fails.
    
    Notes
    -----
    The frames are extracted between indices `s` and `e` (inclusive), resized so that the shortest side
    is 320 pixels, and saved as `frame_{index:07d}.jpg` in the specified output directory. 
    The frame indices are adjusted to be 0-based.

    Examples
    --------
    >>> from pathlib import Path
    >>> extract_frames(Path('input.mp4'), 100, 200, Path('/output_dir/uuid_dir'))
    This will extract frames 100 to 200 from 'input.mp4', resize them, and save them as 'frame_0000100.jpg' to 'frame_0000200.jpg' in the output directory.
    """
    assert len(p_outdir_this_clip.stem) == 36, f'Output directory name must be a valid UUID: {p_outdir_this_clip}'
    p_outdir_this_clip.mkdir(parents=True, exist_ok=True)
    fps = get_clip_fps(p_clip)
    frame_count = get_frame_count(p_clip)
    stride = np.round(fps_orig / fps).astype(int).item()  # 1 or 6 in most cases
    assert stride == 1
    s, e = round(s / stride), round(e / stride)
    e = np.clip(e, s + 1, frame_count - 1).item()  # ensure e > s

    # N.B. don't remove existing frames here, as this function is called in parallel

    command = [
        'ffmpeg',
        '-hide_banner', '-loglevel', 'error',  # shutup
        '-i', str(p_clip),  # Input video file
        '-vf', f"select='between(n\\,{s}\\,{e})',scale='if(gt(iw,ih),-1,{short_side}):if(gt(iw,ih),{short_side},-1)'",
        '-vsync', 'vfr',  # Variable frame rate for syncing frames
        str(p_outdir_this_clip / f'tmp_{s:07d}_%07d.jpg')  # Output frame format, idx is 1-based
    ]

    process = subprocess.Popen(command)
    return process


def extract_and_resize_for_single_clip(
    p_fps5_clips_dir, p_out_frames_dir, clip_uid,
    exts, num_frames, p_outdir_this_clip,
    short_side=320, stride=1,
    f_report=None, pbar=None
):
    messages = []
    p_clip = p_fps5_clips_dir / f'{clip_uid}.mp4'
    if not p_clip.exists():
        msg = f'Clip {clip_uid} does not exist.'
        messages.append(msg)
    else:
        p_outdir_this_clip = p_out_frames_dir / clip_uid
        p_outdir_this_clip.mkdir(parents=True, exist_ok=True)

        for p_exist in p_outdir_this_clip.glob('*.jpg'):
            p_exist.unlink()

        ps: list[subprocess.Popen[bytes]] = []
        idxs_expected = []
        for ext in exts:
            if isinstance(ext, int):
                s, e = ext - stride, ext + stride
            else:
                s, e = ext
                c, l = (s + e) // 2, e - s + 1
                s, e = c - 3 * l // 2, c + 3 * l // 2  # 3 times the length of the GT interval
            s, e = shift_indices_to_clip_range([s, e], num_frames)
            idxs_expected.extend(range(int(s / stride) * stride, e + 1, stride))
            p = extract_and_resize_frames(
                p_clip, p_outdir_this_clip=p_outdir_this_clip,
                s=s, e=e, short_side=short_side)  # s and e are based on the annotation fps
            ps.append(p)

        for p in ps:
            p.wait()

        # rename
        for p_tmp in p_outdir_this_clip.glob('tmp_*.jpg'):
            _, offset, frame_idx = p_tmp.stem.split('_')
            offset, frame_idx = int(offset), int(frame_idx)
            frame_idx -= 1
            frame_idx = (offset + frame_idx) * stride
            p_frame = p_outdir_this_clip / f'frame_{frame_idx:07d}.jpg'
            if not p_frame.exists():
                p_tmp.rename(p_frame)

        # check validity of the extracted frames with the expected count
        p_frames = sorted(list(p_outdir_this_clip.glob('*.jpg')))
        count_expected = len(set(idxs_expected))
        count_processed = len(p_frames)
        if count_processed != count_expected:
            msg = f'Clip {clip_uid} has {count_processed} frames, expected {count_expected}.'
            messages.append(msg)
        processed_frame_count += count_processed

    return messages


def extract_and_resize_3times_extended_positive_and_query_frames(
    short_side: int = 320, world_size: int = 1, rank: int = 0
):
    p_fps5_clips_dir = Path('/data/datasets/ego4d_data/v2/vq2d_clips_448')  # fps 5
    p_out_frames_dir = Path(f'/data/datasets/ego4d_data/v2/vq2d_frames_{short_side}ss')
    p_report = Path(f'./not_extracted-rank{rank:02d}.txt')
    f_report = p_report.open('w')
    assert p_fps5_clips_dir.exists() and p_fps5_clips_dir.is_dir()
    print(f'Extracting frames from {p_fps5_clips_dir} to {p_out_frames_dir}.')
    print(f'World size {world_size}, Rank {rank}.')
    print(f'Reporting to {p_report}.')
    print()

    # get flat annotations
    p_anns = [Path(f'./data/vq_v2_{split}_anno.json') for split in ['train', 'val']]
    all_anns = sum([json.load(p_ann.open()) for p_ann in p_anns], [])

    # setup clip_uid mappings
    clip2ext = defaultdict(list)
    clip2len = {}
    # clip2fps = {}
    not_found_clips = set()
    for ann in all_anns:
        clip_uid = ann['clip_uid']
        if not (p_fps5_clips_dir / f'{clip_uid}.mp4').exists():
            not_found_clips.add(clip_uid)
            continue
        clip2ext[clip_uid].append(ann['response_track_valid_range'])
        clip2ext[clip_uid].append(ann['visual_crop']['fno'])
        clip2len[clip_uid] = ann['query_frame']
        # clip2fps[clip_uid] = ann['clip_fps']
    assert not not_found_clips, f'{len(not_found_clips)} clips are not found: {" ".join(list(not_found_clips))}'

    # subsample clips for distributed processing
    all_clip_uids = sorted(clip2ext.keys())
    rank_clip_uids = all_clip_uids[rank::world_size]

    # extract frames
    pbar = tqdm(total=len(rank_clip_uids), leave=True, mininterval=0, maxinterval=60, disable=rank>0)
    msg_fmt = f'World size {world_size}, Total clips {len(all_clip_uids)}'
    processed_frame_count = 0
    stride = 1

    for cidx, clip_uid in enumerate(rank_clip_uids):
        p_clip = p_fps5_clips_dir / f'{clip_uid}.mp4'
        exts = clip2ext[clip_uid]
        num_frames = clip2len[clip_uid]
        if not p_clip.exists():
            msg = f'Clip {clip_uid} does not exist.'
            f_report.write(msg + '\n')
            tqdm.write(msg)
        p_outdir_this_clip = p_out_frames_dir / clip_uid

        for p_exist in p_outdir_this_clip.glob('*.jpg'):
            p_exist.unlink()

        ps: list[subprocess.Popen[bytes]] = []
        idxs_expected = []
        for ext in exts:
            if isinstance(ext, int):
                s, e = ext - stride, ext + stride
            else:
                s, e = ext
                c, l = (s + e) // 2, e - s + 1
                s, e = c - 3 * l // 2, c + 3 * l // 2  # 3 times the length of the GT interval
            s, e = shift_indices_to_clip_range([s, e], num_frames)
            idxs_expected.extend(range(int(s / stride) * stride, e + 1, stride))
            p = extract_and_resize_frames(
                p_clip, p_outdir_this_clip=p_outdir_this_clip,
                s=s, e=e, short_side=short_side)  # s and e are based on the annotation fps
            ps.append(p)

        for p in ps:
            p.wait()

        # rename
        for p_tmp in p_outdir_this_clip.glob('tmp_*.jpg'):
            _, offset, frame_idx = p_tmp.stem.split('_')
            offset, frame_idx = int(offset), int(frame_idx)
            frame_idx -= 1
            frame_idx = (offset + frame_idx) * stride
            p_frame = p_outdir_this_clip / f'frame_{frame_idx:07d}.jpg'
            if not p_frame.exists():
                p_tmp.rename(p_frame)

        # check validity of the extracted frames with the expected count
        p_frames = sorted(list(p_outdir_this_clip.glob('*.jpg')))
        count_expected = len(set(idxs_expected))
        count_processed = len(p_frames)
        if count_processed != count_expected:
            msg = f'Clip {clip_uid} has {count_processed} frames, expected {count_expected}.'
            f_report.write(msg + '\n')
            tqdm.write(msg)
        processed_frame_count += count_processed

        # logging
        pbar.set_description(f'{msg_fmt}, Frames {processed_frame_count}')
        pbar.write(f'Clip {clip_uid} is processed. {p_outdir_this_clip}')
        pbar.update(1)
        f_report.flush()
    pbar.close()
    f_report.close()

    if rank == 0:
        all_clip_uids = set(all_clip_uids)
        extracted_clips = set(p_out_frames_dir.iterdir())
        not_extracted = all_clip_uids - extracted_clips
        if not_extracted:
            p_report = Path('./not_extracted.txt')
            json.dump(sorted(list(not_extracted)), p_report.open('w'))
            print(f'WARNING: {len(not_extracted)} clips are not extracted: {p_report}')
        else:
            print('All clips are extracted successfully.')


if __name__ == '__main__':
    import fire
    fire.Fire(extract_and_resize_3times_extended_positive_and_query_frames)
