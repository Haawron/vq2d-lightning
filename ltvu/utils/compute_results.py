import json

import torch

import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, medfilt


def compute_response_track(preds):
    # logit to prob
    probs = torch.sigmoid(preds['ret_scores']).numpy()

    # 1D median filter
    probs_sm = medfilt(probs, 5)

    # find the last valid peak, valid = large enough
    peaks, _ = find_peaks(probs_sm)
    if peaks.size == 0:
        peaks = np.array([np.argmax(probs_sm)])
    max_peak_prob = probs_sm[peaks].max()
    peak_threshold = max_peak_prob * 0.8
    valid_peaks = peaks[probs_sm[peaks] >= peak_threshold]
    if valid_peaks.size == 0:
        valid_peaks = np.array([np.argmax(probs_sm)])
    last_valid_peak = valid_peaks[-1]

    # find the last plateau
    plateau_threshold = probs_sm[last_valid_peak] * 0.7
    last_plateau_idx1 = np.where(probs_sm[:last_valid_peak] < plateau_threshold)[0]
    if last_plateau_idx1.size == 0:
        last_plateau_idx1 = len(probs) - 2
        last_plateau_idx2 = len(probs) - 1
    else:
        last_plateau_idx1 = last_plateau_idx1[-1]
        last_plateau_idx2 = np.where(probs_sm[last_valid_peak:] < plateau_threshold)[0]
        if last_plateau_idx2.size == 0:
            last_plateau_idx2 = len(probs) - 1
        else:
            last_plateau_idx2 = last_plateau_idx2[0] + last_valid_peak

    return preds['ret_bboxes'][last_plateau_idx1:last_plateau_idx2].numpy(), last_plateau_idx1, last_plateau_idx2


def get_final_preds(preds, split='val'):
    """Convert whole-clip predictions to submittable format.
    
    Usage:
    
        preds = torch.load('SOMEPATH/intermediate_predictions.pt', weights_only=True)
        results = get_final_preds(preds)
        json.dump(results, open('SOMEPATH/predictions.json', 'w'))
    """

    result = {
        'version': '1.0.5',
        'challenge': 'ego4d_vq2d_challenge',
        'results': {
            'videos': [
            ]
        }
    }

    anns = json.load(open(f'data/vq_v2_{split}_anno.json'))
    pred_tree = {}  # video_uid -> clip_uid -> annotation_uid -> qset_id -> prediction
    for ann in anns:
        video_uid = ann['video_uid']
        clip_uid = ann['clip_uid']
        annotation_uid = ann['annotation_uid']
        qset_id = ann['query_set']

        qset_uuid = f'{annotation_uid}_{qset_id}'
        qset_pred = preds[qset_uuid]
        bboxes, fno_s, _ = compute_response_track(qset_pred)  # bboxes and start fno of the response track
        # just edit keys and prepend fno_s to the bboxes
        final_bboxes = [
            {'fno': fno, 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]}
            for fno, b in enumerate(bboxes.astype(int).tolist(), start=fno_s)]
        pred_tree.setdefault(video_uid, {}).setdefault(clip_uid, {}).setdefault(annotation_uid, {})[qset_id] \
            = final_bboxes

    for video_uid, clips in pred_tree.items():
        result['results']['videos'].append({
            'video_uid': video_uid,
            'clips': []
        })
        for clip_uid, annotations in clips.items():
            result['results']['videos'][-1]['clips'].append({
                'clip_uid': clip_uid,
                'predictions': []
            })
            for annotation_uid, query_sets in annotations.items():
                result['results']['videos'][-1]['clips'][-1]['predictions'].append({
                    'annotation_uid': annotation_uid,
                    'query_sets': {}
                })
                for qset_id, final_bboxes in query_sets.items():
                    result['results']['videos'][-1]['clips'][-1]['predictions'][-1]['query_sets'][qset_id] = {
                        'bboxes': final_bboxes,
                        'score': 1.0
                    }

    return result


if __name__ == '__main__':
    from pathlib import Path
    from ltvu.metrics import get_metrics, print_metrics

    p_tmp_outdir = Path('outputs/debug/2024-09-25/126347/tmp')
    p_int_pred = p_tmp_outdir.parent / 'intermediate_predictions.pt'
    p_pred = p_tmp_outdir.parent / 'predictions.json'

    # all_seg_preds = {}
    # for p_pt in p_tmp_outdir.glob('*.pt'):
    #     rank_seg_preds = torch.load(p_pt, weights_only=True)
    #     for qset_uuid, seg_idx, num_segments, pred_output in rank_seg_preds:
    #         if qset_uuid not in all_seg_preds:
    #             all_seg_preds[qset_uuid] = [None] * num_segments
    #         all_seg_preds[qset_uuid][seg_idx] = pred_output

    # # merge features
    # qset_preds = {}
    # for qset_uuid, qset_seg_preds in all_seg_preds.items():
    #     new_ret_bboxes, new_ret_scores, frame_idxs = [], [], []
    #     num_segments = len(qset_seg_preds)
    #     for seg_idx, seg_pred in enumerate(qset_seg_preds):
    #         assert seg_pred is not None, f'{qset_uuid}_{seg_idx}_{num_segments}'
    #         new_ret_bboxes.append(seg_pred['ret_bboxes'])
    #         new_ret_scores.append(seg_pred['ret_scores'])
    #         frame_idxs.append(seg_pred['frame_idxs'])
    #     frame_idxs = torch.cat(frame_idxs, dim=0)
    #     mask_duplicated = frame_idxs == torch.cat([torch.tensor([-1]), frame_idxs[:-1]])
    #     qset_preds[qset_uuid] = {
    #         'ret_bboxes': torch.cat(new_ret_bboxes, dim=0)[~mask_duplicated].numpy(),
    #         'ret_scores': torch.cat(new_ret_scores, dim=0)[~mask_duplicated].numpy(),
    #     }

    # json.dump(qset_preds, p_int_pred.open('w'))
    qset_preds = torch.load(p_int_pred, weights_only=True)

    # get final predictions
    final_preds = get_final_preds(qset_preds)

    # write the final predictions to json
    # json.dump(final_preds, open(p_pred, 'w'))
    json.dump(final_preds, open('/tmp/pred.json', 'w'))

    # print metrics
    subset_metrics = get_metrics('data/vq_v2_val_anno.json', '/tmp/pred.json')
    print_metrics(subset_metrics)
