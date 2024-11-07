import json

import torch

import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, medfilt


def compute_response_track(preds, plateau_threshold_ratio=0.7):
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
    plateau_threshold = probs_sm[last_valid_peak] * plateau_threshold_ratio
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


def get_final_preds(preds, split='val', plateau_threshold_ratio=0.7):
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
        bboxes, fno_s, _ = compute_response_track(qset_pred, plateau_threshold_ratio)  # bboxes and start fno of the response track
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


def fix_predictions_order(final_preds, p_official_ann):
    # Load the ground truth and predictions
    with open(p_official_ann, "r") as fp:
        ann = json.load(fp)
    # Extract the video_uid list from the ground truth
    gt_video_uids = [v["video_uid"] for v in ann["videos"]]
    # Extract the clips for each video from the ground truth
    gt_clips_dict = {
        v["video_uid"]: {clip["clip_uid"]: clip for clip in v["clips"]}
        for v in ann["videos"]
    }
    # Extract video predictions and create a dictionary for faster lookup
    video_predictions = final_preds["results"]["videos"]
    video_predictions_dict = {v["video_uid"]: v for v in video_predictions}
    # Combine predictions and extra data (empty entries for missing video_uids)
    combined_predictions = []
    for uid in gt_video_uids:
        if uid in video_predictions_dict:
            # Get the prediction for this video
            vpred = video_predictions_dict[uid]

            # Sort clips based on ground truth clip order
            if "clips" in vpred and uid in gt_clips_dict:
                gt_clips = gt_clips_dict[uid]
                vpred_clips_dict = {clip["clip_uid"]: clip for clip in vpred["clips"]}

                # Ensure all ground truth clips are present in predictions
                sorted_clips = []
                for clip_uid, gt_clip in gt_clips.items():
                    if clip_uid in vpred_clips_dict:
                        pred_clip = vpred_clips_dict[clip_uid]
                        # Ensure number of predictions matches the number of annotations
                        num_annotations = len(gt_clip["annotations"])
                        num_predictions = len(pred_clip["predictions"])
                        if num_predictions < num_annotations:
                            # Add empty predictions if there are fewer predictions than annotations
                            for _ in range(num_predictions, num_annotations):
                                pred_clip["predictions"].append({"query_sets": {}})

                        sorted_clips.append(pred_clip)
                    else:
                        # Add an empty clip with the same number of empty predictions as annotations
                        sorted_clips.append({
                            "clip_uid": clip_uid,
                            "predictions": [{"query_sets": {}} for _ in range(len(gt_clip["annotations"]))]
                        })
                vpred["clips"] = sorted_clips
            else:
                # If no clips exist in predictions, create empty clips for the video
                vpred["clips"] = [
                    {
                        "clip_uid": clip_uid,
                        "predictions": [{"query_sets": {}} for _ in range(len(gt_clips_dict[uid][clip_uid]["annotations"]))]
                    }
                    for clip_uid in gt_clips_dict[uid]
                ]
            combined_predictions.append(vpred)
        else:
            # Add missing video entries as empty predictions
            combined_predictions.append({
                "video_uid": uid,
                "split": "test",
                "clips": [
                    {
                        "clip_uid": clip_uid,
                        "predictions": [{"query_sets": {}} for _ in range(len(gt_clips_dict[uid][clip_uid]["annotations"]))]
                    }
                    for clip_uid in gt_clips_dict[uid]
                ]
            })
    # Update the predictions in the model_predictions object
    final_preds["results"]["videos"] = combined_predictions
    return final_preds


if __name__ == '__main__':
    from pathlib import Path
    from ltvu.metrics import get_metrics_vq2d, print_metrics

    p_tmp_outdir = Path('outputs/debug/2024-09-25/126347/tmp')
    p_int_pred = p_tmp_outdir.parent / 'intermediate_predictions.pt'
    p_pred = p_tmp_outdir.parent / 'predictions.json'

    # json.dump(qset_preds, p_int_pred.open('w'))
    qset_preds = torch.load(p_int_pred, weights_only=True)

    # get final predictions
    final_preds = get_final_preds(qset_preds)

    # write the final predictions to json
    # json.dump(final_preds, open(p_pred, 'w'))
    json.dump(final_preds, open('/tmp/pred.json', 'w'))

    # print metrics
    subset_metrics = get_metrics_vq2d('data/vq_v2_val_anno.json', '/tmp/pred.json')
    print_metrics(subset_metrics)
