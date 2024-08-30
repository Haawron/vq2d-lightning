from pathlib import Path
from collections import defaultdict
import numpy as np
import json

from ltvu.structures import ResponseTrack


# clip level
def compute_tious(all_preds: dict[list[ResponseTrack]], all_pos: dict[ResponseTrack]):
    """
    Bboxes are in the format of xyhw, aspect to the input clip size.

    Parameters
    ----------
    all_preds : qset_uuid -> list of predicted response tracks
    gts : qset_uuid -> GT response track
    """
    all_qset_uuids = set(all_preds.keys())
    assert all_qset_uuids == set(all_pos.keys())
    all_qset_uuids = sorted(all_qset_uuids)

    tious = {}
    for qset_uuid in all_qset_uuids:
        pos_rt = all_pos[qset_uuid]
        pred_rt = max(all_preds[qset_uuid], key=lambda rt: rt.score)  # highest score
        pos_ext, pred_ext = pos_rt.temporal_extent, pred_rt.temporal_extent
        tinter = pos_ext.intersection(pred_ext)
        tunion = pos_ext.union(pred_ext)
        tiou = tinter / tunion
        tious[qset_uuid] = tiou
    return tious


# clip level
def compute_stious(all_preds: dict[list[ResponseTrack]], all_pos: dict[ResponseTrack]):
    all_qset_uuids = set(all_preds.keys())
    assert all_qset_uuids == set(all_pos.keys())
    all_qset_uuids = sorted(all_qset_uuids)

    stious = {}
    for qset_uuid in all_qset_uuids:
        pos_rt = all_pos[qset_uuid]
        pred_rt = max(all_preds[qset_uuid], key=lambda rt: rt.score)  # highest score
        stinter = pos_rt.intersection(pred_rt)
        stunion = pos_rt.union(pred_rt)
        stiou = stinter / stunion
        stious[qset_uuid] = stiou
    return stious


# frame level
def compute_sious_for_positives(all_preds: dict[list[ResponseTrack]], all_pos: dict[ResponseTrack]):
    all_qset_uuids = set(all_preds.keys())
    assert all_qset_uuids == set(all_pos.keys())
    all_qset_uuids = sorted(all_qset_uuids)

    all_sious: dict[str, list[float]] = defaultdict(list)  # qset_uuid -> list of sious for each GT frame
    all_areas: dict[str, list[float]] = defaultdict(list)  # qset_uuid -> list of areas for each GT frame
    for qset_uuid in all_qset_uuids:
        pos_rt = all_pos[qset_uuid]
        pred_rt = max(all_preds[qset_uuid], key=lambda rt: rt.score)  # highest score
        pos_ext, pred_ext = pos_rt.temporal_extent, pred_rt.temporal_extent
        for fno in pos_ext:
            pos_bbox = pos_rt[fno]
            pred_bbox = pred_rt.get(fno)
            sinter = pos_bbox.intersection(pred_bbox)
            sunion = pos_bbox.union(pred_bbox)
            if sunion == 0:
                print(qset_uuid, pos_bbox, pred_bbox)
            siou = sinter / sunion
            all_sious[qset_uuid].append(siou)
            all_areas[qset_uuid].append(pos_bbox.area)
    return all_sious, all_areas  # all sious for positives, all areas for positives


def compute_average_precision_dict(
    ious: np.ndarray,  # [N_samples]
    thresholds: np.ndarray = np.array([.25, .5, .75, .95])
):
    num_samples = ious.shape[0]
    thresholds = np.array(thresholds)
    tp = ious >= thresholds[..., None]  # [N_ths, N_samples]
    precisions = tp.cumsum(axis=1) / np.arange(1, num_samples + 1)  # [N_ths, N_samples], increasing and decreasing
    recalls = tp.cumsum(axis=1) / num_samples  # [N_ths, N_samples], increasing
    rec_diffs = np.diff(recalls, prepend=0, axis=1)  # [N_ths, N_samples]
    aps = (precisions * rec_diffs).sum(axis=1)  # [N_ths]
    return {
        'mAP': aps.mean(),
        'APs': [{'threshold': th, 'AP': ap} for th, ap in zip(thresholds, aps)],
        'precisions': precisions,
        'recalls': recalls,
    }


if __name__ == '__main__':
    p_ann = Path("/data/gunsbrother/repos/vq2d-lightning/data/vq_v2_val_anno.json")
    p_pred = Path("/data/gunsbrother/prjs/ltvu2/VQLoC/notebooks/43634_results.json.gz")
    all_anns = json.load(p_ann.open())
    all_preds_stratified = json.load(p_pred.open())

    quid2pred, quid2pos = defaultdict(list), {}
    for ann in all_anns:
        qset_uuid = f'{ann["annotation_uid"]}_{ann["query_set"]}'
        ann['bboxes'] = ResponseTrack.from_json(ann)
        quid2pos[qset_uuid] = ann['bboxes']

    for pred_video in all_preds_stratified['results']['videos']:
        video_uid = pred_video['video_uid']
        for pred_clip in pred_video['clips']:
            clip_uid = pred_clip['clip_uid']
            for pred_qsets in pred_clip['predictions']:
                annotation_uid = pred_qsets['annotation_uid']
                for qset_id, qset in pred_qsets['query_sets'].items():
                    qset_uuid = f'{annotation_uid}_{qset_id}'
                    if qset_uuid in quid2pos:
                        quid2pred[qset_uuid].append(ResponseTrack.from_json(qset))

    # print(len(all_anns), len(quid2pred), len(quid2pos))

    quid2pos = {quid: pos for quid, pos in quid2pos.items()}
    quid2pred = {quid: pred for quid, pred in quid2pred.items()}

    tious: dict[str, float] = compute_tious(quid2pred, quid2pos)
    stious: dict[str, float] = compute_stious(quid2pred, quid2pos)
    all_sious, all_areas = compute_sious_for_positives(quid2pred, quid2pos)
    all_max_areas = {quid: max(areas) for quid, areas in all_areas.items()}

    # subsample GT samples based on their max bbox area of each RT
    subset_max_bbox_area_ranges_base = {
        'all':    (0, np.inf),
        'large':  (96 ** 2, np.inf),
        'medium': (32 ** 2, 96 ** 2),
        'small':  (0, 32 ** 2),
    }
    input_area_base = 224 ** 2
    input_area = 448 ** 2
    mul = input_area / input_area_base
    subset_max_bbox_area_ranges = {k: (mul*v[0], mul*v[1]) for k, v in subset_max_bbox_area_ranges_base.items()}

    # theoretically, python >3.5 dicts keep insertion order but sort it for safety
    all_quids_in_order = sorted(set(quid2pos.keys()))
    tious_arr = np.array([tious[quid] for quid in all_quids_in_order])
    stious_arr = np.array([stious[quid] for quid in all_quids_in_order])
    all_sious_arr = np.array([all_sious[quid] for quid in all_quids_in_order], dtype=list)  # flattened
    all_max_areas_arr = np.array([all_max_areas[quid] for quid in all_quids_in_order], dtype=list)  # flattened
    # subset_results = {}
    for subset_name, (lbd, ubd) in subset_max_bbox_area_ranges.items():
        max_area_mask = (lbd <= all_max_areas_arr) & (all_max_areas_arr < ubd)
        tap_dict = compute_average_precision_dict(tious_arr[max_area_mask])
        stap_dict = compute_average_precision_dict(stious_arr[max_area_mask])
        sap_dict = compute_average_precision_dict(np.array(sum(all_sious_arr[max_area_mask], [])), thresholds=[.5])
        succ_dict = compute_average_precision_dict(stious_arr[max_area_mask], thresholds=[.05])
        rec_dict = compute_average_precision_dict(tious_arr[max_area_mask], thresholds=[.3, .5])

        tap = tap_dict['mAP']
        taps = tap_dict['APs'][0]["AP"]
        stap = stap_dict['mAP']
        staps = stap_dict['APs'][0]["AP"]
        rec = sap_dict['recalls'][-1, -1]
        succ = succ_dict['precisions'][-1, -1]
        r1_03 = rec_dict['recalls'][0, -1]
        r1_05 = rec_dict['recalls'][1, -1]

        # display
        print(f'{subset_name} ({lbd**.5:.0f}^2 - {ubd**.5:.0f}^2) ({max_area_mask.sum()} samples / {max_area_mask.mean():.1%})')
        print('VQ2D Evaluation')
        print(f'tAP             : {tap:6.3f}')
        print(f'tAP  @ IoU=0.25 : {taps:6.3f}')
        print(f'stAP            : {stap:6.3f}')
        print(f'stAP @ IoU=0.25 : {staps:6.3f}')
        print(f'Recovery %      : {100*rec:6.3f}')
        print(f'Success         : {100*succ:6.3f}')
        print('EgoNLQ Evaluation')
        print(f'R1 @ IoU=.3     : {100*r1_03:6.3f}')
        print(f'R1 @ IoU=.5     : {100*r1_05:6.3f}')
        print()
