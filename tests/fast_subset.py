import sys
sys.path.append('.')
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ltvu.metrics import get_metrics_vq2d, print_metrics_vq2d


def calculate_diff(x1, x2, y1, y2, w1, w2, h1, h2, ori_w, ori_h, mode = 'l2_distance'):
    x1 = np.maximum(x1, 0)
    x2 = np.maximum(x2, 0)
    y1 = np.maximum(y1, 0)
    y2 = np.maximum(y2, 0)
    w1 = np.maximum(w1, 0)
    w2 = np.maximum(w2, 0)
    h1 = np.maximum(h1, 0)
    h2 = np.maximum(h2, 0)
    x_diff = (x2 - x1) / ori_w
    y_diff = (y2 - y1) / ori_h
    w_diff = (w2 - w1) / ori_w
    h_diff = (h2 - h1) / ori_h

    position_diff = (x_diff ** 2 + y_diff ** 2) ** 0.5
    size_diff = (w_diff ** 2 + h_diff ** 2) ** 0.5
    if mode == 'add':
        all_diff = position_diff + size_diff
    elif mode == 'add2':
        all_diff = (x_diff ** 2 + y_diff ** 2 + w_diff ** 2 + h_diff ** 2) ** 0.5
    elif mode == 'mul':
        all_diff = position_diff * size_diff
    elif mode == 'harmonic_mean':
        area1 = w1 * h1
        area2 = w2 * h2
        area_hm = 2 * area1 * area2 / (area1 + area2 + 1e-6)

        # Normalize changes by image dimensions if applicable
        x_diff = (x2 - x1) / (area_hm ** 0.5 + 1e-6)
        y_diff = (y2 - y1) / (area_hm ** 0.5 + 1e-6)
        w_diff = (w2 - w1) / (area_hm ** 0.5 + 1e-6)
        h_diff = (h2 - h1) / (area_hm ** 0.5 + 1e-6)
    elif mode == 'use_log':
        x_diff = np.log(1 + x2 / (x1 + 1e-6))
        y_diff = np.log(1 + y2 / (y1 + 1e-6))
        w_diff = np.log(1 + w2 / (w1 + 1e-6))
        h_diff = np.log(1 + h2 / (h1 + 1e-6))

        all_diff = np.abs(x_diff) + np.abs(y_diff) + np.abs(w_diff) + np.abs(h_diff)
    elif mode == 'l1_distance':
        w_diff = np.log(w2 / w1)
        h_diff = np.log(h2 / h1)

        all_diff = np.abs(x_diff) + np.abs(y_diff) + np.abs(w_diff) + np.abs(h_diff)
    elif mode == 'l1_distance2':
        all_diff = np.abs(x_diff) + np.abs(y_diff) + np.abs(w_diff) + np.abs(h_diff)
    elif mode == 'l2_distance':
        w_diff = np.log(w2 / w1 + 1e-6)
        h_diff = np.log(h2 / h1 + 1e-6)
        all_diff = (x_diff ** 2 + y_diff ** 2 + w_diff ** 2 + h_diff ** 2) ** 0.5
    elif mode == 'l2_xy':
        all_diff = (x_diff ** 2 + y_diff ** 2) ** 0.5
    elif mode == 'l2_xywh':
        w_diff = np.minimum(w2 / w1, w1 / w2)
        h_diff = np.minimum(h2 / h1, h1 / h2)
        all_diff = (x_diff ** 2 + y_diff ** 2 + w_diff ** 2 + h_diff ** 2) ** 0.5
    elif mode ==  'l2_xy_bbox':
        x_diff = (x2 - x1) / ((w1 + w2) / 2)
        y_diff = (y2 - y1) / ((h1 + h2) / 2)
        all_diff = (x_diff ** 2 + y_diff ** 2) ** 0.5

    return all_diff


def main():
    for split in ['train', 'val']:
        print('=' * 80)
        print(split)
        p_ann = Path(f"data/vq_v2_{split}_anno.json")
        anns = json.load(p_ann.open())

        quid2ann = {
            f'{ann["annotation_uid"]}_{ann["query_set"]}': ann
            for ann in anns
        }

        all_diffs = {}
        for ann in anns:
            df_gt_rt = pd.DataFrame.from_records(ann['response_track'])
            df_gt_rt['cx'] = df_gt_rt['x'] + df_gt_rt['w'] / 2
            df_gt_rt['cy'] = df_gt_rt['y'] + df_gt_rt['h'] / 2
            qset_uuid = f'{ann["annotation_uid"]}_{ann["query_set"]}'
            if df_gt_rt.shape[0] < 2:
                all_diffs[qset_uuid] = 999999
                continue
            diffs = calculate_diff(
                df_gt_rt['cx'].iloc[:-1].reset_index(drop=True),
                df_gt_rt['cx'].iloc[1:].reset_index(drop=True),
                df_gt_rt['cy'].iloc[:-1].reset_index(drop=True),
                df_gt_rt['cy'].iloc[1:].reset_index(drop=True),
                df_gt_rt['w'].iloc[:-1].reset_index(drop=True),
                df_gt_rt['w'].iloc[1:].reset_index(drop=True),
                df_gt_rt['h'].iloc[:-1].reset_index(drop=True),
                df_gt_rt['h'].iloc[1:].reset_index(drop=True),
                ori_w=ann['original_width'], ori_h=ann['original_height'], mode='l2_xy_bbox')
            all_diffs[qset_uuid] = diffs.dropna().median()
            if np.isnan(all_diffs[qset_uuid]):
                print(qset_uuid)
                print(diffs)
                break

        diff_thres1, diff_thres2 = np.percentile(list(all_diffs.values()), [30, 60])
        print(f'Thresholds: {diff_thres1:.4f}, {diff_thres2:.4f}')

        p_tmp_ann = Path('/tmp/tmp_ann.json')
        ann_subsets = {
            'slow': [],
            'medium': [],
            'fast': []
        }

        for qset_uuid, max_diff in all_diffs.items():
            subset_name = 'slow' if max_diff < diff_thres1 else 'medium' if max_diff < diff_thres2 else 'fast'
            ann_subsets[subset_name].append(quid2ann[qset_uuid])

        for exp in ['clean', 'noised']:
            print()
            print('-' * 40)
            print(exp)
            print()
            p_pred = {
                'train': {
                    'clean': Path('outputs/ckpts/130094/vq2d/train/predictions.json'),
                    'noised': Path('outputs/debug/2024-11-12/35043/vq2d/predictions.json'),
                },
                'val': {
                    'clean': Path('outputs/ckpts/130094/predictions.json'),
                    'noised': Path('outputs/debug/2024-11-12/35042/vq2d/predictions.json'),
                }
            }[split][exp]

            for subset_name in ann_subsets:
                print(subset_name)
                p_tmp_ann.write_text(json.dumps(ann_subsets[subset_name]))
                metrics = get_metrics_vq2d(p_tmp_ann, p_pred)
                # print_metrics_vq2d(metrics)
                print_metrics_vq2d({'all': metrics['all']})
                print()

    print('=' * 80)


if __name__ == '__main__':
    main()
