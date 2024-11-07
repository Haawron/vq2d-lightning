"""
Usage

---

python tests/egotracks_flat.py
"""

import sys
sys.path.append('.')
import json
from pathlib import Path
from pprint import pprint

from ltvu.preprocess import generate_flat_annotations_egotracks


if __name__ == '__main__':
    p_anns_out = Path('./data/egotracks')
    p_anns_out.mkdir(exist_ok=True, parents=True)

    p_egotracks = Path('/data/datasets/ego4d_data/v2/egotracks')

    p_ann = p_egotracks / 'egotracks_train.json'
    print(p_ann)
    anns_flat = generate_flat_annotations_egotracks(p_ann)
    print(len(anns_flat))
    print(len({q['uuid_ltt'] for q in anns_flat if 'clip_uid' in q}))
    print(len({q['clip_uid'] for q in anns_flat if 'clip_uid' in q}))
    json.dump(anns_flat, open(p_anns_out / 'egotracks_train_anno.json', 'w'), indent=1)
    print()

    sample = anns_flat[0]

    p_ann = p_egotracks / 'egotracks_val.json'
    print(p_ann)
    anns_flat = generate_flat_annotations_egotracks(p_ann)
    print(len(anns_flat))
    print(len({q['uuid_ltt'] for q in anns_flat if 'clip_uid' in q}))
    print(len({q['clip_uid'] for q in anns_flat if 'clip_uid' in q}))
    json.dump(anns_flat, open(p_anns_out / 'egotracks_val_anno.json', 'w'), indent=1)
    print()

    p_ann = p_egotracks / 'egotracks_challenge_test_unannotated.json'
    print(p_ann)
    anns_flat = generate_flat_annotations_egotracks(p_ann)
    print(len(anns_flat))
    print(len({q['uuid_ltt'] for q in anns_flat if 'clip_uid' in q}))
    print(len({q['clip_uid'] for q in anns_flat if 'clip_uid' in q}))
    json.dump(anns_flat, open(p_anns_out / 'egotracks_challenge_test_unannotated_anno.json', 'w'), indent=1)
    print()

    pprint(sample, indent=2)
