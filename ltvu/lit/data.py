import json
from pathlib import Path

import torch
import torch.utils.data

import lightning as L

import kornia
import kornia.augmentation as K
from kornia.constants import DataKey
from einops import rearrange
from PIL import Image, ImageDraw

from ltvu.dataset import (
    VQ2DFitDataset, VQ2DTestDatasetSeparated,
)
from ltvu.preprocess import generate_flat_annotations_vq2d


class LitVQ2DDataModule(L.LightningDataModule):
    ALL_NUM_CLIPS = 4690  # train + val
    ALL_NUM_ANNS = [13607, 4504]  # train, val

    def __init__(self, config):
        super().__init__()
        self.config = config
        ds_config = self.config.dataset
        self.p_clips_dir = Path(ds_config.clips_dir)
        self.p_official_anns_dir = Path(ds_config.official_anns_dir)
        self.p_anns_dir = Path(ds_config.flat_anns_dir)
        self.batch_size = ds_config.batch_size
        self.num_workers = ds_config.num_workers
        self.pin_memory = ds_config.pin_memory
        self.prefetch_factor = ds_config.prefetch_factor
        self.persistent_workers = ds_config.persistent_workers

        # TODO: Integrate into the ds_config
        aug_config = config.train.augments
        self.segment_aug: bool = aug_config.segment
        self.segment_size: tuple[int] = tuple(aug_config.segment_size)  # h, w
        self.crop_ratio_min = aug_config.crop_ratio_min
        self.crop_ratio_max = aug_config.crop_ratio_max
        cr_ratio = self.crop_ratio_min, self.crop_ratio_max

        self.save_hyperparameters()

        # GPU accelerated data preprocessing
        self.normalization = kornia.enhance.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_clip = K.AugmentationSequential(
            K.ColorJiggle(brightness=0.4, contrast=0.4, saturation=0.3, hue=0, p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomResizedCrop(self.segment_size, scale=(0.66, 1.0), ratio=cr_ratio, p=1.0),
            data_keys=[DataKey.INPUT, DataKey.BBOX_XYXY],  # Just to define the future input here.
            same_on_batch=True)

    def prepare_data(self):
        """Calls generate_flat_annotations_vq2d to save the flat annotations.
        And report the number of video uids and clip uids.
        """
        print('Preparing data...')
        video_uids, clip_uids = set(), set()
        for split, desired_num_anns in zip(['train', 'val'], self.ALL_NUM_ANNS):
            p_raw_ann = self.p_official_anns_dir / f'vq_{split}.json'
            all_anns = json.load(p_raw_ann.open())
            flat_anns = generate_flat_annotations_vq2d(all_anns)
            assert len(flat_anns) == desired_num_anns, f'Split {split} has {len(flat_anns)} annotations, expected {desired_num_anns}'
            p_ann = self.p_anns_dir / f'vq_v2_{split}_anno.json'
            json.dump(flat_anns, p_ann.open('w'))
            for ann in flat_anns:
                video_uids.add(ann['video_uid'])
                clip_uids.add(ann['clip_uid'])
        assert len(clip_uids) == self.ALL_NUM_CLIPS, f'Expected {self.ALL_NUM_CLIPS} clips, got {len(clip_uids)}'
        p_video_uids = self.p_anns_dir / 'video_uids.txt'
        p_clip_uids = self.p_anns_dir / 'clip_uids.txt'
        p_video_uids.write_text(' '.join(sorted(video_uids)))
        p_clip_uids.write_text(' '.join(sorted(clip_uids)))
        print(f'Found {len(video_uids)} video uids and {len(clip_uids)} clip uids.')
        print(f'Video uids are saved in {p_video_uids}')
        print(f'Clip uids are saved in {p_clip_uids}')
        print('Data preparation done.')

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """GPU accelerated data preprocessing"""
        segment, query, gt_bboxes = batch['segment'], batch['query'], batch['gt_bboxes']
        if self.trainer.training and self.segment_aug:
            segment, gt_bboxes = self.augment(segment, gt_bboxes)
        # TODO: augment query as well
        segment, query = self.normalize(segment, query)
        batch.update({'segment': segment, 'query': query, 'gt_bboxes': gt_bboxes})
        return batch

    def augment(self, segment: torch.Tensor, gt_bboxes: torch.Tensor):   # TODO: static
        """Augment the segment and gt_bboxes.
        Ensure both segment and gt_bboxes are normalized. And an input bbox of the augment op should be in pixel space. Be aware that `segment` don't have to be in pixel space, only its shape matters. `kornia.augmentation.AugmentationSequential` takes `[B, C, H, W]` for segments and `[B, N, 4]` for bboxes as input. Each augmentation is applied to both segment and gt_bboxes simultaneously. So, the bounding boxes are consistent with the augmented segment.

        Parameters
        ----------
        segment : torch.Tensor
            Normalized segment of shape `[b, t, c, h, w]`.
        gt_bboxes : torch.Tensor
            Normalized bounding boxes of shape `[b, t, 4]`, format xyxy.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Augmented segment and gt_bboxes.
        """
        h, w = self.segment_size
        bsz, device = segment.shape[0], segment.device

        bbox_scale = torch.tensor([h, w, h, w], device=device)[None, None]  # [1,1,4]
        bboxes_px = gt_bboxes * bbox_scale  # [b,t,4], pixel space
        seg_queue, bboxes_queue = [], []
        # DON'T FLATTEN AND PRALLELIZE THIS LOOP
        #    AS t IS TREATED AS BATCH AXIS
        #    SO AUGMENTATIONS SHOULD BE CONSISTENT ACROSS t.
        #    `same_on_batch=True` IS SET BUT THIS LOOP MAKES AUGS FOR EACH SAMPLE DIFFERENT WHICH IS DESIRED.
        # FLATTENING WILL MAKE AUGMENTATIONS FOR THIS BATCH ALL THE SAME.
        for i in range(bsz):  # augs are different for each sample (desired)
            # augs are consistent across t (desired)
            # [t,c,h,w] -> [c,h,w], [b,t,4] -> [t,4] -> [t,N=1,4]
            segment_aug, bboxes_aug = self.transform_clip(segment[i], bboxes_px[i, :, None])
            bboxes_aug = bboxes_aug / bbox_scale  # [t,4]
            bboxes_aug = bboxes_aug.squeeze(1)  # [t,4]
            # imgs = []
            # for t in range(segment_aug.shape[0]):
            #     img = Image.fromarray(segment_aug[t].permute(1, 2, 0).mul(255).byte().cpu().numpy())
            #     draw = ImageDraw.Draw(img)
            #     bbox = bboxes_aug[t].mul(torch.tensor([w, h, w, h], device=device))
            #     draw.rectangle(bbox.tolist(), outline='red', width=5)
            #     imgs.append(img)
            # p_img = Path(f'outputs/dataset/seg-{device}-{i}.gif')
            # imgs[0].save(p_img, save_all=True, append_images=imgs[1:], duration=100, loop=0)
            # print(f'bbox: {p_img} {bbox}')
            seg_queue.append(segment_aug)
            bboxes_queue.append(bboxes_aug)
        segment = torch.stack(seg_queue, dim=0)
        gt_bboxes = torch.stack(bboxes_queue, dim=0)  # [b,t,4]
        return segment, gt_bboxes

    def normalize(self, segment, query):  # TODO: static
        bsz = segment.shape[0]
        segment = rearrange(segment, 'b t c h w -> (b t) c h w')
        segment = self.normalization(segment)  # [b,t,c,h,w]
        segment = rearrange(segment, '(b t) c h w -> b t c h w', b=bsz)
        query = self.normalization(query)  # [b,c,h,w]
        return segment, query

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            VQ2DFitDataset(self.config, split='train'),
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            VQ2DFitDataset(self.config, split='val'),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            VQ2DTestDatasetSeparated(self.config, split='val'),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )

    predict_dataloader = test_dataloader


if __name__ == '__main__':
    import os
    os.environ["SLURM_JOB_NAME"] = "bash"
    from hydra import compose, initialize
    import torch.distributed as dist

    L.seed_everything(42)
    with initialize(config_path="../../config", version_base='1.3'):
        config = compose(config_name="base")

    pdm = LitVQ2DDataModule(config)
    dev_trainer = L.Trainer(fast_dev_run=int(1e6))

    if not (dist.is_available() and dist.is_initialized()):
        config.dataset.num_workers = 1  # for debugging
        pdm.prepare_data()

        print('Checking the first batch...')
        for batch in pdm.train_dataloader():
            print('keys: ')
            print('\t', batch.keys())
            print('Shape of a segment:', batch['segment'].shape)
            print('Shape of a query:', batch['query'].shape)
            b, t, c, h, w = batch['segment'].shape
            segments = rearrange(batch['segment'], 'b t c h w -> (b t) c h w')
            break
        print('Done.')

    print('Checking the first epoch...')
    class DevModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.randn(1))
        def training_step(self, batch, batch_idx):
            return self.x ** 2
        val_step = training_step
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    dev_plm = DevModel()
    pdm.num_workers = 8
    pdm.batch_size = 16  # 32 causes OOM on CPU
    dev_trainer.fit(dev_plm, datamodule=pdm)
    print('Done.')
