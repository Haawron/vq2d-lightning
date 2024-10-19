import json
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import torch.optim
import torch.utils.data

import lightning as L

import kornia
import kornia.augmentation as K
from einops import rearrange

from ltvu.dataset import (
    VQ2DFitDataset, VQ2DEvalDataset,
)
from ltvu.preprocess import generate_flat_annotations_vq2d
from ltvu.bbox_ops import check_bbox


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


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

        aug_config = config.augment
        self.segment_aug: bool = aug_config.segment.apply
        self.strict_bbox_check: bool = aug_config.strict_bbox_check
        
        self.rt_pos_query = config.get('rt_pos_query')
        self.lt_track_query = config.get('lt_track_query')
        
        self.save_hyperparameters(ignore='config')  # to avoid saving unresolved config as a hyperparameter
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True), logger=False)  # to save the config in the checkpoint

        # GPU accelerated data preprocessing
        self.normalization = kornia.enhance.Normalize(mean=MEAN, std=STD)
        self.transform_clip: K.AugmentationSequential = instantiate(aug_config.segment.aug_list)

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
            print(f'Found {len(flat_anns)} annotations in {split}.')
            p_ann = self.p_anns_dir / f'vq_v2_{split}_anno.json'
            if not p_ann.exists():
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
        segment, query, gt_bboxes, gt_probs = batch['segment'], batch['query'], batch['gt_bboxes'], batch['gt_probs']
        if self.trainer is not None:
            if self.trainer.training:
            # if self.trainer.training and not self.enable_rt_pos_query:
                if self.segment_aug:
                    segment, gt_bboxes, gt_probs_update = self.augment(segment, gt_bboxes)
                    self.trainer.strategy.barrier()
                    gt_probs = gt_probs.logical_and(gt_probs_update).float()
            else:
                batch.update({'segment_original': segment, 'query_original': query})
        segment, query = self.normalize(segment, query)
        batch.update({
            'segment': segment, 'query': query,
            'gt_bboxes': gt_bboxes, 'gt_probs': gt_probs})
        
        if self.rt_pos_query is not None and self.trainer is not None and self.trainer.training:
            rt_pos_queries = batch['experiment']['multi_query']['rt_pos_queries']  # [b, #Q, c, h, w]
            bsz = rt_pos_queries.shape[0]
            rt_pos_queries = rearrange(rt_pos_queries, 'b q c h w -> (b q) c h w')
            rt_pos_queries = self.normalization(rt_pos_queries)  # [b*#Q, c, h, w]
            rt_pos_queries = rearrange(rt_pos_queries, '(b q) c h w -> b q c h w', b=bsz)
            batch['rt_pos_queries'] = rt_pos_queries
            
        if self.lt_track_query is not None and self.trainer is not None and self.trainer.training:
            lt_track_query = batch['experiment']['multi_query']['lt_track_query']  # [b, c, h, w]
            bsz = lt_track_query.shape[0]
            lt_track_query = self.normalization(lt_track_query)  # [b, c, h, w]
            batch['lt_track_query'] = lt_track_query
            
        return batch

    def augment(self, segments: torch.Tensor, gt_bboxes: torch.Tensor):   # TODO: static
        """Augment the segment and gt_bboxes.
        Ensure both segment and gt_bboxes are normalized. And an input bbox of the augment op should be in pixel space. Be aware that `segment` don't have to be in pixel space, only its shape matters. `kornia.augmentation.AugmentationSequential` takes `[B, C, H, W]` for segments and `[B, N, 4]` for bboxes as input. Each augmentation is applied to both segment and gt_bboxes simultaneously. So, the bounding boxes are consistent with the augmented segment.

        Parameters
        ----------
        segment : torch.Tensor
            Normalized segment of shape `[b, t, c, h, w]`.
        gt_bboxes : torch.Tensor
            Normalized bounding boxes of shape `[b, t, 4]`, format yxyx.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Augmented segment and gt_bboxes.
        """
        _, _, _, h, w = segments.shape
        device = segments.device

        # setup
        gt_bboxes = gt_bboxes[..., [1, 0, 3, 2]]  # [b,t,4], yxyx -> xyxy
        bbox_scale = torch.tensor([h, w, h, w], dtype=gt_bboxes.dtype, device=device)  # [4]
        bboxes_px = gt_bboxes * bbox_scale[None, None]  # [b,t,4], pixel space
        bboxes_px = bboxes_px.unsqueeze(2)  # [b,t,4] -> [b,t,N=1,4]
        bboxes_px = bboxes_px[..., [[0, 1], [2, 1], [2, 3], [0, 3]]]  # [b,t,1,4,2], xyxy -> 4 points, clockwise

        # augment
        segments_aug, bboxes_px_aug = self.transform_clip(segments, bboxes_px)  # [b,t,c,h,w], [b,t,1,4]

        # recover
        bboxes_px_aug = bboxes_px_aug.squeeze(2)  # [b,t,4]
        bboxes_px_aug = bboxes_px_aug[..., [1, 0, 3, 2]]  # [b,t,4], xyxy -> yxyx
        bboxes_px_aug, gt_probs_update = check_bbox(bboxes_px_aug, h, w, self.strict_bbox_check)  # [b,t,4], [b,t]
        gt_bboxes = bboxes_px_aug / bbox_scale[None, None]  # [b,t,4]

        return segments_aug, gt_bboxes, gt_probs_update

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
            VQ2DEvalDataset(self.config, split='val'),
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
            return torch.optim.adam.Adam(self.parameters(), lr=1e-3)

    dev_plm = DevModel()
    pdm.num_workers = 8
    pdm.batch_size = 16  # 32 causes OOM on CPU
    dev_trainer.fit(dev_plm, datamodule=pdm)
    print('Done.')
