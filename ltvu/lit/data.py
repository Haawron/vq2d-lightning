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
    EgoTracksFitDataset, EgoTracksEvalDataset,
    LaSOTFitDataset, LaSOTEvalDataset,
)
from ltvu.preprocess import generate_flat_annotations_vq2d, generate_flat_annotations_egotracks
from ltvu.bbox_ops import check_bbox


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class LitVQ2DDataModule(L.LightningDataModule):
    # ALL_NUM_CLIPS = 4690  # train + val
    # ALL_NUM_ANNS = [13607, 4504]  # train, val
    ALL_NUM_CLIPS = 5814  # train + val + test_unannotated
    ALL_NUM_ANNS = [13607, 4504, 4461]  # train, val, test_unannotated

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
        self.test_submit = ds_config.get('test_submit', False)
        self.eval_on_train = ds_config.get('eval_on_train', False)
        self.movement = ds_config.get('movement', "")

        aug_config = config.augment
        self.segment_aug: bool = aug_config.segment.apply
        self.strict_bbox_check: bool = aug_config.strict_bbox_check

        self.rt_pos_query = config.get('rt_pos_query')

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
        for split, desired_num_anns in zip(['train', 'val', 'test_unannotated'], self.ALL_NUM_ANNS):
            p_official_ann = self.p_official_anns_dir / f'vq_{split}.json'
            flat_anns = generate_flat_annotations_vq2d(p_official_ann)
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
            batch['rt_pos_idx'] = batch['experiment']['multi_query']['rt_pos_idx']
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

    def denormalize(self, segment, query):  # TODO: static
        bsz = segment.shape[0]
        denorm = kornia.enhance.Denormalize(
            mean=torch.tensor(MEAN, device=segment.device, dtype=segment.dtype),
            std=torch.tensor(STD, device=segment.device, dtype=segment.dtype),)
        segment = rearrange(segment, 'b t c h w -> (b t) c h w')
        segment = denorm(segment)  # [b,t,c,h,w]
        segment = rearrange(segment, '(b t) c h w -> b t c h w', b=bsz)
        query = denorm(query)  # [b,c,h,w]
        return segment, query

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            VQ2DFitDataset(self.config, split='train', movement=self.movement),
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
            VQ2DFitDataset(self.config, split='val', movement=self.movement),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def pred_dataloader(self):
        return torch.utils.data.DataLoader(
            VQ2DEvalDataset(self.config, split='train' if self.eval_on_train else 'val', movement=self.movement),
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
            VQ2DEvalDataset(self.config, split='test_unannotated'),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def predict_dataloader(self):
        if self.test_submit:
            return self.test_dataloader()
        else:
            return self.pred_dataloader()

    def get_val_sample(self, idx):
        """Get a single sample from the validation set as a batch for debugging."""
        ds = VQ2DFitDataset(self.config, split='val')
        ds.all_anns = ds.all_anns[idx:idx+1]
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=1,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return next(iter(dl))

    def get_train_sample(self, idx):
        """Get a single sample from the validation set as a batch for debugging."""
        ds = VQ2DFitDataset(self.config, split='train')
        ds.all_anns = ds.all_anns[idx:idx+1]
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=1,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return next(iter(dl))


class LitEgoTracksDataModule(LitVQ2DDataModule):
    ALL_NUM_CLIPS = 3537 + 1157 + 66  # train + val + challenge_test_unannotated
    ALL_NUM_ANNS = [13129 + 13619, 4468 + 4504, 1 + 244]  # train, val, challenge_test_unannotated

    def prepare_data(self):
        """Calls generate_flat_annotations_vq2d to save the flat annotations.
        And report the number of video uids and clip uids.
        """
        print('Preparing data...')
        video_uids, clip_uids = set(), set()
        for split, desired_num_anns in zip(['train', 'val', 'challenge_test_unannotated'], self.ALL_NUM_ANNS):
            p_ann = self.p_anns_dir / f'egotracks_{split}_anno.json'
            p_official_ann = self.p_official_anns_dir / f'egotracks_{split}.json'
            if p_ann.exists():
                flat_anns = json.load(p_ann.open())
            else:
                flat_anns = generate_flat_annotations_egotracks(p_official_ann)
            assert len(flat_anns) == desired_num_anns, f'Split {split} has {len(flat_anns)} annotations, expected {desired_num_anns}'
            print(f'Found {len(flat_anns)} annotations in {split}.')
            if not p_ann.exists():
                json.dump(flat_anns, p_ann.open('w'))
            for ann in flat_anns:
                video_uids.add(ann['video_uid'])
                if 'clip_uid' in ann:
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

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            EgoTracksFitDataset(self.config, split='train'),
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
            EgoTracksFitDataset(self.config, split='val'),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def pred_dataloader(self):
        return torch.utils.data.DataLoader(
            EgoTracksEvalDataset(self.config, split='val'),
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
            EgoTracksEvalDataset(self.config, split='challenge_test_unannotated'),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )


class LitLaSOTDataModule(LitVQ2DDataModule):
    ALL_NUM_CLIPS = 1120 + 280  # train + test
    ALL_NUM_ANNS = [1120, 280]  # train, test

    def prepare_data(self):
        pass

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            LaSOTFitDataset(self.config, split='train'),
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
            LaSOTFitDataset(self.config, split='test'),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def pred_dataloader(self):
        return torch.utils.data.DataLoader(
            LaSOTEvalDataset(self.config, split='test'),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        raise NotImplementedError


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
