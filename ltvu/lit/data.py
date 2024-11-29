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
    LaSOTFitDataset, LaSOTEvalDataset,
)
from ltvu.preprocess.extract_frames import generate_flat_annotations_vq2d
from ltvu.utils.bbox_ops import check_bbox


# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class LitVQ2DDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the VQ2D dataset. It handles data preparation, augmentation,
    normalization, and loading for training, validation, and testing phases.

    Attributes
    ----------
    config : omegaconf.DictConfig
        Configuration object containing dataset and augmentation settings.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of worker processes for data loading.
    pin_memory : bool
        Whether to pin memory during data loading.
    prefetch_factor : int
        Number of batches to prefetch in the data loader.
    persistent_workers : bool
        Whether to keep workers alive between epochs.
    test_submit : bool
        Flag to determine if the test dataloader is for submission.
    eval_on_train : bool
        If True, evaluates on training split during validation.
    segment_aug : bool
        Whether to apply augmentation to video segments.
    strict_bbox_check : bool
        Whether to apply strict bounding box validation during augmentation.
    rt_pos_query : Any
        Configuration for real-time positional queries, if provided.

    Methods
    -------
    prepare_data()
        Prepares data by generating flat annotations and saving metadata.
    on_after_batch_transfer(batch, dataloader_idx)
        Applies GPU-accelerated preprocessing to the batch.
    augment(segments, gt_bboxes)
        Augments video segments and bounding boxes using Kornia augmentations.
    normalize(segment, query)
        Normalizes input video segments and query frames.
    denormalize(segment, query)
        Denormalizes input video segments and query frames.
    train_dataloader(shuffle=True)
        Returns the dataloader for the training split.
    val_dataloader()
        Returns the dataloader for the validation split.
    pred_dataloader()
        Returns the dataloader for prediction on the validation or training split.
    test_dataloader()
        Returns the dataloader for the test split.
    predict_dataloader()
        Returns the appropriate dataloader for prediction or test submission.
    get_val_sample(idx)
        Fetches a single sample from the validation split for debugging.
    get_train_sample(idx)
        Fetches a single sample from the training split for debugging.
    """
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
        """
        Generates and saves flat annotations for the dataset, ensuring consistency
        in the number of annotations and clips.

        Raises
        ------
        AssertionError
            If the number of annotations or clips is inconsistent with expectations.
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
        """
        Applies GPU-accelerated preprocessing to the batch, including normalization
        and augmentation for training mode.

        Parameters
        ----------
        batch : dict
            A dictionary containing batch data, including 'segment', 'query', 
            'gt_bboxes', and 'gt_probs'.
        dataloader_idx : int
            Index of the dataloader being used.

        Returns
        -------
        dict
            Updated batch with preprocessed and normalized data.
        """
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

    def augment(self, segments: torch.Tensor, gt_bboxes: torch.Tensor):
        """
        Augments video segments and bounding boxes using Kornia augmentations.

        Parameters
        ----------
        segments : torch.Tensor
            Normalized video segments of shape [batch_size, num_frames, channels, height, width].
        gt_bboxes : torch.Tensor
            Normalized ground truth bounding boxes of shape [batch_size, num_frames, 4], 
            in yxyx format.

        Returns
        -------
        tuple
            - Augmented video segments (torch.Tensor).
            - Augmented ground truth bounding boxes (torch.Tensor).
            - Update mask indicating valid bounding boxes (torch.Tensor).
        """
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

    def normalize(self, segment, query):
        """
        Normalizes video segments and query frames using ImageNet statistics.

        Parameters
        ----------
        segment : torch.Tensor
            Video segments of shape [batch_size, num_frames, channels, height, width].
        query : torch.Tensor
            Query frames of shape [batch_size, channels, height, width].

        Returns
        -------
        tuple
            - Normalized segments (torch.Tensor).
            - Normalized query frames (torch.Tensor).
        """
        bsz = segment.shape[0]
        segment = rearrange(segment, 'b t c h w -> (b t) c h w')
        segment = self.normalization(segment)  # [b,t,c,h,w]
        segment = rearrange(segment, '(b t) c h w -> b t c h w', b=bsz)
        query = self.normalization(query)  # [b,c,h,w]
        return segment, query

    def denormalize(self, segment, query):
        """
        Denormalizes video segments and query frames to their original pixel values.
        This method is for visualization purposes only.

        Parameters
        ----------
        segment : torch.Tensor
            Normalized video segments of shape [batch_size, num_frames, channels, height, width].
        query : torch.Tensor
            Normalized query frames of shape [batch_size, channels, height, width].

        Returns
        -------
        tuple
            - Denormalized segments (torch.Tensor).
            - Denormalized query frames (torch.Tensor).
        """
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
