import math
from pprint import pprint

import hydra.utils
from omegaconf import OmegaConf

import torch
import kornia
import kornia.augmentation as K
from kornia.constants import DataKey
from einops import rearrange, repeat

import lightning as L

from .models import *


# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
class LitModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        aug_config = config.train.augments
        self.clip_aug: bool = aug_config.clip
        self.clip_size: list[int] = aug_config.clip_size
        self.crop_ratio_min = aug_config.crop_ratio_min
        self.crop_ratio_max = aug_config.crop_ratio_max

        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        self.model: VQLOC = hydra.utils.instantiate(config.model)

        # GPU accelerated data preprocessing
        self.normalization = kornia.enhance.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_clip = K.AugmentationSequential(
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0, p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomResizedCrop(self.clip_size, scale=(0.66, 1.0), ratio=(self.crop_ratio_min, self.crop_ratio_max), p=1.0),
            data_keys=[DataKey.INPUT, DataKey.BBOX_XYXY],  # Just to define the future input here.
            same_on_batch=True)

    def on_fit_start(self):
        if self.trainer.global_rank == 0:
            print('Starting training...')

    def training_step(self, batch, batch_idx):
        segment, query, gt_bboxes = batch['segment'], batch['query'], batch['gt_bboxes']
        if self.clip_aug:
            segment, gt_bboxes = self.augment(segment, gt_bboxes)
        segment, query = self.normalize(segment, query)
        total_loss, loss_dict, preds_top, gts = self.model.forward(**batch, training=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output_dict = self.model.forward(**batch)
        return output_dict

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        segment, query = batch['segment'], batch['query']
        segment, query = self.normalize(segment, query)
        results = self.model.forward(segment, query)

        pred_bbox, pred_prob = results['bbox'], results['prob']  # [b,t,N,4], [b,t,N]
        b, t, N = pred_prob.shape
        pred_prob = rearrange(pred_prob, 'b t N -> (b t) N')
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t) N c')
        pred_prob_top, top_idx = torch.max(pred_prob, dim=-1)    # [b*t], [b*t]
        top_idx = repeat(top_idx, 'B -> B n c', n=1, c=4)
        pred_bbox_top = torch.gather(pred_bbox, dim=1, index=top_idx).squeeze()   # [b*t,4]
        pred_top = {
            'bbox': rearrange(pred_bbox_top, '(b t) c -> b t c', b=b, t=t),
            'prob': rearrange(pred_prob_top, '(b t) -> b t', b=b, t=t)
        }
        return output_dict

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.optimizer = hydra.utils.instantiate(
            self.config.optim.optimizer, params=params)
        return self.optimizer

    def augment(self, segment, gt_bboxes):
        h, w = self.clip_size
        bsz, device = segment.shape[0], segment.device
        bbox_scale = torch.tensor([h, w, h, w], device=device)[None, None]  # [1,1,4]
        seg_queue, bboxes_queue = [], []
        for i in range(bsz):
            bboxes_px = gt_bboxes[i] * bbox_scale  # [t,4]
            segment_aug, bboxes_aug = self.transform_clip(segment[i], bboxes_px)
            bboxes_aug = bboxes_aug / bbox_scale  # [t,4]
            seg_queue.append(segment_aug)
            bboxes_queue.append(bboxes_aug)
        segment = torch.stack(seg_queue, dim=0)
        gt_bboxes = torch.stack(bboxes_queue, dim=0)  # [b,t,4]
        return segment, gt_bboxes

    def normalize(self, segment, query):
        bsz = segment.shape[0]
        segment = rearrange(segment, 'b t c h w -> (b t) c h w')
        segment = self.normalization(segment)  # [b,t,c,h,w]
        segment = rearrange(segment, '(b t) c h w -> b t c h w', b=bsz)
        query = self.normalization(query)  # [b,c,h,w]
        return segment, query
