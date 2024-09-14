import time
import hydra.utils
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
import torch.optim

import lightning as L
from lightning.pytorch.loggers import WandbLogger

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from transformers import get_linear_schedule_with_warmup

from ltvu.models import *


# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
class LitModule(L.LightningModule):
    """
    https://lightning.ai/docs/pytorch/stable/common/trainer.html#under-the-hood

    # Pseudo code for training loop

    ```python
    torch.set_grad_enabled(True)

    losses = []
    for batch in train_dataloader:
        on_train_batch_start()  # return -1 to skip the rest of the epoch
        loss = training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    ```

    ---

    # Pseudo code for validation loop

    ```python
    for batch_idx, batch in enumerate(train_dataloader):
        # ... model train loop
        if validate_at_some_point:  # when meet some condition
            torch.set_grad_enabled(False)  # disable grads + batchnorm + dropout
            model.eval()
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                val_out = model.validation_step(val_batch, val_batch_idx)  # -> should be handled in `on_validation_epoch_end`
            torch.set_grad_enabled(True)  # enable grads + batchnorm + dropout
            model.train()
    ```

    ---

    # Pseudo code for prediction loop

    ```python
    torch.set_grad_enabled(False)  # disable grads + batchnorm + dropout
    model.eval()
    all_preds = []
    for batch_idx, batch in enumerate(predict_dataloader):
        pred = model.predict_step(batch, batch_idx)
        all_preds.append(pred)  # -> will be stacked and returned in `trainer.predict`
    ```

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model: VQLoC = hydra.utils.instantiate(
            config.model, compile_backbone=config.get('compile', True))
        self.fix_backbone = config.model.fix_backbone
        self.save_hyperparameters(ignore='config')
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        self.sample_step = 0

    ############ major hooks ############

    def training_step(self, batch, batch_idx):
        bsz = batch['segment'].shape[0]
        output_dict = self.model.forward(**batch, compute_loss=True)
        assert output_dict['loss'].requires_grad
        assert torch.isfinite(output_dict['loss']), f'Loss is {output_dict["loss"]}'
        self.log_dict(set_prefix_to_keys(
            output_dict['log_dict'], 'Train'),
            batch_size=bsz, rank_zero_only=True)
        self.sample_step = (1 + self.global_step) * bsz * self.trainer.world_size
        self.log('sample_step', self.sample_step, batch_size=1, rank_zero_only=True)
        return output_dict['loss']

    def validation_step(self, batch, batch_idx):
        bsz = batch['segment'].shape[0]
        output_dict = self.model.forward(**batch, compute_loss=True, training=False)
        self.log_dict(set_prefix_to_keys(
            output_dict['log_dict'], 'Val'),
            batch_size=bsz, on_epoch=True, sync_dist=True)
        if self.sample_step > 0:  # after sanity check done
            if batch_idx % 50 == 0:
                self.print_outputs(batch, output_dict, bidxs=[0])
        self.trainer.strategy.barrier('validation_step_end')  # processing times may vary

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        optim_config = self.config.optim
        all_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(optim_config.optimizer, params=all_params)
        if sched_cfg := optim_config.lr_scheduler:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=sched_cfg.warmup_iter,
                num_training_steps=sched_cfg.max_steps)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'step'
                }
            }
        else:
            return optimizer


    ############ other hooks ############

    def on_fit_start(self):
        if self.global_rank == 0:
            print('Starting training...')

    def on_train_epoch_start(self):
        if self.fix_backbone:
            self.model.backbone.eval()

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)

    ############ helper functions ############

    def get_wnb_logger(self):
        return [l for l in self.loggers if isinstance(l, WandbLogger)][0]

    def print_outputs(self, batch, output_dict, bidxs=None):
        clip_uids = batch['clip_uid']
        annotation_uids = batch['annotation_uid']
        query_sets = batch['query_set']
        segments = batch['segment_original'].cpu().numpy()
        queries = batch['query_original'].cpu().numpy()
        vcs: dict[str, torch.Tensor] = batch['visual_crop']
        gt_bboxes = batch['gt_bboxes'].cpu().numpy()
        gt_probs = batch['gt_probs'].cpu().numpy()

        prob_pred_tops = output_dict['info_dict']['preds_top']['prob'].float().sigmoid().cpu().numpy()  # [b,t]
        prob_rts = output_dict['info_dict']['preds_top']['bbox'].float().cpu().numpy()  # [b,t,4]
        bsz, T, _, h, w = segments.shape
        p_preds_dir = Path(self.trainer.log_dir or 'outputs/debug') / 'preds'
        p_preds_dir.mkdir(parents=True, exist_ok=True)
        step = self.sample_step

        gt_bboxes_xyxy = gt_bboxes[..., [1, 0, 3, 2]]  # yxyx -> xyxy
        prob_rts_xyxy = prob_rts[..., [1, 0, 3, 2]]  # yxyx -> xyxy

        for bidx in range(bsz):
            if bidxs is not None and bidx not in bidxs:
                continue
            else:
                quid = f'{clip_uids[bidx]}_{annotation_uids[bidx]}_{query_sets[bidx]}'
                p_save_plot = p_preds_dir / f'{quid}-prob_preds_{bidx}-{step:07d}.png'
                plt.figure(figsize=(12, 6))
                plt.plot(gt_probs[bidx], label='gt')
                plt.plot(prob_pred_tops[bidx], label='pred_top')
                plt.legend()
                plt.title(f'Probabilities: {quid}')
                plt.savefig(p_save_plot)
                plt.close()
                print(f'Saved: {p_save_plot}')

                frames = []
                for tidx in range(T):
                    frame = Image.fromarray((255*segments[bidx, tidx].transpose(1, 2, 0)).astype(np.uint8))
                    draw = ImageDraw.Draw(frame)
                    if gt_probs[bidx, tidx] > 1e-3:
                        bbox_ = (gt_bboxes_xyxy[bidx, tidx] * [w, h, w, h]).astype(int).tolist()
                        draw.rectangle(bbox_, outline=(0, 255, 0), width=5)  # [4]
                    if prob_pred_tops[bidx, tidx] > .5:  # 0.5 following the threshold in loss.py
                        bbox_ = (prob_rts_xyxy[bidx, tidx] * [w, h, w, h]).astype(int).tolist()
                        draw.rectangle(bbox_, outline=(255, 50, 50), width=5)  # [4]
                    frames.append(frame)
                p_save_gif = p_save_plot.with_name(f'{quid}-preds_{bidx}-{step:07d}.gif')
                frames[0].save(p_save_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
                print(f'Saved: {p_save_gif}')
                del frames, frame, bbox_, draw

                p_save_query = p_save_plot.with_name(f'{quid}-query.png')
                if not p_save_query.exists():
                    qx, qy, qh, qw = vcs['x'][bidx], vcs['y'][bidx], vcs['h'][bidx], vcs['w'][bidx]
                    query_frame = Image.fromarray((255*queries[bidx].transpose(1, 2, 0)).astype(np.uint8))
                    draw = ImageDraw.Draw(query_frame)
                    bbox_ = int(qx * w), int(qy * h), int((qx + qw) * w), int((qy + qh) * h)
                    draw.rectangle(bbox_, outline=(255, 255, 0), width=5)  # [4]
                    query_frame.save(p_save_query)
                    print(f'Saved: {p_save_query}')
                    del query_frame, bbox_, draw
                break
        return


def set_prefix_to_keys(d: dict, prefix: str) -> dict:
    return {f'{prefix}/{k}': v for k, v in d.items()}


if __name__ == '__main__':
    print('Testing a single train step')
    config = OmegaConf.load('config/base.yaml')
    model = LitModule(config)
    segment = torch.rand(2, 3, 224, 224)
    query = torch.rand(2, 3, 224, 224)
    gt_bboxes = torch.rand(2, 10, 4)
    batch = {'segment': segment, 'query': query, 'gt_bboxes': gt_bboxes}
    loss = model.training_step(batch, 0)
    print(loss)
