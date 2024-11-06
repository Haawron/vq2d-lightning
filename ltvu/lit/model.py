import re
import hydra.utils
from omegaconf import OmegaConf

import torch
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
            torch.set_grad_enabled(False)  # disable grads
            model.eval()  # disable batchnorm + dropout
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                val_out = model.validation_step(val_batch, val_batch_idx)  # -> should be handled in `on_validation_epoch_end`
            torch.set_grad_enabled(True)  # enable grads
            model.train()  # enable batchnorm + dropout
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
        if isinstance(config, dict):  # eval.py, config from a checkpoint, for backward compatibility
            config = OmegaConf.create(config)
        self.config = config
        self.model: VQLoC = hydra.utils.instantiate(
            config.model, compile_backbone=config.get('compile', True))
        self.fix_backbone = config.model.fix_backbone

        self.save_hyperparameters(ignore='config')  # to avoid saving unresolved config as a hyperparameter
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))  # to save the config in the checkpoint
        self.sample_step = 0
        self.use_hnm = config.get('use_hnm', False)

        self.rt_pos_query = config.get('rt_pos_query')

    ############ major hooks ############

    def training_step(self, batch, batch_idx):
        bsz = batch['segment'].shape[0]
        extra_args = {}
        if self.use_hnm is not None:
            extra_args['use_hnm']=True
        if self.rt_pos_query is not None:
            self.late_epoch_rt_pos = self.rt_pos_query.late_epoch_rt_pos
            self.mode = self.rt_pos_query.mode
            self.sim_thr = self.rt_pos_query.sim_thr
            extra_args['sim_thr']=self.sim_thr
            extra_args['enable_rt_pq_threshold']=self.rt_pos_query.enable_rt_pq_threshold

            if self.current_epoch >= self.late_epoch_rt_pos:
                extra_args['rt_pos']=True
            if self.mode == 'easy':
                extra_args['sim_mode']='max'
            elif self.mode == 'hard':
                extra_args['sim_mode']='min'
            elif self.mode == 'both':
                if self.current_epoch >= self.late_epoch_rt_pos:
                    extra_args['sim_mode']='min'
                else:
                    extra_args['sim_mode']='max'
        output_dict = self.model.forward(
            **batch,
            compute_loss=True,
            cur_epoch=self.trainer.current_epoch,
            max_epochs=self.trainer.max_epochs,
            **extra_args
        )

        assert output_dict['loss'].requires_grad
        assert torch.isfinite(output_dict['loss']), f'Loss is {output_dict["loss"]}'
        log_dict = set_prefix_to_keys(output_dict['log_dict'], 'Train')
        self.sample_step = (1 + self.global_step) * bsz * self.trainer.world_size  # +1 for maintaining monotonicity
        log_dict['sample_step'] = self.sample_step
        self.log_dict(log_dict, batch_size=bsz, rank_zero_only=True)
        return output_dict['loss']

    def validation_step(self, batch, batch_idx):
        bsz = batch['segment'].shape[0]
        output_dict = self.model.forward(**batch, compute_loss=True, training=False)
        if 'log_dict' in output_dict:
            log_dict = set_prefix_to_keys(output_dict['log_dict'], 'Val')
            self.log_dict(log_dict, batch_size=bsz, on_epoch=True, sync_dist=True)
            # if self.sample_step > 0:  # after sanity check done
            #     if batch_idx % 50 == 0:
            #       try:
            #           self.print_outputs(batch, output_dict, bidxs=[0])
            #       except Exception as e:
            #           print(f"Error in {batch['clip_uid']} print_outputs: {e}")
            self.trainer.strategy.barrier('validation_step_end')  # processing times may vary

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        bsz = batch['segment'].shape[0]
        device = batch['segment'].device
        output_dict = self.model.forward(**batch, compute_loss=True, training=False)
        # bbox: [b,t,4], in pixels wrt the original, yxyx, float
        # prob: [b,t], logits, float
        preds_top = output_dict['info_dict']['preds_top']
        pred_outputs = []
        for bidx in range(bsz):
            ow, oh = batch['original_width'][bidx], batch['original_height'][bidx]
            bbox_yxyx = preds_top['bbox'][bidx]
            bbox_xyxy: torch.Tensor = bbox_yxyx[..., [1, 0, 3, 2]]  # yxyx -> xyxy
            pad_size_float = (1. - oh / ow) / 2
            bbox_xyxy -= torch.tensor([0, pad_size_float, 0, pad_size_float], device=device)
            bbox_xyxy *= ow  # unnormalize
            bbox_xyxy = bbox_xyxy.clamp(torch.tensor(0, device=device), torch.tensor([ow, oh, ow, oh], device=device))
            pred_outputs.append({
                # crucial information for segment indexing
                'qset_uuid': batch['qset_uuid'][bidx],
                'seg_idx': batch['seg_idx'][bidx].item(),  # 0-based
                'num_segments': batch['num_segments'][bidx].item(),

                # predictions
                'ret_bboxes': bbox_xyxy.cpu(),
                'ret_scores': preds_top['prob'][bidx].cpu(),

                # for debugging, visualization or analysis
                'clip_uid': batch['clip_uid'][bidx],
                'frame_idxs': batch['frame_idxs'][bidx].cpu(),  # check missing or duplicated frames (last frame can be duplicated)
            })
        return pred_outputs

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

    def on_load_checkpoint(self, checkpoint):
        param_names = set('model.' + k for k in self.model.state_dict().keys())
        new_state_dict = {}

        replacements = (
            (
                ('backbone._orig_mod', 'backbone'),
                ('backbone', 'backbone._orig_mod'),
            ),
            (
                ('reduce._orig_mod', 'reduce'),
                ('reduce', 'reduce._orig_mod'),
            ),
            (
                (r'CQ_corr_transformer\.(\d+)\._orig_mod', r'CQ_corr_transformer.\1'),
                (r'CQ_corr_transformer\.(\d+)', r'CQ_corr_transformer.\1._orig_mod'),
            ),
            (
                (r'CQ_corr_transformer\.(\d+)', r'CQ_corr_transformer.\1.net'),
                (r'CQ_corr_transformer\.(\d+)\.net', r'CQ_corr_transformer.\1'),
                (r'(.*CQ_corr_transformer.*)\.self_attn\.(.*)', r'\1.self_attn.net.\2'),
                (r'(.*CQ_corr_transformer.*)\.self_attn\.net\.(.*)', r'\1.self_attn.\2'),
            ),
        )

        def dfs(k, idx=0):
            if idx == len(replacements):
                return None

            if (k0 := dfs(k, idx+1)) is not None:
                return k0

            for base, repl in replacements[idx]:
                k1 = re.sub(base, repl, k)
                if k1 == k: # no change
                    continue

                if k1 in param_names:
                    return k1

                if (kx := dfs(k1, idx+1)) is not None:
                    return kx

        for k, v in checkpoint['state_dict'].items():
            # ignore conditions
            if 'query_down_heads' in k:
                continue
            if 'pe_stx' in k:
                if not isinstance(v, torch.Tensor) or v.numel() < 256:  # Null tensor
                    continue

            # replace conditions
            if k in param_names:
                new_state_dict[k] = v
            else:
                if (kk := dfs(k)) is not None:
                    new_state_dict[kk] = v
                else:
                    raise ValueError(f'Key {k} not found in the model\n\n{param_names}\n')

        # reset the training state
        checkpoint['state_dict'] = new_state_dict
        checkpoint['epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step
        del checkpoint['loops']  # remove previous fit loop state
        del checkpoint['callbacks']
        checkpoint['optimizer_states'] = {}
        if 'lr_schedulers' in checkpoint:
            checkpoint['lr_schedulers'] = {}

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
                cquid = f'{clip_uids[bidx]}_{annotation_uids[bidx]}_{query_sets[bidx]}'
                p_preds_dir_cquid = p_preds_dir / cquid
                p_preds_dir_cquid.mkdir(parents=True, exist_ok=True)
                p_save_plot = p_preds_dir_cquid / f'prob_preds_{bidx}-{step:07d}.png'
                plt.figure(figsize=(12, 6))
                plt.plot(gt_probs[bidx], label='gt')
                plt.plot(prob_pred_tops[bidx], label='pred_top')
                plt.ylim(0, 1.1)
                plt.legend()
                plt.title(f'Probabilities {cquid}')
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
                p_save_gif = p_preds_dir_cquid / f'preds_{bidx}-{step:07d}.gif'
                frames[0].save(p_save_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
                print(f'Saved: {p_save_gif}')
                del frames, frame, bbox_, draw

                p_save_query = p_preds_dir_cquid / f'query.png'
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
