import hydra.utils
from omegaconf import OmegaConf

import torch

import lightning as L

from transformers import get_linear_schedule_with_warmup

from ltvu.models import *


# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
class LitModule(L.LightningModule):
    """
    
    # Pseudo code for training loop
    https://lightning.ai/docs/pytorch/stable/common/trainer.html#under-the-hood

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
        self.save_hyperparameters()
        self.model: VQLOC = hydra.utils.instantiate(config.model)


    ############ major hooks ############

    def training_step(self, batch, batch_idx):
        bsz = batch['segment'].shape[0]
        output_dict = self.model.forward(**batch, compute_loss=True)
        assert output_dict['loss'].requires_grad
        self.log_dict(set_prefix_to_keys(
            output_dict['log_dict'], 'Train'), batch_size=bsz, rank_zero_only=True)
        return output_dict['loss']

    def validation_step(self, batch, batch_idx):
        bsz = batch['segment'].shape[0]
        output_dict = self.model.forward(**batch, compute_loss=True)
        self.log_dict(set_prefix_to_keys(
            output_dict['log_dict'], 'Val'), batch_size=bsz, sync_dist=True)
        return output_dict

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        optim_config = self.config.optim
        alL_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        optimizer = hydra.utils.instantiate(optim_config.optimizer, params=alL_params)
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
        if self.trainer.global_rank == 0:
            print('Starting training...')


    ############ helper functions ############

    def get_wnb_logger(self):
        return [l for l in self.logger if isinstance(l, L.WandbLogger)][0]


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
