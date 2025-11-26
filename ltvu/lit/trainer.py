import json
from pathlib import Path
import datetime
import hydra.utils
from omegaconf import OmegaConf, DictConfig, open_dict

import torch
import os

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelSummary, ModelCheckpoint, TQDMProgressBar, Callback
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from ltvu.lit.callback import PerSegmentWriter


type_loggers = WandbLogger | CSVLogger


def get_trainer(config, jid, enable_progress_bar=False, enable_checkpointing=True, ddp_timeout=300, movement=""):
    runtime_outdir = Path(config.runtime_outdir)
    trainer_config: DictConfig = config.trainer
    task = config.dataset.name

    # callbacks
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(),
        TQDMProgressBar(refresh_rate=1 if enable_progress_bar else 20, leave=True),
    ]

    if task == 'vq2d':
        callbacks.append(PerSegmentWriter(
            output_dir=runtime_outdir / 'vq2d',
            official_anns_dir=config.dataset.official_anns_dir,
            test_submit=config.dataset.get('test_submit', False),
            movement=movement))

    if enable_checkpointing:
        ckpt_callback_iou = ModelCheckpoint(
            dirpath=runtime_outdir,
            save_last=False,
            monitor='Val/iou',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='epoch={epoch}-iou={Val/iou:.4f}')
        ckpt_callback_prob = ModelCheckpoint(
            dirpath=runtime_outdir,
            save_last=False,
            monitor='Val/prob_acc',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='epoch={epoch}-prob_acc={Val/prob_acc:.4f}')
        ckpt_callback_last = ModelCheckpoint(
            dirpath=runtime_outdir,
            filename='last-{epoch}')
        callbacks.append(ckpt_callback_iou)
        callbacks.append(ckpt_callback_prob)
        callbacks.append(ckpt_callback_last)
        callbacks.append(CheckpointLogger())
        callbacks.append(ChangeFilePermissionsCallback(runtime_outdir))
    else:
        ckpt_callback_prob = None

    assert jid is not None, 'jid must be provided when loggers are enabled'
    with open_dict(trainer_config):  # obtaining write access
        loggers_config = trainer_config.pop('logger', [])  # to not pass it to the Trainer

    loggers = [CSVLogger(save_dir=runtime_outdir, name="lit", version=jid)]
    for logger_config in loggers_config:
        logger: type_loggers = hydra.utils.instantiate(logger_config)
        loggers.append(logger)

    # Note: do not let hydra instantiate the Trainer or it is highly inflexible
    trainer_config = OmegaConf.to_container(trainer_config, resolve=True)
    if 'strategy' not in trainer_config:
        trainer_config['strategy'] = DDPStrategy(
            timeout=datetime.timedelta(seconds=ddp_timeout),
            find_unused_parameters=True)
    trainer = L.Trainer(
        **trainer_config,
        enable_model_summary=False,
        default_root_dir=runtime_outdir,
        logger=loggers,
        callbacks=callbacks,
    )
    return trainer, ckpt_callback_prob

class ChangeFilePermissionsCallback(Callback):
    def __init__(self, dirpath):
        self.dirpath = dirpath

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        for filename in os.listdir(self.dirpath):
            if filename.endswith('.ckpt'):
                filepath = os.path.join(self.dirpath, filename)
                os.chmod(filepath, 0o644)  # Setting file permissions to rw-r--r--

class CheckpointLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if callback.monitor == "Val/prob_acc":
                    best_ckpt_path_prob = callback.best_model_path
                    if trainer.is_global_zero and best_ckpt_path_prob:
                        print(f"Best Prob epoch : {best_ckpt_path_prob}")
                elif callback.monitor == "Val/iou":
                    best_ckpt_path_iou = callback.best_model_path
                    if trainer.is_global_zero and best_ckpt_path_iou:
                        print(f"Best IOU ckpt   : {best_ckpt_path_iou}")