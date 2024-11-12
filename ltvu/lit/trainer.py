import json
from pathlib import Path
import datetime
import hydra.utils
from omegaconf import OmegaConf, DictConfig, open_dict

import torch

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelSummary, ModelCheckpoint, TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from ltvu.lit.callback import PerSegmentWriter, PerSegmentWriterEgoTracks, PerSegmentWriterLaSOT


type_loggers = WandbLogger | CSVLogger


def get_trainer(config, jid, enable_progress_bar=False, enable_checkpointing=True, ddp_timeout=30):
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
            test_submit=config.dataset.get('test_submit', False)))
    elif task == 'egotracks':
        callbacks.append(PerSegmentWriterEgoTracks(
            output_dir=runtime_outdir / 'egotracks',
            official_anns_dir=config.dataset.official_anns_dir,
            test_submit=config.dataset.get('test_submit', False)))
    elif task == 'lasot':
        callbacks.append(PerSegmentWriterLaSOT(
            output_dir=runtime_outdir / 'lasot',
            official_anns_dir=config.dataset.official_anns_dir))

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
        callbacks.append(ckpt_callback_iou)
        callbacks.append(ckpt_callback_prob)
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
