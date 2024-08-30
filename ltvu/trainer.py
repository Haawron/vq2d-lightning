import datetime

import hydra
from omegaconf import OmegaConf

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelSummary,
    ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy


def get_trainer(config, jid, enable_progress_bar=False):
    trainer_config = config.trainer
    path_outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    model_checkpoints = [
        ModelCheckpoint(
            save_last=False,
            monitor='stAP25',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='step={step}-stAP25={stAP25:.4f}'),
    ]

    callbacks = [
        ModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval='epoch'),
        *model_checkpoints,
    ]

    logger_names = trainer_config.get('loggers', [])
    loggers = []
    loggers.append(CSVLogger(save_dir=path_outdir, version=jid, name="lit"))
    if 'tensorboard' in logger_names:
        loggers.append(TensorBoardLogger(
            save_dir=path_outdir, version=jid, name="lit", default_hp_metric=False))
    if 'wandb' in logger_names:
        loggers.append(WandbLogger(
            project='ltvu++', save_dir=path_outdir, version=jid, name="lit"))

    return L.Trainer(
        **OmegaConf.to_container(config.trainer, resolve=True),
        strategy=DDPStrategy(timeout=datetime.timedelta(seconds=600)),
        enable_model_summary=False,
        default_root_dir=path_outdir,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
    )
