import datetime

import hydra.utils
from omegaconf import OmegaConf, DictConfig, open_dict

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelSummary, ModelCheckpoint
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy


def get_trainer(config, jid, enable_progress_bar=False):
    runtime_outdir: str = config.runtime_outdir
    trainer_config: DictConfig = config.trainer

    # callbacks
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=runtime_outdir,
            save_last=False,
            monitor='iou',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='step={step}-iou={iou:.4f}'),
    ]

    loggers = []
    loggers.append(CSVLogger(save_dir=runtime_outdir, name="lit", version=jid))
    with open_dict(trainer_config):  # obtaining write access
        for logger in trainer_config.pop('logger', []):
            loggers.append(hydra.utils.instantiate(logger))

    return L.Trainer(
        **OmegaConf.to_container(trainer_config, resolve=True),
        strategy=DDPStrategy(
            timeout=datetime.timedelta(seconds=600),
            find_unused_parameters=True),
        enable_model_summary=False,
        default_root_dir=runtime_outdir,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
    )
