import datetime

import hydra.utils
from omegaconf import OmegaConf, DictConfig, open_dict

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelSummary, ModelCheckpoint, TQDMProgressBar
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy


type_loggers = WandbLogger | CSVLogger


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
            monitor='Val/iou',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='epoch={epoch}-iou={Val/iou:.4f}'),
        ModelCheckpoint(
            dirpath=runtime_outdir,
            save_last=False,
            monitor='Val/prob_acc',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='epoch={epoch}-prob_acc={Val/prob_acc:.4f}'),
        TQDMProgressBar(refresh_rate=1 if enable_progress_bar else 20, leave=True),
    ]

    loggers = []
    loggers.append(CSVLogger(save_dir=runtime_outdir, name="lit", version=jid))
    with open_dict(trainer_config):  # obtaining write access
        loggers_config = trainer_config.pop('logger', [])  # to not pass it to the Trainer
    for logger_config in loggers_config:
        logger: type_loggers = hydra.utils.instantiate(logger_config)
        loggers.append(logger)

    # Note: do not let hydra instantiate the Trainer or it is highly inflexible
    return L.Trainer(
        **OmegaConf.to_container(trainer_config, resolve=True),
        strategy=DDPStrategy(
            timeout=datetime.timedelta(seconds=600),
            find_unused_parameters=True),
        enable_model_summary=False,
        default_root_dir=runtime_outdir,
        logger=loggers,
        callbacks=callbacks,
        # enable_progress_bar=enable_progress_bar,
    )
