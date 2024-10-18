from dotenv import load_dotenv
load_dotenv()

# ⭐ import order matters ⭐
# built-in + hydra
import os
import subprocess
from pathlib import Path
import hydra
import hydra.core.hydra_config
from omegaconf import OmegaConf, DictConfig

# torch
import torch

# lightning
import lightning as L
import lightning.pytorch.utilities as L_utils
from lightning.pytorch.loggers import WandbLogger

# others (numpy, ...)

# local (ours)
from ltvu.lit import *


@L_utils.rank_zero_only
def log_to_console(msg):
    print(msg)


@L_utils.rank_zero_only
def write_batch_script(jid, default_root_dir):
    p_script = Path(default_root_dir) / f"slurm-{jid}.sh"
    command = f"scontrol write batch_script {jid} {p_script}"
    print(f'Writing batch script to {p_script}')
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)


@L_utils.rank_zero_only
def log_dict(logger, param_dict):
    logger.log_hyperparams(param_dict)


def within_slurm_batch():
    command = (
        "scontrol show jobid " + os.environ.get("SLURM_JOB_ID", "") +
        " | grep -oP '(?<=BatchFlag=)([0-1])'"
    )
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    batch_flag = int(result.stdout.strip())
    return batch_flag == 1


@hydra.main(config_path='config', config_name='train', version_base='1.3')
def main(config: DictConfig):
    L.seed_everything(config.random_seed, workers=True)
    torch.set_float32_matmul_precision('highest')
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jid = os.environ.get("SLURM_JOB_ID")

    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(config, resolve=True))
    log_to_console("="*80 + '\n')

    plm = LitModule(config)
    pdm = LitVQ2DDataModule(config)
    trainer, ckpt_callback = get_trainer(config, jid, enable_progress_bar=not within_slurm_batch())

    # if wandblogger is present, log the hostname to the wandb dashboard
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            log_dict(logger, {"hostname": os.uname().nodename})

    if within_slurm_batch():
        write_batch_script(jid, default_root_dir)

    resume_config = config.get('resume')
    if resume_config is not None:
        trainer.fit(plm, datamodule=pdm, ckpt_path=resume_config)
    else:
        trainer.fit(plm, datamodule=pdm)

    if config.predict_val or config.predict_test:
        p_ckpt = 'outputs/batch/2024-10-13/130884/epoch=105-iou=0.4454.ckpt'
        p_ckpt = p_ckpt if config.get('debug') else ckpt_callback.best_model_path
        eval_config = hydra.compose(config_name='eval', overrides=[
            f'ckpt={str(p_ckpt).replace('=', '\\=')}',
            f'batch_size={config.batch_size}',
            f'num_workers={config.num_workers}',
            f'prefetch_factor={config.prefetch_factor}'
        ])
        plm = LitModule.load_from_checkpoint(p_ckpt)
        pdm = LitVQ2DDataModule(eval_config)

        if config.predict_val:
            log_to_console('\n' + "="*80 + '\n')
            log_to_console('Evaluating the best model')
            trainer.predict(plm, datamodule=pdm, return_predictions=False)
            log_to_console('\n' + "="*80 + '\n')

        if config.predict_test:
            log_to_console('\n' + "="*80 + '\n')
            log_to_console('Evaluating the best model on test set')
            pdm.test_submit = True
            trainer.predict(plm, datamodule=pdm, return_predictions=False)
            log_to_console('\n' + "="*80 + '\n')


if __name__ == '__main__':
    os.environ["SLURM_JOB_NAME"] = "bash"  # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    OmegaConf.register_new_resolver("job_type", lambda : 'batch' if within_slurm_batch() else 'debug')
    OmegaConf.register_new_resolver('runtime_outdir', lambda : hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    main()
