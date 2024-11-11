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
    match config.dataset.name:
        case 'vq2d':
            litdatamodule = LitVQ2DDataModule
        case 'egotracks':
            litdatamodule = LitEgoTracksDataModule
        case 'lasot':
            litdatamodule = LitLaSOTDataModule
    pdm = litdatamodule(config)
    trainer, ckpt_callback = get_trainer(config, jid, enable_progress_bar=not within_slurm_batch())

    log_to_console(str(plm.model))

    # if wandblogger is present, log the hostname to the wandb dashboard
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            log_dict(logger, {"hostname": os.uname().nodename})

    if within_slurm_batch():
        write_batch_script(jid, default_root_dir)

    # don't convert 'ckpt_finetune_from' to 'ckpt' as it may conflict with the eval config
    if ckpt_finetune_from := config.get('ckpt_finetune_from'):
        trainer.fit(plm, datamodule=pdm, ckpt_path=ckpt_finetune_from)
    else:
        trainer.fit(plm, datamodule=pdm)

    if config.predict_val or config.predict_test:
        p_ckpt = 'outputs/batch/2024-10-13/130884/epoch=105-iou=0.4454.ckpt'
        p_ckpt = p_ckpt if config.get('debug') else ckpt_callback.best_model_path
        eval_config = hydra.compose(config_name=config.get('eval_config', 'eval'), overrides=[
            f'dataset={config.dataset.name}',
            f'dataset.clips_dir={config.dataset.clips_dir}',
            f'ckpt={str(p_ckpt).replace('=', '\\=')}',
            f'batch_size={config.batch_size}',
            f'num_workers={config.num_workers}',
            f'prefetch_factor={config.prefetch_factor}'
        ])
        plm = LitModule.load_from_checkpoint(p_ckpt)
        pdm = litdatamodule(eval_config)

        if config.predict_val:
            trainer, _ = get_trainer(eval_config, jid=jid, enable_progress_bar=not within_slurm_batch(), enable_checkpointing=False, ddp_timeout=600)
            log_to_console('\n' + "="*80 + '\n')
            log_to_console('Evaluating the best model')
            trainer.predict(plm, datamodule=pdm, return_predictions=False)
            log_to_console('\n' + "="*80 + '\n')

        if config.predict_test and not config.dataset.name != 'lasot':
            eval_config.test_submit = True
            pdm.test_submit = True
            trainer, _ = get_trainer(eval_config, jid=jid, enable_progress_bar=not within_slurm_batch(), enable_checkpointing=False, ddp_timeout=600)
            log_to_console('\n' + "="*80 + '\n')
            log_to_console('Evaluating the best model on test set')
            trainer.predict(plm, datamodule=pdm, return_predictions=False)
            log_to_console('\n' + "="*80 + '\n')


if __name__ == '__main__':
    if int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) == 1 or int(os.environ.get('SLURM_NNODES', 1)) == 1:
        os.environ["SLURM_JOB_NAME"] = "bash"  # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    OmegaConf.register_new_resolver("job_type", lambda : 'batch' if within_slurm_batch() else 'debug')
    OmegaConf.register_new_resolver('runtime_outdir', lambda : hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    main()
