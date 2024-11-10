from dotenv import load_dotenv
load_dotenv()

import os
import subprocess
from pathlib import Path
import hydra
import hydra.core.hydra_config
from omegaconf import OmegaConf, DictConfig

import torch

import lightning as L
import lightning.pytorch.utilities as L_utils

from ltvu.lit import *


@L_utils.rank_zero_only
def log_to_console(msg):
    print(msg)


def within_slurm_batch():
    command = (
        "scontrol show jobid " + os.environ.get("SLURM_JOB_ID", "") +
        " | grep -oP '(?<=BatchFlag=)([0-1])'"
    )
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    batch_flag = int(result.stdout.strip())
    return batch_flag == 1


@hydra.main(config_path='config', config_name='eval', version_base='1.3')
def main(config: DictConfig):
    L.seed_everything(config.random_seed, workers=True)
    torch.set_float32_matmul_precision('highest')
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jid = os.environ.get("SLURM_JOB_ID")

    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(config, resolve=True))
    log_to_console("="*80 + '\n')

    trainer, _ = get_trainer(config, jid=jid, enable_progress_bar=not within_slurm_batch(), enable_checkpointing=False)

    assert config.ckpt is not None, "Please provide a checkpoint path"
    p_ckpt = Path(config.ckpt)
    plm = LitModule.load_from_checkpoint(p_ckpt)
    match config.dataset.name:
        case 'vq2d':
            litdatamodule = LitVQ2DDataModule
        case 'egotracks':
            litdatamodule = LitEgoTracksDataModule
    pdm = litdatamodule(config)  # eval config

    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(plm.config.model))
    log_to_console("="*80 + '\n')

    trainer.predict(plm, datamodule=pdm, return_predictions=False)


if __name__ == '__main__':
    if int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) == 1 or int(os.environ.get('SLURM_NNODES', 1)) == 1:
        os.environ["SLURM_JOB_NAME"] = "bash"  # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    OmegaConf.register_new_resolver("job_type", lambda : 'batch' if within_slurm_batch() else 'debug')
    OmegaConf.register_new_resolver('runtime_outdir', lambda : hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    main()
