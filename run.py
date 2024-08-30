from dotenv import load_dotenv
load_dotenv()

import os
import time
import datetime

import torch
import hydra
from omegaconf import OmegaConf, DictConfig

import lightning as L
import lightning.pytorch.utilities as L_utils

from project.trainer import get_trainer
from model.lightning_module import LitModule
from model.dataset import LitDataModule


@L_utils.rank_zero_only
def log_to_console(msg):
    print(msg)


def within_slurm_batch():
    batch_flag = int(os.system(r"batch_flag=$(scontrol show jobid ${SLURM_JOB_ID} | grep -oP '(?<=BatchFlag=)([0-1])')"))
    return batch_flag == 1


@hydra.main(config_path='config', config_name='base', version_base='1.3')
def main(config: DictConfig):
    L.seed_everything(config.get('seed', 42), workers=True)
    torch.set_float32_matmul_precision('medium')
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jid = os.environ.get("SLURM_JOB_ID")

    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(config, resolve=True))
    log_to_console("="*80 + '\n')

    plm = LitModule(config)
    pdm = LitDataModule(config)
    trainer = get_trainer(config, jid, enable_progress_bar=not within_slurm_batch())

    if within_slurm_batch() and trainer.global_rank == 0:
        cmd = f"scontrol write batch_script {jid} {default_root_dir}/slurm-{jid}.sh"
        os.system(cmd)

    trainer.fit(plm, datamodule=pdm)


if __name__ == '__main__':
    # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    os.environ["SLURM_JOB_NAME"] = "bash"
    OmegaConf.register_new_resolver("batch_dir", lambda : 'batch' if within_slurm_batch() else 'debug')
    main()
