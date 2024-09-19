from dotenv import load_dotenv
load_dotenv()

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


@hydra.main(config_path='config/eval', config_name='base', version_base='1.3')
def main(config: DictConfig):
    torch.set_float32_matmul_precision('highest')
    p_ckpt = Path(config.ckpt)
    plm = LitModule.load_from_checkpoint(p_ckpt)
    print(plm.config)
    pdm = LitVQ2DDataModule(config)
    # trainer = get_trainer(config, jid, enable_progress_bar=not within_slurm_batch())
    print(plm)
    print(pdm)
    L.Trainer.save_checkpoint


if __name__ == '__main__':
    main()
