from dotenv import load_dotenv
load_dotenv()

# ⭐ import order matters ⭐
# built-in + hydra
import os
import subprocess
import hydra
from omegaconf import OmegaConf, DictConfig

# torch
import torch

# lightning
import lightning as L
import lightning.pytorch.utilities as L_utils

# others (numpy, ...)

# local (ours)
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


@hydra.main(config_path='config', config_name='base', version_base='1.3')
def main(config: DictConfig):
    L.seed_everything(config.random_seed, workers=True)
    torch.set_float32_matmul_precision('medium')
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jid = os.environ.get("SLURM_JOB_ID")

    log_to_console('\n' + "="*80 + '\n')
    log_to_console(OmegaConf.to_yaml(config, resolve=True))
    log_to_console("="*80 + '\n')

    plm = LitModule(config)
    plm.model.backbone = torch.compile(plm.model.backbone)
    pdm = LitVQ2DDataModule(config)
    trainer = get_trainer(config, jid, enable_progress_bar=not within_slurm_batch())

    if within_slurm_batch() and trainer.global_rank == 0:
        cmd = f"scontrol write batch_script {jid} {default_root_dir}/slurm-{jid}.sh"
        os.system(cmd)

    trainer.fit(plm, datamodule=pdm)


if __name__ == '__main__':
    os.environ["SLURM_JOB_NAME"] = "bash"  # https://github.com/Lightning-AI/pytorch-lightning/issues/16236#issuecomment-1690552495
    OmegaConf.register_new_resolver("job_type", lambda : 'batch' if within_slurm_batch() else 'debug')
    OmegaConf.register_new_resolver('runtime_outdir', lambda : hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    OmegaConf.register_new_resolver("eval", eval)
    main()
