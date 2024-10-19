import os
from pprint import pprint

import hydra
from omegaconf import OmegaConf, DictConfig


def within_slurm_batch():
    batch_flag = int(os.system(r"batch_flag=$(scontrol show jobid ${SLURM_JOB_ID} | grep -oP '(?<=BatchFlag=)([0-1])')"))
    return batch_flag == 1


@hydra.main(config_path='../../config', config_name='base', version_base='1.3')
def main(config: DictConfig):
    pprint(OmegaConf.to_container(config, resolve=True))
    default_root_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(default_root_dir)

if __name__ == '__main__':
    OmegaConf.register_new_resolver("batch_dir", lambda : 'batch' if within_slurm_batch() else 'debug')
    main()
