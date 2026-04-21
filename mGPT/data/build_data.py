from omegaconf import OmegaConf
from os.path import join as pjoin
from mGPT.config import instantiate_from_config


def build_data(cfg, phase="train"):
    data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
    data_config['params'] = {'cfg': cfg, 'phase': phase}
    if isinstance(data_config['target'], str):
        return instantiate_from_config(data_config)
    raise TypeError(f"Unsupported DATASET.target: {type(data_config['target']).__name__}")
