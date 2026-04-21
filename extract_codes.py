import os
import sys
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained_vae


def resolve_device(cfg):
    if torch.cuda.is_available() and cfg.ACCELERATOR == "gpu":
        return torch.device(f"cuda:{cfg.DEVICE[0]}")
    cfg.ACCELERATOR = "cpu"
    cfg.DEVICE = 1
    return torch.device("cpu")

def main():
    
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.EVAL.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 1
    cfg.TRAIN.NUM_WORKERS = 0
    cfg.EVAL.NUM_WORKERS = 0
    cfg.TEST.NUM_WORKERS = 0
    device = resolve_device(cfg)

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")

    # load the output dir
    output_dir = os.path.join(datasets.hparams.dataset_dir, cfg.DATASET.CODE_PATH)
    os.makedirs(output_dir, exist_ok=True)
    print("Output dir for the tokens:", output_dir)

    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae
    print("model loaded")

    # Strict load vae model
    assert cfg.TRAIN.PRETRAINED_VAE is not None
    load_pretrained_vae(cfg, model)

    model = model.to(device)

    compression_factor = 2**cfg.model.params.motion_vae.params.down_t
    for dataloader in [datasets.train_dataloader(), datasets.val_dataloader(), datasets.test_dataloader()]:
        for batch in tqdm(dataloader,
                        desc=f'motion tokenize'): 
            name = batch['seq_name']
            pose = batch['motion']
            pose = pose.to(device).float()

            if batch.get("motion_mask") is not None:
                mm = batch.get("motion_mask")[0]
                seq_len = torch.where(mm == 1)[0].max().item() + 1
                seq_len = (seq_len // compression_factor) * compression_factor
                pose = pose[:, :seq_len]

            if pose.shape[1] == 0:
                continue
            target, _ = model.vae.encode(pose)
            target = target.to('cpu').numpy()

            target_path = os.path.join(output_dir, name[0] + '.npy')
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(target_path, target)

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
