import os
import sys
import shutil
import numpy as np  # must come before chumpy/smplx to apply compat patch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# NumPy >=1.24 removed deprecated type aliases used by chumpy/smplx
if not hasattr(np, 'bool'):   np.bool   = bool
if not hasattr(np, 'int'):    np.int    = int
if not hasattr(np, 'float'):  np.float  = float
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'str'):    np.str    = str

import pytorch_lightning as pl

from mGPT.callback import build_callbacks
from mGPT.config import parse_args, instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Metric Logger
    pl_loggers = []
    for logger_name in cfg.LOGGER.TYPE:
        if logger_name == "wandb" and not cfg.LOGGER.WANDB.params.project:
            continue
        pl_logger = instantiate_from_config(
            eval(f'cfg.LOGGER.{logger_name.upper()}'))
        pl_loggers.append(pl_logger)

    # Save a copy of the config to the experiment folder
    logger.info(f"Copying {cfg.cfg_file} to EXP folder")
    cfg_file = cfg.cfg_file
    new_file_to_folder_loc = os.path.join(cfg.FOLDER_EXP, "input_train_config.yaml")
    shutil.copy2(cfg_file, new_file_to_folder_loc)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        log_every_n_steps=cfg.LOGGER.get('LOG_EVERY_N_STEPS', 50),
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy=cfg.TRAIN.get('STRATEGY', "ddp_find_unused_parameters_true")
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
    )
    logger.info("Trainer initialized")

    # Strict load pretrianed model
    if cfg.TRAIN.PRETRAINED:
        load_pretrained(cfg, model, logger)

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    try:
        if cfg.TRAIN.RESUME:
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.TRAIN.PRETRAINED)
        else:
            trainer.fit(model, datamodule=datamodule)

    except RuntimeError as e:
        import traceback
        logger.error("**********************************")
        logger.error("**********************************")
        logger.error("**********************************")
        logger.error("❌ Training failed due to a runtime error:")
        logger.error(f"{e}")
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(tb_str) 
        logger.error("**********************************")
        logger.error("**********************************")
        logger.error("**********************************")
        sys.exit(1)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")
 

if __name__ == "__main__":
    main()
