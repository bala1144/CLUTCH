from pathlib import Path
import os
import time
import logging
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

def create_logger(cfg, phase='train', tag=None, sub_dir=None):
    # root dir set by cfg
    root_output_dir = Path(cfg.FOLDER)
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')

    cfg_name = cfg.NAME
    model = cfg.model.target.split('.')[-2]
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    if phase=='train':
        # set up logger
        if not root_output_dir.exists():
            print('=> creating {}'.format(root_output_dir))
            root_output_dir.mkdir()

        cfg.FOLDER_EXP = os.path.join(cfg.FOLDER, model, cfg_name)
        os.makedirs(cfg.FOLDER_EXP, exist_ok=True)
        final_output_dir = Path(cfg.FOLDER_EXP)
    elif phase=='test':
        ckpts=cfg.TEST.CHECKPOINTS
        out_root_dir = ckpts.split("/checkpoints/")[0]
        ckpt = ckpts.split("/checkpoints/")[-1].split(".")[0]
        out_dir_name = f"{time_str}_test_{ckpt}"
        cfg.TEST_FOLDER_EXP = os.path.join(out_root_dir, out_dir_name)
        os.makedirs(cfg.TEST_FOLDER_EXP, exist_ok=True)
        final_output_dir = Path(cfg.TEST_FOLDER_EXP)
    elif phase in ["test_w_visual_results", "compute_vqvae_metric", "compute_metric", "compute_metric_p", "compute_m2t_metric", "compute_m2m_metric"] or "vqvae_" in phase:
        ckpts=cfg.TEST.CHECKPOINTS
        out_root_dir = ckpts.split("/checkpoints/")[0]
        ckpt = ckpts.split("/checkpoints/")[-1].split(".")[0]
        out_dir_name = f"{time_str}_{phase}_{ckpt}"
        if tag is not None:
            out_dir_name = f"{time_str}_{phase}_{tag}_{ckpt}"

        cfg.TEST_FOLDER_EXP = os.path.join(out_root_dir, out_dir_name)
        if sub_dir is not None:
             cfg.TEST_FOLDER_EXP = os.path.join(out_root_dir, sub_dir, out_dir_name)

        os.makedirs(cfg.TEST_FOLDER_EXP, exist_ok=True)
        final_output_dir = Path(cfg.TEST_FOLDER_EXP)
    elif phase in ["final_results"]:
        ckpts=cfg.TEST.CHECKPOINTS
        out_root_dir = ckpts.split("/checkpoints/")[0]
        ckpt = ckpts.split("/checkpoints/")[-1].split(".")[0]
        out_dir_name = f"{time_str}_{phase}_{ckpt}"
        if tag is not None:
            out_dir_name = f"{time_str}_{phase}_{tag}_{ckpt}"

        out_root_dir = out_root_dir.replace("mgpt_GRAB_instruct_train", "mgpt_final_results")
        out_root_dir = out_root_dir.replace("mgpt_GRAB_mdm", "mgpt_final_results")

        cfg.TEST_FOLDER_EXP = os.path.join(out_root_dir, out_dir_name)
        if sub_dir is not None:
             cfg.TEST_FOLDER_EXP = os.path.join(out_root_dir, sub_dir, out_dir_name)

        os.makedirs(cfg.TEST_FOLDER_EXP, exist_ok=True)
        final_output_dir = Path(cfg.TEST_FOLDER_EXP)
    else:
        raise ("Enter valid phase")

        
    # if not os.path.exists(cfg.FOLDER_EXP): # if doesnt exist create it
    #     os.makedirs(cfg.FOLDER_EXP, exist_ok=True)
    #     final_output_dir = Path(cfg.FOLDER_EXP)
    # ## if the folder exists and resume is false, add time snap to it
    # elif not os.path.exists(cfg.TRAIN.RESUME):
    #     cfg.NAME = cfg_name + f"_{time_str}"
    #     final_output_dir = root_output_dir / model / cfg.NAME
    #     cfg.FOLDER_EXP = str(final_output_dir)
    #     print('=> creating {}'.format(final_output_dir))

    new_dir(cfg, phase, time_str, final_output_dir)

    head = '%(asctime)-15s %(message)s'
    logger = config_logger(final_output_dir, time_str, phase, head)
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        logging.basicConfig(format=head)
    return logger


@rank_zero_only
def config_logger(final_output_dir, time_str, phase, head):
    log_file = '{}_{}_{}.log'.format('log', time_str, phase)
    final_log_file = final_output_dir / log_file
    logging.basicConfig(filename=str(final_log_file))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    file_handler = logging.FileHandler(final_log_file, 'w')
    file_handler.setFormatter(logging.Formatter(head))
    file_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(file_handler)

    print(f"Logging file: {final_log_file}\n")

    return logger


@rank_zero_only
def new_dir(cfg, phase, time_str, final_output_dir):
    # new experiment folder
    cfg.TIME = str(time_str)
    if os.path.exists(final_output_dir) and not os.path.exists(cfg.TRAIN.RESUME) and not cfg.DEBUG and phase not in ['test', 'demo']:
        file_list = sorted(os.listdir(final_output_dir), reverse=True)
        for item in file_list:
            if item.endswith('.log'):
                os.rename(str(final_output_dir), str(final_output_dir) + '_' + cfg.TIME)
                break
    final_output_dir.mkdir(parents=True, exist_ok=True)
    # write config yaml
    config_file = '{}_{}_{}.yaml'.format('config', time_str, phase)
    final_config_file = final_output_dir / config_file
    OmegaConf.save(config=cfg, f=final_config_file)

    print(f"\nConfig file: {final_config_file}\n")
