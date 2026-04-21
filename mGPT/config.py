import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
from os.path import join as pjoin
import os
import glob


def get_module_config(cfg, filepath="./configs"):
    """
    Load yaml config files from subfolders
    """

    yamls = glob.glob(pjoin(filepath, '*', '*.yaml'))
    yamls = [y.replace(filepath, '') for y in yamls]
    for yaml in yamls:
        nodes = yaml.replace('.yaml', '').replace(os.sep, '.')
        nodes = nodes[1:] if nodes[0] == '.' else nodes
        OmegaConf.update(cfg, nodes, OmegaConf.load('./configs' + yaml))

    return cfg


def get_obj_from_str(string, reload=False):
    """
    Get object from string
    """

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate object from config
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def resume_config(cfg: OmegaConf):
    """
    Resume model and wandb
    """
    
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        if ".ckpt" in resume and os.path.exists(resume):
            print("Resumming from", resume)
            cfg.TRAIN.PRETRAINED = resume
        elif os.path.exists(resume):
            # Checkpoints
            cfg.TRAIN.PRETRAINED = pjoin(resume, "checkpoints", "last.ckpt")
            # Wandb
            wandb_files = os.listdir(pjoin(resume, "wandb", "latest-run"))
            wandb_run = [item for item in wandb_files if "run-" in item][0]
            cfg.LOGGER.WANDB.params.id = wandb_run.replace("run-","").replace(".wandb", "")
        else:
            raise ValueError("Resume path is not right.")

    return cfg

def parse_args(phase="train"):
    """
    Parse arguments and load config files
    """

    parser = ArgumentParser()
    group = parser.add_argument_group("Training options")

    # Assets
    group.add_argument(
        "--cfg_assets",
        type=str,
        required=False,
        default="./configs/assets.yaml",
        help="config file for asset paths",
    )

    # Default config
    if phase in ["train", "test", "demo"]:
        cfg_default = "./configs/default.yaml"
    else:
        raise ValueError(f"Unsupported phase: {phase}")
        
    group.add_argument(
        "--cfg",
        type=str,
        required=False,
        default=cfg_default,
        help="config file",
    )

    # Parse for each phase
    if phase in ["train", "test"]:
        group.add_argument("--batch_size",
                           type=int,
                           required=False,
                           help="training batch size")
        group.add_argument("--num_nodes",
                           type=int,
                           required=False,
                           help="number of nodes")
        group.add_argument("--device",
                           type=int,
                           nargs="+",
                           required=False,
                           help="training device")
        group.add_argument("--task",
                           type=str,
                           required=False,
                           help="evaluation task type")
        group.add_argument("--nodebug",
                           action="store_true",
                           required=False,
                           help="debug or not")
        
        group.add_argument("--PRETRAINED_VAE",
                           type=str,
                           required=False,
                           help="model task type")
        
        group.add_argument("--test_t2m_model",
                    type=str,
                    default=None,
                    required=False,
                    help="model task type")
        group.add_argument("--render_predictions",
                    action="store_true",
                    required=False,
                    help="render predicted hand motion during test")
        group.add_argument("--dump_npy",
                    action="store_true",
                    required=False,
                    help="dump predicted motion as npy during test")
        group.add_argument("--dump_obj",
                    action="store_true",
                    required=False,
                    help="dump predicted meshes as obj during test")
        group.add_argument("--render_limit",
                    type=int,
                    default=0,
                    required=False,
                    help="max number of samples to visualize; 0 means no limit")
        group.add_argument("--prompt",
                    action="append",
                    required=False,
                    help="prompt text for direct T2M generation; can be passed multiple times")
        group.add_argument("--prompts_file",
                    type=str,
                    default=None,
                    required=False,
                    help="path to a text or json file containing prompts for direct T2M generation")
        group.add_argument("--num_prompt_samples",
                    type=int,
                    default=1,
                    required=False,
                    help="number of generations per prompt in direct T2M mode")
        group.add_argument("--prompt_motion_length",
                    type=int,
                    default=144,
                    required=False,
                    help="reference motion length used for direct T2M generation")


    if phase == "demo":
        group.add_argument("--task",
            type=str,
            required=False,
            help="evaluation task type")
        group.add_argument(
            "--example",
            type=str,
            required=False,
            help="input text and lengths with txt format",
        )
        group.add_argument(
            "--out_dir",
            type=str,
            required=False,
            help="output dir",
        )

    params = parser.parse_args()
    if phase == "test" and "default" in params.cfg:
        ## load the train cfg
        cfg_path = params.test_t2m_model.split("/checkpoints/")[0]
        cfg_file = os.path.join(cfg_path, "input_train_config.yaml")
        assert os.path.exists(cfg_file)
        params.cfg = cfg_file
        print("\nAssigning the train cfg to load: ", cfg_file)


    # Load yaml config files
    OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, 'default.yaml'))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)

    ### merging assests to exp config
    # cfg = OmegaConf.merge(cfg_assets, cfg_assets)

    ### merging exp to assets config to new exp dir etc
    cfg = OmegaConf.merge(cfg_assets, cfg_exp)
    

    ### updating the folder fro


    # Update config with arguments
    if phase in ["train", "test"]:
        cfg.TRAIN.BATCH_SIZE = params.batch_size if params.batch_size else cfg.TRAIN.BATCH_SIZE
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.NUM_NODES = params.num_nodes if params.num_nodes else cfg.NUM_NODES
        cfg.model.params.task = params.task if params.task else cfg.model.params.task
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG
        ## added by bala
        cfg.PRETRAINED_VAE = params.PRETRAINED_VAE
        cfg.test_t2m_model = params.test_t2m_model
        cfg.cfg_file = params.cfg
        cfg.render_predictions = params.render_predictions
        cfg.dump_npy = params.dump_npy
        cfg.dump_obj = params.dump_obj
        cfg.render_limit = params.render_limit
        cfg.prompt = params.prompt
        cfg.prompts_file = params.prompts_file
        cfg.num_prompt_samples = params.num_prompt_samples
        cfg.prompt_motion_length = params.prompt_motion_length


        # Force no debug in test
        if phase == "test":
            cfg.DEBUG = False
            # cfg.DEVICE = [0]
            # print("Force no debugging and one gpu when testing")


        # Logger
        if cfg.test_t2m_model is not None:
            cfg.TEST.CHECKPOINTS = cfg.test_t2m_model
            print(f"\nUpdating cfg.TEST.CHECKPOINTS to the parse args {cfg.TEST.CHECKPOINTS}\n")

    if phase == "demo":
        cfg.DEMO.EXAMPLE = params.example
        cfg.DEMO.TASK = params.task
        cfg.TEST.FOLDER = params.out_dir if params.out_dir else cfg.TEST.FOLDER
        os.makedirs(cfg.TEST.FOLDER, exist_ok=True)

    ### added by bala
    if "USE_FILE_NAME" in cfg.NAME:
        cfg.NAME = params.cfg.split("/")[-1].split(".")[0]

    # Debug mode
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.params.offline = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1


        
    # Resume config
    cfg = resume_config(cfg)

    return cfg
