import json
import os
import sys
import numpy as np
import pytorch_lightning as pl
import torch
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mGPT.config import parse_args
from mGPT.hand.body_models.mano_xx import mano_full_pose_to_mano_params
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae
from tqdm import tqdm

def resolve_device(cfg):
    if torch.cuda.is_available() and cfg.ACCELERATOR == "gpu":
        return torch.device(f"cuda:{cfg.DEVICE[0]}")
    cfg.ACCELERATOR = "cpu"
    cfg.DEVICE = 1
    return torch.device("cpu")


def maybe_dump_visualizations(cfg, model, batch_out, output_dir, batch, rendered_count):
    if not (cfg.render_predictions or cfg.dump_npy or cfg.dump_obj):
        return rendered_count

    if cfg.render_limit and rendered_count >= cfg.render_limit:
        return rendered_count

    vis_dir = os.path.join(output_dir, "visualizations")
    npy_dir = os.path.join(output_dir, "npy")
    obj_dir = os.path.join(output_dir, "obj")

    curr_pred = curr_name = None
    if cfg.render_predictions or cfg.dump_obj:
        curr_pred, curr_name = extract_prediction_mesh(model, batch_out, batch)
        if cfg.render_predictions and getattr(model, "render_helper", None) is not None:
            os.makedirs(vis_dir, exist_ok=True)
            model.visualize_test_results(batch_out, vis_dir, batch)

    if cfg.dump_npy:
        os.makedirs(npy_dir, exist_ok=True)
        model.dump_as_npy(batch_out, npy_dir, batch)

    if cfg.dump_obj and curr_pred is not None and curr_name is not None:
        dump_objs(curr_pred, curr_name, model.mano_doubleX.faces, obj_dir)

    return rendered_count + 1


def dump_objs(curr_pred, curr_name, faces, out_folder):
    import trimesh

    pred_obj_folder = os.path.join(out_folder, curr_name)
    os.makedirs(pred_obj_folder, exist_ok=True)

    for frame in tqdm(range(curr_pred.shape[0]), desc=f"Dumping objs {curr_name}"):
        mesh = trimesh.Trimesh(
            vertices=curr_pred[frame],
            faces=faces,
            process=False,
        )
        mesh.export(os.path.join(pred_obj_folder, f"{frame}.obj"))


def extract_prediction_mesh(model, batch_out, batch):
    model_output = model.datamodule.denormalize(batch_out["m_rst"])
    pred_mano_full_pose = model_output[:1]
    pred_mano_params = mano_full_pose_to_mano_params(pred_mano_full_pose)
    curr_pred = model.mano_doubleX.get_scene_verts_from_batch(**pred_mano_params)[0]
    return curr_pred, batch["seq_name"][0]


def resolve_prompt_list(cfg):
    prompts = list(cfg.prompt or [])
    if cfg.prompts_file:
        with open(cfg.prompts_file, "r", encoding="utf-8") as handle:
            if cfg.prompts_file.endswith(".json"):
                file_prompts = json.load(handle)
                if not isinstance(file_prompts, list):
                    raise ValueError("prompts_file JSON must contain a list of prompt strings.")
                prompts.extend(file_prompts)
            else:
                prompts.extend(
                    line.strip() for line in handle.readlines() if line.strip()
                )
    return prompts


class PromptOnlyDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = "egovid5M"
        self.njoints = 21
        self.nfeats = 198
        norm_relpath = cfg.DATASET.get("norm_dict_file_to_load", "")
        full_path = os.path.join(cfg.DATASET.dataset_dir, norm_relpath) if norm_relpath else ""

        self.hparams = SimpleNamespace(
            fps=cfg.DATASET.fps,
            data_root=cfg.DATASET.dataset_dir,
            mean=None,
            std=None,
            min=None,
            max=None,
            name=self.name,
            w_vectorizer=None,
        )

        if full_path and os.path.exists(full_path) and "mean_std" in norm_relpath:
            stat_dict = np.load(full_path, allow_pickle=True).item()
            self.hparams.mean = stat_dict["mean"].clone().detach().reshape(1, 1, -1).float()
            self.hparams.std = stat_dict["std"].clone().detach().reshape(1, 1, -1).float()

    def feats2joints(self, features):
        return features

    def renorm4t2m(self, features):
        return features

    def denormalize(self, features):
        if self.hparams.mean is None:
            return features
        mean = self.hparams.mean.to(features.device, dtype=features.dtype)
        std = self.hparams.std.to(features.device, dtype=features.dtype)
        return features * std + mean


def build_prompt_batch(text, sample_idx, max_seq_len, nfeats, task_template, device):
    safe_name = "_".join(text.split())[:160] or "prompt"
    return {
        "text": [text],
        "motion_ref": torch.zeros((1, max_seq_len, nfeats), device=device),
        "length": [max_seq_len],
        "all_captions": [[text]],
        "seq_name": [f"{safe_name}_ns{sample_idx}"],
        "tasks": [task_template],
    }


def run_prompt_generation(cfg, model, output_dir, device, logger):
    prompts = resolve_prompt_list(cfg)
    if not prompts:
        raise ValueError("Prompt mode requires at least one --prompt or a non-empty --prompts_file.")

    task_template = {
        "class": "t2m",
        "input": ["<Caption_Placeholder>"],
        "output": ["<Motion_Placeholder>"],
    }
    rendered_count = 0

    for prompt in prompts:
        logger.info(f"Generating from prompt: {prompt}")
        for sample_idx in range(cfg.num_prompt_samples):
            batch = build_prompt_batch(
                prompt,
                sample_idx,
                cfg.prompt_motion_length,
                model.datamodule.nfeats,
                task_template,
                device,
            )
            batch_out = model.val_t2m_forward(batch)
            rendered_count = maybe_dump_visualizations(
                cfg, model, batch_out, output_dir, batch, rendered_count
            )

    logger.info("Prompt-list generation completed.")


def main():
    
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.EVAL.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 1
    cfg.TRAIN.NUM_WORKERS = 0
    cfg.EVAL.NUM_WORKERS = 0
    cfg.TEST.NUM_WORKERS = 0
    cfg.DATASET.JOINT_TYPE = "egovid5M"
    cfg.DATASET.NFEATS = 198
    prompt_list = resolve_prompt_list(cfg)
    if not prompt_list:
        raise ValueError("test_t2m.py is now demo-style prompt generation only. Pass --prompt or --prompts_file.")

    if "lm" not in str(cfg.TRAIN.STAGE):
        raise ValueError(
            "test_t2m.py requires a Stage 2/3/4 language-model checkpoint. "
            f"Received TRAIN.STAGE={cfg.TRAIN.STAGE!r}, which is not a T2M inference checkpoint."
        )

    logger = create_logger(cfg, phase="test_w_visual_results")
    output_dir = cfg.TEST_FOLDER_EXP 
    device = resolve_device(cfg)

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cfg.METRIC.TYPE = []
    both_ann_datamodule = PromptOnlyDataModule(cfg)

    logger.info("datasets module {} initialized".format("".join(cfg.DATASET.target.split('.')[-2])))

    """
    Loading model
    """
    model = build_model(cfg, both_ann_datamodule) # its a pylighting module
    logger.info("model {} loaded".format(cfg.model.target))
    ### Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)
    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        load_pretrained(cfg, model, logger, phase="test")
    else:
        logger.warning("No checkpoints provided!!!")

    if not hasattr(model, "lm"):
        raise ValueError(
            "Loaded checkpoint/config does not contain a language model. "
            "Please provide a Stage 2/3/4 T2M checkpoint instead of a Stage 1 VQ-VAE checkpoint."
        )
    ### model vae
    model = model.to(device)
    model.vae = model.vae.to(device)
    model = model.eval()
    run_prompt_generation(cfg, model, output_dir, device, logger)
            

if __name__ == "__main__":
    main()
