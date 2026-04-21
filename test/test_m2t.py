import json
import os
import sys
import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae
from tqdm import tqdm

def print_table(title, metrics, logger=None):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = float(value)
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")

    logger.info(metrics) if logger else None


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def resolve_device(cfg):
    if torch.cuda.is_available() and cfg.ACCELERATOR == "gpu":
        return torch.device(f"cuda:{cfg.DEVICE[0]}")
    cfg.ACCELERATOR = "cpu"
    cfg.DEVICE = 1
    return torch.device("cpu")


def main():
    
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.TRAIN.BATCH_SIZE = 8
    cfg.EVAL.BATCH_SIZE = 8
    cfg.TEST.BATCH_SIZE = 8
    cfg.TRAIN.NUM_WORKERS = 0
    cfg.EVAL.NUM_WORKERS = 0
    cfg.TEST.NUM_WORKERS = 0

    cfg.DATASET.tasks_to_load = ["Text-to-Motion"]
    cfg.EVAL.DatasetEval = "Text2MotionDatasetEval"
    if "lm" not in str(cfg.TRAIN.STAGE):
        raise ValueError(
            "test_m2t.py requires a Stage 2/3/4 language-model checkpoint. "
            f"Received TRAIN.STAGE={cfg.TRAIN.STAGE!r}, which is not an M2T inference checkpoint."
        )

    logger = create_logger(cfg, phase="compute_m2t_metric", sub_dir="_full_random_choice")
    output_dir = cfg.TEST_FOLDER_EXP 
    device = resolve_device(cfg)

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    """
    Setting up the dataset to run
    """
    replication_times=5
    cfg.DATASET.GRAB_to_load = False
    cfg.DATASET.egovid5M_to_load = True

    cfg.DATASET.annotation_file_to_load = ["naive_claude_summary.json", "vila_cpot_cls_summ.json"]
    cfg.DATASET.ann_sample_probs= [0.5, 0.5] 
    both_ann_datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(cfg.DATASET.target.split('.')[-2])))

    """
    Metrics setup
    """
    ### LLM text 2 motion metrics
    metric = None
    tm2t_ckpt = cfg.METRIC.TM2T.get("ckpt", "")
    if tm2t_ckpt and not tm2t_ckpt.startswith("/path/to/") and os.path.exists(tm2t_ckpt):
        from mGPT.metrics.quan_eval.egovid5m_m2t import Egovid5M_M2TMetrics
        metric = Egovid5M_M2TMetrics(cfg, w_vectorizer=both_ann_datamodule.hparams.w_vectorizer)
    else:
        logger.warning("TM2T evaluator checkpoint is not configured; running inference smoke test without quantitative metrics.")


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
            "Please provide a Stage 2/3/4 M2T checkpoint instead of a Stage 1 VQ-VAE checkpoint."
        )

    model = model.to(device)
    model.vae = model.vae.to(device)
    model = model.eval()

    from mGPT.hand.utils.temporal_dict import temporal_dict
    seq_dict_for_all_runs = temporal_dict()
    for i in range(replication_times):
        logger.info(f"Running metric for run {i}\n")

        for dataloader in [both_ann_datamodule.test_dataloader()]:
            for batch in tqdm(dataloader, desc=f'Running eval'): 

                text_name = "_".join(batch["text"][0].split(" ")[:10])
                batch["seq_name"][0]  = batch["seq_name"][0] + f"_{text_name}_ns_{i}"
                device_batch = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        device_batch[k] = v.to(device)
                    else:
                        device_batch[k] = v

                if device_batch.get("motion_ref", None) is None:
                    device_batch["motion_ref"] = device_batch["motion"]
                
                motion_len = device_batch.get("motion_mask").sum(dim=1)
                motion_len = motion_len.cpu().long()
                motion_len = [ int(motion_len[x]) for x in range(motion_len.shape[0])]

                batch_out = model.val_m2t_forward(device_batch)
                if metric is not None:
                    metric.update(
                            feats_ref=device_batch["motion_ref"],
                            pred_texts=batch_out["t_pred"],
                            gt_texts=device_batch["text"],
                            lengths=motion_len,
                            word_embs=device_batch["word_embs"],
                            pos_ohot=device_batch["pos_ohot"],
                            text_lengths=device_batch["text_len"],
                        )

        if metric is None:
            logger.info("Inference smoke test completed for the main pass.")
            return

        out_metric = metric.compute(False)
        logger.info(f"Compute metric for run {i}")
        print_table(f"Egovid5M m2t Metrics", out_metric, logger=logger)
        logger.info("\n**************************************")
        seq_dict_for_all_runs.add_dict(out_metric)
        logger.info("**************************************")
        logger.info("**************************************")
        logger.info("**************************************")

    run_wise_out_file = os.path.join(output_dir, f"metric_run_wise.csv")
    seq_dict_for_all_runs.dumps_as_csv(run_wise_out_file)
    
    mean_conf_outfile = os.path.join(output_dir, f"mean_conf_metric.json")
    mean_conf_dict = seq_dict_for_all_runs.compute_mean_conf_dict(replication_times=replication_times)
    seq_dict_for_all_runs.dump_mean_conf_dict(mean_conf_outfile, out_metric=mean_conf_dict)
    print_table(f"Final dict of run", mean_conf_dict, logger=logger)
            

if __name__ == "__main__":
    main()
