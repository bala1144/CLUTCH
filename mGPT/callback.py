import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []
    logger = logger

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))
        
    return callbacks

def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []

    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        # "Train_jf": "recons/text2jfeats/train",
        # "Val_jf": "recons/text2jfeats/val",
        # "Train_rf": "recons/text2rfeats/train",
        # "Val_rf": "recons/text2rfeats/val",
        # "APE root": "Metrics/APE_root",
        # "APE mean pose": "Metrics/APE_mean_pose",
        # "AVE root": "Metrics/AVE_root",
        # "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/Bleu_1",
        "R_TOP_1": "Metrics/ROUGE_L",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",

    }

    ## GRAB
    metrics = ['Recon','JTS_Recon', "wrist_JT_recon",
                'AC_div', 'AC_multi_mod', 'AC_gt_div', 'AC_gt_multi_mod', 'WT_div', 'WT_multi_mod', 'FID']
    for met in metrics:
        metric_monitor[met] =  f"Metrics/{met}"

    callbacks.append(
        progressLogger(logger,metric_monitor=metric_monitor,log_every_n_steps=1))



    ## handing the checkpoint default 
    if cfg.LOGGER.get("CHECKPOINT", None) is None:
       cfg.LOGGER.CHECKPOINT = ["save_every_n","metric_checkpoint"]
    
    # Save 10 latest checkpoints
    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 4,
        'save_last': True,
        'save_on_train_epoch_end': False # True, previous
    }

    if "save_last_n" in cfg.LOGGER.CHECKPOINT:
        callbacks.append(ModelCheckpoint(**checkpointParams))

    # Save checkpoint every n*10 epochs
    checkpointParams.update({
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': -1,
        'save_last': False
    })
    if "save_every_n"  in cfg.LOGGER.CHECKPOINT:
        callbacks.append(ModelCheckpoint(**checkpointParams))

    if "metric_checkpoint"  in cfg.LOGGER.CHECKPOINT:

        metrics = cfg.METRIC.TYPE
        metric_monitor_map = {
            'TemosMetric': {
                'Metrics/APE_root': {
                    'abbr': 'APEroot',
                    'mode': 'min'
                },
            },
            'TM2TMetrics': {
                'Metrics/FID': {
                    'abbr': 'FID',
                    'mode': 'min'
                },
                'Metrics/R_precision_top_3': {
                    'abbr': 'R3',
                    'mode': 'max'
                } 
            },
            'MRMetrics': {
                'Metrics/MPJPE': {
                    'abbr': 'MPJPE',
                    'mode': 'min'
                }
            },
            'HUMANACTMetrics': {
                'Metrics/Accuracy': {
                    'abbr': 'Accuracy',
                    'mode': 'max'
                }
            },
            'UESTCMetrics': {
                'Metrics/Accuracy': {
                    'abbr': 'Accuracy',
                    'mode': 'max'
                }
            },
            'UncondMetrics': {
                'Metrics/FID': {
                    'abbr': 'FID',
                    'mode': 'min'
                }
            },
            ### GRABMetrics
            'GRABMetrics': {
                'Metrics/FID': {
                    'abbr': 'FID',
                    'mode': 'min'
                },
                'Metrics/Recon': {
                    'abbr': 'Recon',
                    'mode': 'min'
                }
                
            },

            ### GRABMetrics
            'Egovid5M_Metrics': {
                'Metrics/wrist_JT_recon': {
                    'abbr': 'wrist_JT_recon',
                    'mode': 'min'
                },
                'Metrics/Recon': {
                    'abbr': 'Recon',
                    'mode': 'min'
                }
                
            },


        }

        checkpointParams.update({
            'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
            'save_top_k': 1,
        })

        for metric in metrics:
            if metric in metric_monitor_map.keys():
                metric_monitors = metric_monitor_map[metric]

                # Delete R3 if training VAE
                if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
                    del metric_monitors['Metrics/R_precision_top_3']

                for metric_monitor in metric_monitors:
                    checkpointParams.update({
                        'filename':
                        metric_monitor_map[metric][metric_monitor]['mode']
                        + "-" +
                        metric_monitor_map[metric][metric_monitor]['abbr']
                        + "{ep}",
                        'monitor':
                        metric_monitor,
                        'mode':
                        metric_monitor_map[metric][metric_monitor]['mode'],
                    })
                    callbacks.append(ModelCheckpoint(**checkpointParams))
  
    ### add custom save checkpoints
    if cfg.LOGGER.get("SaveSpecificEpochsCallback", None) is not None:
        checkpointParams = {
            'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            'save_epochs': cfg.LOGGER.SaveSpecificEpochsCallback,
            }
        callbacks.append(SaveSpecificEpochsCallback(**checkpointParams))

    return callbacks

class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)



class SaveSpecificEpochsCallback(Callback):
    def __init__(self, dirpath, save_epochs, **kwargs):
        """
        Args:
            save_epochs (list): List of epochs (1-indexed) at which to save the model.
            save_path (str): Base path to save the model checkpoints.
        """
        super().__init__(**kwargs)
        self.save_epochs = save_epochs # list
        self.dirpath = dirpath

    def on_train_epoch_end(self, trainer, pl_module):

        # if trainer.global_rank == 0:
        # `trainer.current_epoch` is 0-indexed, add 1 for human-readable indexing
        current_epoch = trainer.current_epoch
        if (current_epoch+1) in self.save_epochs:
            save_file = f"{self.dirpath}/epoch={current_epoch}.ckpt"
            trainer.save_checkpoint(save_file)
            print(f"Model checkpoint saved at epoch {current_epoch} to {save_file}")

