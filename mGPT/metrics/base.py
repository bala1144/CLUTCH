from torch import Tensor, nn
from os.path import join as pjoin
from .mr import MRMetrics
from .t2m import TM2TMetrics
from .mm import MMMetrics
from .m2t import M2TMetrics
from .grab import GRABMetrics
from .egovid5m import Egovid5M_Metrics
from .egovid5m_m2t import Egovid5M_M2TMetrics
from .t2mt_metric import TM2TMetrics_V2
from .quan_eval.egovid5m_t2m import Egovid5M_TM2TMetrics as Egovid5M_TM2TMetrics_t2m
from .quan_eval.egovid5m_m2t import Egovid5M_M2TMetrics as Egovid5M_TM2TMetrics_m2t
# from .quan_eval.egovid5m_mr import MRMetrics as Egovid5M_TM2TMetrics_mr

class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
        super().__init__()

        njoints = datamodule.njoints
        data_name = datamodule.name
        if data_name in ["humanml3d", "kit"]:
            # self.TM2TMetrics = TM2TMetrics(
            #     cfg=cfg,
            #     dataname=data_name,
            #     diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
            #     dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            # )
            # self.M2TMetrics = M2TMetrics(
            #     cfg=cfg,
            #     dataname=data_name,
            #     w_vectorizer=datamodule.hparams.w_vectorizer,
            #     diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
            #     dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)
            # self.MMMetrics = MMMetrics(
            #     cfg=cfg,
            #     dataname=data_name,
            #     mm_num_times=cfg.METRIC.MM_NUM_TIMES,
            #     dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            # )
            pass

        self.MRMetrics = MRMetrics(
            njoints=njoints,
            jointstype=cfg.DATASET.JOINT_TYPE,
            dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
        )

        if  "GRABMetrics" in cfg.METRIC.TYPE:
            ### mmy metics
            self.GRABMetrics = GRABMetrics(
                cfg=cfg,
                dataname=data_name,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )

        if  "Egovid5M_Metrics" in cfg.METRIC.TYPE:
            self.Egovid5M_Metrics = Egovid5M_Metrics(
                cfg=cfg,
                dataname=data_name,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )

        ### this is added for fixing the developement
        if  "Egovid5M_M2TMetrics" in cfg.METRIC.TYPE:
            self.Egovid5M_M2TMetrics = Egovid5M_M2TMetrics(
                cfg=cfg,
                dataname=data_name,
                w_vectorizer=datamodule.hparams.w_vectorizer,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)

        
        #### used for training the constrastive learning model
        ### this is added for fixing the developement
        if  "TM2TMetrics_V2" in cfg.METRIC.TYPE:
            self.TM2TMetrics_V2 = TM2TMetrics_V2(
                cfg=cfg,
                dataname=data_name,
                w_vectorizer=datamodule.hparams.w_vectorizer,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)

        ######## QUANTITATIVE EVAL METRICS
        ### EGOVID5M T2MT t2m
        if  "Egovid5M_TM2TMetrics_t2m" in cfg.METRIC.TYPE:
            self.Egovid5M_TM2TMetrics_t2m = Egovid5M_TM2TMetrics_t2m(
                cfg=cfg,
                dataname=data_name,
                w_vectorizer=datamodule.hparams.w_vectorizer,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)


