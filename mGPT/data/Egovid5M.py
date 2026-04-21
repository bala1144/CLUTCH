import numpy as np
import torch
import os 
from os.path import join as pjoin
from .grab.utils.word_vectorizer import WordVectorizer
from .grab.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .egovid5M import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T, MotionDataset_CLS, MotionDataset_CONTRAST, Text2MotionDataset_Diff
from .utils import grab_collate


class DataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=grab_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'egovid5M'
        self.name = "egovid5M"
        self.njoints = 21
        
        ### add extract CFG and add to hyperparams
        for key in cfg.DATASET:
            setattr(self.hparams, key, cfg.DATASET[key])
        self.hparams.data_root = cfg.DATASET.dataset_dir # for backward compatabitliy

         #### self load the mean dict
        self.hparams.mean = None
        self.hparams.std = None
        self.hparams.min = None
        self.hparams.max = None
        self.hparams.name = self.name
        self.hparams.stage = cfg.TRAIN.STAGE


        self.normalization_dict_file = cfg.DATASET.get("norm_dict_file_to_load", False) 
        if self.normalization_dict_file and "mean_std" in self.normalization_dict_file:
            full_path = os.path.join(cfg.DATASET.dataset_dir, self.normalization_dict_file)
            assert os.path.exists(full_path), f"{full_path} does not exist"
            stat_dict = np.load(full_path, allow_pickle=True).item()
            self.hparams.mean = stat_dict["mean"].reshape(1, 1, -1)
            self.hparams.std = stat_dict["std"].reshape(1, 1, -1)
            print()
            print("********** Using Std Mean normalization **********")
            print()

        elif self.normalization_dict_file and "min_max" in self.normalization_dict_file:
            full_path = os.path.join(cfg.DATASET.dataset_dir, self.normalization_dict_file)
            assert os.path.exists(full_path), f"{full_path} does not exist"
            stat_dict = np.load(full_path, allow_pickle=True).item()

            self.hparams.min = stat_dict["min"].reshape(1, 1, -1)
            self.hparams.max = stat_dict["max"].reshape(1, 1, -1)
            print()
            print("********** Using Std Mean normalization **********")
            print()
        
        else:
            print("*********************************************")
            print("********** Using No normalization **********")
            print("*********************************************")


        ##### Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.GRAB.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.GRAB.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.GRAB.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.GRAB.UNIT_LEN
        self.hparams.w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # Dataset switch
        self.DatasetEval = Text2MotionDatasetEval
        if cfg.TRAIN.STAGE in [ "vae"] :
            if cfg.model.params.motion_vae.target.split('.')[-1].lower() in ["vqvae", "rvqvae", "classifier"]:
                self.hparams.win_size = kwargs.get("win_size", 20)
                self.Dataset = MotionDatasetVQ
            else:
                self.Dataset = MotionDataset
        
        elif cfg.TRAIN.STAGE in [ "classifier"] :
            self.Dataset = MotionDataset_CLS
            self.DatasetEval = MotionDataset_CLS

        elif cfg.TRAIN.STAGE in [ "contrastive"] :
            self.Dataset = MotionDataset_CONTRAST
            self.DatasetEval = Text2MotionDatasetEval

        elif cfg.TRAIN.STAGE in [ "diffusion"]:
            self.Dataset = Text2MotionDataset_Diff
            self.DatasetEval = Text2MotionDataset_Diff
    
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.GRAB.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
            # if cfg.EVAL.get("DatasetEval", "Text2MotionDatasetEval"):
            self.DatasetEval = eval(cfg.EVAL.get("DatasetEval", "Text2MotionDatasetEval"))

        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
            
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset

        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "test", "seqs_split.test":[0, 1]})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats

    def feats2joints(self, features):
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = features * std + mean
        # return recover_from_ric(features, self.njoints)
        return features
    
    def joints2feats(self, features):
        # example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        # example_data = example_data.reshape(len(example_data), -1, 3)
        # example_data = torch.from_numpy(example_data)
        # features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):

        if self.hparams.mean is not None:
            mean = self.hparams.mean.to(features)
            std = self.hparams.std.to(features)
            features = (features - mean) / std

        elif self.hparams.min is not None:
            features = (2 * (features - self.hparams.min.to(features.device)) / (self.hparams.min.to(features.device) - self.hparams.max.to(features.device)) - 1)

        return features

    def denormalize(self, features):

        if self.hparams.mean is not None:
            mean = self.hparams.mean.to(features)
            std = self.hparams.std.to(features)
            features = features * std + mean

        elif self.hparams.min is not None:
            features  = ((features + 1) * (self.hparams.max.to(features.device) - self.hparams.min.to(features.device)) / 2 + self.hparams.min.to(features.device))

        return features

    def renorm4t2m(self, features):

        # if self.hparams.mean is not None:
        #     # renorm to t2m norms for using t2m evaluators
        #     ori_mean = torch.tensor(self.hparams.mean).to(features)
        #     ori_std = torch.tensor(self.hparams.std).to(features)
        #     eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        #     eval_std = torch.tensor(self.hparams.std_eval).to(features)
        #     features = features * ori_std + ori_mean
        #     features = (features - eval_mean) / eval_std

        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list

            if  self.cfg.METRIC.MM_NUM_SAMPLES < len(self.name_list):
                self.mm_list = np.random.choice(self.name_list,
                                                self.cfg.METRIC.MM_NUM_SAMPLES,
                                                replace=False)
            else:
                self.mm_list =  self.name_list 

            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
