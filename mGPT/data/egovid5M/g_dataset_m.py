import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
# from .loader_dataset import LoadData
from .g_dataset_t2m import Text2MotionDataset

class MotionDataset(Text2MotionDataset):
    def __init__(
        self,
        # data_root,
        # split,
        # mean,
        # std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,

        **kwargs,
    ):

        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length

        self.mean = kwargs.get("mean")# 1 x 198
        self.std = kwargs.get("std") # 1 x 198

        if self.mean is not None:
            self.mean = self.mean.numpy()[0] # 1 x 198
            self.std = self.std.numpy()[0] # 1 x 198


        # print("Text2MotionDataset", tiny, debug)
        
        ## loading the dict
        super().__init__(**kwargs)
   
    def __len__(self):
        return len(self.name_list)

    # def __getitem__(self, item):
    #     data = self.data_dict[self.name_list[item]]
    #     motion_list, m_length = data["motion"], data["length"]

    #     # Randomly select a motion
    #     motion = random.choice(motion_list)

    #     # # Crop the motions in to times of 4, and introduce small variations
    #     # if self.unit_length < 10:
    #     #     coin2 = np.random.choice(["single", "single", "double"])
    #     # else:
    #     #     coin2 = "single"

    #     # if coin2 == "double":
    #     #     m_length = (m_length // self.unit_length - 1) * self.unit_length
    #     # elif coin2 == "single":
    #     #     m_length = (m_length // self.unit_length) * self.unit_length
    #     # idx = random.randint(0, len(motion) - m_length)
    #     # motion = motion[idx:idx + m_length]
        
    #     if self.mean is not None:
    #         # Z Normalization
    #         motion = (motion - self.mean) / self.std

    #     return None, motion, m_length, None, None, None, None,

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, length = data["motion"], data["length"]

        # idx = random.randint(0, motion.shape[0] - self.window_size)
        # motion = motion[idx:idx + self.window_size]

        if self.mean is not None:
            motion = (motion - self.mean) / self.std

        out_dict = {
            "motion": motion,
            "length": length,
            "seq_name": self.name_list[idx],
            "sentence": data["sentence"],
        }
        return out_dict
