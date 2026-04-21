import random
import numpy as np
from torch.utils import data
from .g_dataset_t2m import Text2MotionDataset
import codecs as cs
from os.path import join as pjoin
from .loader_dataset import LoadData

class Text2MotionDatasetToken(LoadData):

    def __init__(
        self,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        **kwargs,
    ):
        
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length

        # Data mean and std
        self.mean = kwargs.get("mean")# 1 x 198
        self.std = kwargs.get("std") # 1 x 198

        if self.mean is not None:
            self.mean = self.mean.numpy()[0] # 1 x 198
            self.std = self.std.numpy()[0] # 1 x 198

        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.data_dict)  
        
    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']
        caption = data['text'][0]['caption']

        # m_length = (m_length // self.unit_length) * self.unit_length

        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx+m_length]

        if self.mean is not None:
            "Z Normalization"
            motion = (motion - self.mean) / self.std

        out_dict = {

            # text
            "text": caption,
            "motion": motion,
            "length": m_length, # idx 2
            "name": name,
            "seq_name":self.name_list[item]
        }

        if data.get("seq_mask", None) is not None:
            out_dict["motion_mask"] = data.get("seq_mask")

        return out_dict

        # return name, motion, m_length, True, True, True, True, True, True
