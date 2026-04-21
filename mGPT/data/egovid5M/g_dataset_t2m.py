import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .loader_dataset import LoadData

class Text2MotionDataset(LoadData):

    def __init__(
        self,
        # data_root,
        # split,
        # mean,
        # std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        # fps=20,
        # tmpFile=True,
        # tiny=False,
        # debug=False,
        **kwargs,
    ):
        self.max_length = 20
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
        return len(self.name_list) - self.pointer
    
    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data["caption"]

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Z Normalization
        if self.mean is not None:
            motion = (motion - self.mean) / self.std
            
        return caption, motion, m_length, None, None, None, None, all_captions
    
        # return caption, motion, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(tokens), all_captions

        # return 
        # caption, # A person stops for a moment and then runs to the left.
        # motion, # motion.shape = (20 x 251)
        # m_length, # 20
        # word_embeddings, # (22, 300); SOS + text token + [Unk] * buffer + EOS 
        # pos_one_hots,  # pos_one_hots.shape (22, 15)
        # sent_len, # actual sent len = 14
        #  "_".join(tokens), #### sos/OTHER_A/DET_person/NOUN_stop/VERB_for/ADP_a/DET_moment/NOUN_and/CCONJ_then/ADV_run/VERB_to/ADP_the/DET_left/NOUN_eos/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER
        # all_captions
        ####
        ## A person stop for a moment and then run to the left
        ## A person stop for a moment and then run to the left
        ## A person stop for a moment and then run to the left

