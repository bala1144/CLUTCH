import rich
import random
import pickle
import os
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
from mGPT.hand.utils.dict_to_struct import dict_to_struct
from .loader_dataset import LoadData

import torch
from glob import glob

class Text2MotionDatasetCB(LoadData):
    def __init__(
        self,
        # data_root,
        # split,
        # mean,
        # std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text= False,
        tasks_to_load =["Text-to-Motion"],
        pretrain_template_file = 'template_pretrain.json',
        instructions_template_file = 'template_instructions.json',
        ann_sample_probs = None,
        **kwargs,
    ):
        
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.std_text = std_text
        self.ann_sample_probs = ann_sample_probs

        # Data mean and std
        self.mean = kwargs.get("mean")# 1 x 198
        self.std = kwargs.get("std") # 1 x 198
        self.split =  kwargs.get("split")

        if self.mean is not None:
            self.mean = self.mean.numpy()[0] # 1 x 198
            self.std = self.std.numpy()[0] # 1 x 198

        self.w_vectorizer = kwargs.get("w_vectorizer")

        super().__init__(**kwargs)
        
        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain':
            instructions = pjoin(self.data_root, pretrain_template_file)
        elif stage in ['lm_instruct', "lm_rl"]:
            instructions = pjoin(self.data_root, instructions_template_file)
        else:
            raise NotImplementedError(f"stage {stage} not implemented")


        ### load the motion dict
        motion_token_dict = {}
        text_dir = None
        for i, name in enumerate(self.data_dict.keys()):
             if os.path.exists(pjoin(self.data_root, code_path, f'{name}.npy')):
                m_token_list = np.load(pjoin(self.data_root, code_path, f'{name}.npy'))
                motion_token_dict[name] = m_token_list

        self.motion_token_dict = motion_token_dict

        ### subssample after the motion is filtered
        new_data_dict = { k:self.data_dict[k] for k in self.motion_token_dict.keys() }
        self.data_dict_local = new_data_dict
        self.name_list = list(new_data_dict.keys())
        
        ### process the tasks
        self.nlp = spacy.load('en_core_web_sm')
        print("loading instructions", instructions)
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            if task in tasks_to_load:
                for subtask in self.instructions[task].keys():
                    self.tasks.append(self.instructions[task][subtask])

        print("**************************************************")
        print(f"Total number of motions {len(self.name_list)}")
        print(f"Total number of Instruction: {len(self.instructions.keys())}")
        print(f"Total number of Tasks: {len(self.tasks)}")
        print("**************************************************")

        del self.data_dict

    def __len__(self):
        # return len(self.data_dict_local) * len(self.instructions)
        return len(self.data_dict_local) * len(self.tasks)

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = item // len(self.name_list)

        name = self.name_list[data_idx]
        data = self.data_dict_local[name]
        text_list =  data['text']

        # get motion 
        m_token_list = self.motion_token_dict[name]
        m_tokens = random.choice(m_token_list)

        # get text
        if len(text_list) > 1:
            # finding the model type
            if self.ann_sample_probs is None:
                text_data = random.choice(text_list)
            else:
                assert len(self.ann_sample_probs) == len(text_list)
                text_data = random.choices(text_list, weights=self.ann_sample_probs, k=1)[0]
        else:
            text_data = text_list[0]
        caption, tokens = text_data["caption"], text_data["tokens"]

        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN'
                        or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
            caption = ' '.join(word_list)
        
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        m_tokens_len = m_tokens.shape[0]
        tasks = self.tasks[task_idx]

        if self.mean is not None:
            motion_ref = (data["motion"] - self.mean) / self.std

        # Text
        max_text_len = 40
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        out_dict = {

            # text
            "text": caption,
            "motion": m_tokens,
            "length": m_tokens_len, # idx 2 this should be the motion len
            # "length": motion_ref.shape[0], # idx 2 this should be the motion len

            "word_embs": word_embeddings, # idx 3
            "pos_ohot": pos_one_hots, # idx 4
            "text_len":sent_len, # idx 5
            "tokens": "_".join(tokens), # 6

            "all_captions": all_captions, # idx 7
            "tasks": tasks,
            "seq_name": name, # idx 9

            ### motion for ref
            "motion_ref": motion_ref,
            "sentence": data["sentence"]

        }

        # print(f"{name}: len {m_tokens_len}")
        if data.get("seq_mask", None) is not None:
            out_dict["motion_mask"] = data.get("seq_mask")

        # ### print debuggint he ther the caption is lemmatized or not
        # if self.split != "train":
        #     print(f"Caption: {caption}")
        
        return out_dict


        # return caption, m_tokens, m_tokens_len, None, None, None, None, all_captions, tasks
        # return caption 0, m_tokens 1, m_tokens_len 2, None 3, None 4, None 5, None 6, all_captions 7, tasks 8

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
