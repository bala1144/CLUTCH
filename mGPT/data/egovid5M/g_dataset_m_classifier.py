import random
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .g_dataset_t2m import Text2MotionDataset
import torch

class MotionDataset_CLS(Text2MotionDataset):
    def __init__(
        self,
        w_vectorizer,
        augmentation_probs=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_vectorizer = w_vectorizer
        self.augmentation_probs = augmentation_probs

        print()

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"],  data["text"]

        if self.mean is not None:
            motion = (motion - self.mean) / self.std

        ### data text
        """
         data["text"] = list of dict(captions, tokems)
         [
         {'caption': 'A person stops for a moment and then runs to the left.', ''tokens': ['A/DET', 'person/NOUN', 'stop/VERB', 'for/ADP', 'a/DET', 'moment/NOUN', 'and/CCONJ', 'then/ADV', 'run/VERB', 'to/ADP', 'the/DET', 'left/NOUN']
         }
         ]
        """

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]


        ### switch the text caption randomly based on the group
        ### if swapped, then add the label to 1.0, which is a negative sample, 0.0 postive sample

        # Text
        max_text_len = 30
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
            "motion": motion,
            "length": m_length,
            "seq_name": self.name_list[idx],
            "intent_vec": data["intent_vec"] if data.get("intent_vec", None) is not None else np.zeros([29]) ,

            # adding the text
            "word_embs": word_embeddings, # idx 3
            "pos_ohot": pos_one_hots, # idx 4
            "text_len":sent_len, # idx 5
            "tokens": "_".join(tokens), # 6

            "text": caption,
            "all_captions": all_captions,
        }

        if data.get("seq_mask", None) is not None:
            out_dict["motion_mask"] = data.get("seq_mask")

        # augment
        if random.random() < self.augmentation_probs:
            out_dict = self.apply_augmentation(out_dict)
        
        return out_dict
        # return None, motion, length, None, None, None, None,


    def apply_augmentation(self, out_dict):
        aug = random.choice(["translation", "noise", "crop"])
        motion = out_dict["motion"]  # Assume motion is a NumPy array already

        if isinstance(motion, torch.Tensor):
            motion = motion.detach().cpu().numpy()

        motion_denorm = (motion * self.std) + self.mean
        motion_mask = out_dict.get("motion_mask", None)

        # print("\nRunning augment", aug)

        if aug == "translation":

            noise = np.random.randn(*motion.shape) * self.std + self.mean
            if motion.shape[-1] == 198:
                translation_mask = np.zeros_like(motion)
                translation_mask[:, 96:99] = 1.0
                translation_mask[:, 195:198] = 1.0

                motion_denorm = motion_denorm + noise * translation_mask
            else:
                raise ValueError("Input size not supported for translation augmentation")

        elif aug == "noise":
            noise = np.random.randn(*motion.shape) * self.std + self.mean
            motion_denorm = motion_denorm + noise

        elif aug == "crop":
            frames_to_crop = random.choice([1, 2, 3])
            motion_denorm_crop = motion_denorm[frames_to_crop:]
            pad = np.zeros((frames_to_crop, motion_denorm.shape[-1]))
            motion_denorm = np.concatenate([motion_denorm_crop, pad], axis=0)

            if motion_mask is not None:
                motion_mask_crop = motion_mask[frames_to_crop:]  # (T-N_c,) or (T-N_c, 1)
                pad_mask = np.zeros((frames_to_crop,) + motion_mask.shape[1:])  # Pad with same shape
                motion_mask = np.concatenate([motion_mask_crop, pad_mask], axis=0)
                out_dict["motion_mask"] = motion_mask

        if motion_mask is not None:
            motion_denorm = motion_denorm * motion_mask

        # Re-normalize
        motion_denorm = (motion_denorm - self.mean) / self.std

        # Update the output dictionary
        out_dict["motion"] = motion_denorm
        assert motion_denorm.shape == motion.shape # crop test
        return out_dict
        


    
