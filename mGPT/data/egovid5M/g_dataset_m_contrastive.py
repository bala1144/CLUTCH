import random
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .g_dataset_t2m import Text2MotionDataset
from collections import defaultdict
import torch

from copy import deepcopy
def create_group_with_current_group(groups, group_to_remove):
    filtered = [s for s in groups if s != group_to_remove]
    return filtered

class MotionDataset_CONTRAST(Text2MotionDataset):
    def __init__(
        self,
        augmentation_probs=0.0,
        noise_strength=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_vectorizer = kwargs.get("w_vectorizer")
        self.split =  kwargs.get("split")
        self.shuffle_text = kwargs.get("shuffle_text", True)
        
        if not self.shuffle_text:
            print("\n******* START WARNING *********")
            print(f"Using shuffle_text for: {self.shuffle_text} for split: {self.split}")
            print("This is not recommended, for only evaluation and getting feature embeddings")
            print("******* END WARNING *********")

        if self.split == "train":
            self.group_to_indices = self.build_group_index()
            self.groups = list(self.group_to_indices.keys())
            self.confusion_matrix = {group: create_group_with_current_group(self.groups, group) for group in self.groups }
            self.negative_tracker = deepcopy(self.confusion_matrix)

        # noise strength
        self.augmentation_probs = augmentation_probs
        self.noise_strength = noise_strength

    def build_group_index(self):
        group_to_indices = defaultdict(list)
        for idx, seq in enumerate(self.name_list):
            data = self.data_dict[seq]
            action_group = data["action_group"]
            group_to_indices[action_group].append(seq)
        print("Build action groups : ", len(group_to_indices) )
        return group_to_indices

    def __len__(self):
        return len(self.name_list)
    

    def get_negative_sample(self, current_group):

        choosen_negative_group = self.negative_tracker[current_group].pop()
        negative_seq =  random.choice(self.group_to_indices[choosen_negative_group])

        # if empty fill the confusion matrix again
        if len(self.negative_tracker[current_group]) == 0:
             self.negative_tracker[current_group] = deepcopy(self.confusion_matrix[current_group])

        return negative_seq


    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"],  data["text"]

        ### switch the text caption randomly based on the group
        ### if swapped, then add the label to 1.0, which is a negative sample, 0.0 postive sample
        label = random.choice([0,1])
        if label == 1 and self.split == "train" and self.shuffle_text:
            negative_sample = self.get_negative_sample(data["action_group"])
            text_list = self.data_dict[negative_sample]["text"]
        else:
            label = 0

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

        # Text
        max_text_len = 100
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            print("Text is too long, cropping to max length:", max_text_len )
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        ## print debuggint he ther the caption is lemmatized or not
        if self.split != "train":
            print(f"Caption: {caption}")
            
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
            "intent_vec": data["intent_vec"] if data.get("intent_vec", None) is not None else np.zeros([1, 29]) ,

            # adding the text
            "word_embs": word_embeddings, # idx 3
            "pos_ohot": pos_one_hots, # idx 4
            "text_len":sent_len, # idx 5
            "tokens": "_".join(tokens), # 6

            "text": caption,
            "all_captions": all_captions,
            "contrastive_label": np.array([label], dtype=float)
        }

        if data.get("seq_mask", None) is not None:
            out_dict["motion_mask"] = data.get("seq_mask")

        # augment
        if random.random() < self.augmentation_probs and self.split == "train":
            out_dict = self.apply_augmentation(out_dict)
            
        return out_dict
    
    def apply_augmentation(self, out_dict):
        aug = random.choice(["translation", "noise", "crop"])
        motion = out_dict["motion"]

        if isinstance(motion, torch.Tensor):
            motion = motion.detach().cpu().numpy()

        original_shape = motion.shape  # Save shape for consistency
        motion_denorm = (motion * self.std) + self.mean
        motion_mask = out_dict.get("motion_mask", None)
        noise = (np.random.rand(*motion.shape) * 2 - 1) * self.std * self.noise_strength

        if aug == "translation":
            if motion.shape[-1] == 198:
                translation_mask = np.zeros_like(motion)
                translation_mask[:, 96:99] = 1.0
                translation_mask[:, 195:198] = 1.0
                motion_denorm = motion_denorm + noise * translation_mask
            else:
                raise ValueError("Input size not supported for translation augmentation")
            
        elif aug == "noise":
            motion_denorm = motion_denorm + noise

        elif aug == "crop":
            frames_to_crop = random.choice([1, 2, 3])
            motion_denorm_crop = motion_denorm[frames_to_crop:]
            pad = np.zeros((frames_to_crop, motion.shape[-1]))
            motion_denorm = np.concatenate([motion_denorm_crop, pad], axis=0)

            if motion_mask is not None:
                motion_mask = np.array(motion_mask)  # Ensure it's a NumPy array
                motion_mask_crop = motion_mask[frames_to_crop:]
                if motion_mask.ndim == 1:
                    pad_mask = np.zeros((frames_to_crop,))
                else:
                    pad_mask = np.zeros((frames_to_crop,) + motion_mask.shape[1:])
                motion_mask = np.concatenate([motion_mask_crop, pad_mask], axis=0)
                out_dict["motion_mask"] = motion_mask

        if motion_mask is not None:
            motion_denorm = motion_denorm * motion_mask

        # Re-normalize
        motion_denorm = (motion_denorm - self.mean) / self.std

        # Final safety check
        assert motion_denorm.shape == original_shape, f"Shape mismatch: {motion_denorm.shape} != {original_shape}"

        out_dict["motion"] = motion_denorm
        return out_dict


    
    # def apply_augmentation(self, out_dict):
    #     aug = random.choice(["translation", "noise", "crop"])
    #     motion = out_dict["motion"]  # Assume motion is a NumPy array already

    #     if isinstance(motion, torch.Tensor):
    #         motion = motion.detach().cpu().numpy()

    #     motion_denorm = (motion * self.std) + self.mean
    #     motion_mask = out_dict.get("motion_mask", None)


    #     if aug == "translation":

    #         # noise = np.random.randn(*motion.shape) * self.std + self.mean
    #         noise = (np.random.rand(*motion.shape) * 2 - 1) * self.std
    #         if motion.shape[-1] == 198:
    #             translation_mask = np.zeros_like(motion)
    #             translation_mask[:, 96:99] = 1.0
    #             translation_mask[:, 195:198] = 1.0

    #             motion_denorm = motion_denorm + noise * translation_mask
    #         else:
    #             raise ValueError("Input size not supported for translation augmentation")

    #     elif aug == "noise":
    #         # noise = np.random.randn(*motion.shape) * self.std + self.mean
    #         noise = (np.random.rand(*motion.shape) * 2 - 1) * self.std
    #         motion_denorm = motion_denorm + noise

    #     elif aug == "crop":
    #         frames_to_crop = random.choice([1, 2, 3])
    #         motion_denorm_crop = motion_denorm[frames_to_crop:]
    #         pad = np.zeros((frames_to_crop, motion_denorm.shape[-1]))
    #         motion_denorm = np.concatenate([motion_denorm_crop, pad], axis=0)

    #         if motion_mask is not None:
    #             motion_mask_crop = motion_mask[frames_to_crop:]  # (T-N_c,) or (T-N_c, 1)
    #             pad_mask = np.zeros((frames_to_crop,) + motion_mask.shape[1:])  # Pad with same shape
    #             motion_mask = np.concatenate([motion_mask_crop, pad_mask], axis=0)
    #             out_dict["motion_mask"] = motion_mask

    #     if motion_mask is not None:
    #         motion_denorm = motion_denorm * motion_mask

    #     # Re-normalize
    #     motion_denorm = (motion_denorm - self.mean) / self.std

    #     # Update the output dictionary
    #     out_dict["motion"] = motion_denorm
    #     assert motion_denorm.shape == motion.shape # crop test
    #     return out_dict
        
