import random
import numpy as np
from .g_dataset_t2m import Text2MotionDataset


class Text2MotionDatasetEval(Text2MotionDataset):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_vectorizer = kwargs.get("w_vectorizer")
        self.split =  kwargs.get("split")



    def __getitem__(self, item):
        # Get text data
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

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
        
        # # Random crop
        # if self.unit_length < 10:
        #     coin2 = np.random.choice(["single", "single", "double"])
        # else:
        #     coin2 = "single"

        # if coin2 == "double":
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == "single":
        #     m_length = (m_length // self.unit_length) * self.unit_length

        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx + m_length]
        
        # Z Normalization
        if self.mean is not None:
            motion = (motion - self.mean) / self.std

        out_dict = {
            # text
            "text": caption,
            # motion
            "motion": motion,
            "length": m_length, # idx 2

            "word_embs": word_embeddings, # idx 3
            "pos_ohot": pos_one_hots, # idx 4
            "text_len":sent_len, # idx 5
            "tokens": "_".join(tokens), # 6

            "all_captions": all_captions, # idx 7
            # "tasks": # idx 8
            "seq_name":self.name_list[idx], # idx 9
            "sentence": data["sentence"]

        }

        ## 
        # # ## print debuggint he ther the caption is lemmatized or not
        # if self.split != "train":
        #     print(f"Caption: {caption}")
            
        if data.get("seq_mask", None) is not None:
            out_dict["motion_mask"] = data["seq_mask"]
        
        return out_dict
    
        # return caption, motion, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(tokens), all_captions, None, self.name_list[idx]
        # return caption, 
        # motion, 
        # m_length, 
        # word_embeddings,
        #  pos_one_hots,
        #  sent_len, 
        # "_".join(tokens),
        #  all_captions,
        #  None, 
        # self.name_list[idx]
    

        # caption = torch.zeros(20, 1)
        # m_tokens = torch.zeros(20, 1)
        # m_tokens_len = 20
        # all_captions = torch.zeros(3, 20, 1)
        # tasks= torch.zeros(1, 20)
        # return caption, m_tokens, m_tokens_len, None, None, None, None, all_captions, tasks

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

