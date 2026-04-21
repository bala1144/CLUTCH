import os
import sys
import glob
import joblib
import numpy as np
import torch
import smplx as smplx
import trimesh
from torch.utils import data
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from mGPT.hand.utils.dict_to_struct import dict_to_struct
import json 
from mGPT.hand.utils.torch_rotation import *
import spacy
import h5py


def load_spacy_pipeline():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


def load_mesh_vertices(mesh_path):
    mesh = trimesh.load(mesh_path, process=False)
    return mesh.vertices

class LoadData(data.Dataset):
    def __init__(self, 
                 split='train',
                 fps=30,
                 max_frames=160,
                 down_sample=False,
                 batch_size=1,
                 dict_file_to_load="mano_smoothened.npy",
                 **kwargs):
        super().__init__()
        """
        if overt
        """

        from mGPT.hand.body_models.mano_xx import MANO_doubleX
        self.mano_doubleX = MANO_doubleX(
            batch_size=20,
            use_pca=True,
            num_pca_comps=24,
            flat_hand_mean=True,
            device=device,
        )

        self.text_augumentation = False
        # restrian the length of motion and text
        self.max_length = 20
        self.max_motion_length = max_frames
        self.min_motion_length = 60
        self.unit_length = 4

        self.hand2idx = { "right": 1, "left": 0}
        args = dict_to_struct(**kwargs)
        self.fps = fps
        self.max_frames = max_frames
        self.down_sample = down_sample
        self.dict_file_to_load = dict_file_to_load
        self.batch_size = batch_size
        self.sampling_type = kwargs.get("sampling_type", "max_sampling")
        self.down_sample_factor = kwargs.get("down_sample_factor")
        self.use_pca_for_hand_pose = kwargs.get("use_pca_for_hand_pose", False)
        # max_frames: 144 # should be a multiple of the compression factor
        # sampling_type: "uniform_sampling" # max_sampling, uniform_sampling

        ### dataset path
        dataset_dir = os.path.join(args.dataset_dir)
        self.dataset_dir = dataset_dir
        self.data_root =  self.dataset_dir
        self.hawor_recon_base_path = os.path.join(dataset_dir, "hawor_recon")
        self.vlm_annotation_path = self.hawor_recon_base_path.replace("hawor_recon", "VLM_annotations")
        
        self.ann_file_to_load = kwargs.get("annotation_file_to_load", "vila_claude_summary.json")
        self.accumulated_ann_file_to_load = kwargs.get("accumulated_annotation_file_to_load", None)
        self.loaded_accumulated_ann = None
        self.caption_no_lemmatization = kwargs.get("caption_no_lemmatization", True)

        self.shift_pos_to_origin = args.shift_pos_to_origin
        self.nlp = load_spacy_pipeline()

        #### add datasets
        print('*************************')
        print(f"Loading split : {split}")
        print('*************************\n')

        self.load_datasets(args, split, kwargs)


    def load_datasets(self, args, split, kwargs):

        self.cache_version = int(kwargs.get("preprocessed_cache_version", 1))
        dataset_name = kwargs.get(
            "dataset"
        )
        cache_template = kwargs.get("preprocessed_cache_file", None)
        if cache_template:
            try:
                resolved = cache_template.format(dataset=dataset_name, split=split, version=self.cache_version)
            except KeyError as exc:
                raise ValueError(
                    f"Could not format preprocessed_cache_file '{cache_template}'. "
                    "Supported placeholders are {dataset} and {split}."
                ) from exc
            self.cache_file = resolved
        else:
            self.cache_file = None

        if self.cache_file and not os.path.isabs(self.cache_file):
            self.cache_file = os.path.join(args.dataset_dir, self.cache_file)

        self.use_preprocessed_cache = kwargs.get("use_preprocessed_cache", True)
        self.regenerate_cache = kwargs.get("regenerate_preprocessed_cache", False)
        self.dump_preprocessed_cache = kwargs.get("dump_preprocessed_cache", False)

        if (self.cache_file and self.use_preprocessed_cache
                and os.path.exists(self.cache_file)
                and not self.regenerate_cache):
            print(f"Found preprocessed cache at {self.cache_file}, loading it instead of raw files.")
            self._load_preprocessed_cache(self.cache_file, split, dataset_name)
            self.args = args
            self.reset_max_len(self.max_length)
            return
        else:
            print(f"No preprocessed cache found at {self.cache_file}, loading raw files.")
            return self.load_data_from_path(args, split, kwargs)
        

    def load_data_from_path(self, args, split, kwargs):

        dataset_name = kwargs.get(
            "dataset"
        )

        if kwargs.get("egovid5M_to_load", True):
            data_dict, new_name_list, length_list = self.load_egovid5M(args, split, kwargs)
            print(f"Loaded egovid5M {split} seqs", len(data_dict))
        else:
            data_dict, new_name_list, length_list = {}, [], []
            print(f"Not loading egovid5M")

        if kwargs.get("GRAB_to_load", False):
            data_dict_, new_name_list_, length_list_ = self.load_grab(args.Grab_cfg, split) 
            # data_dict, new_name_list, length_list = self.load_grab(args.Grab_cfg, split) 
            length_list = length_list + length_list_
            new_name_list = new_name_list + new_name_list_
            for seq in new_name_list_:
                data_dict[seq] = data_dict_[seq]
            print(f"Loaded GRAB {split} seqs", len(data_dict_))
        
        if kwargs.get("GRAB_aug_to_load", False):
            data_dict_, new_name_list_, length_list_ = self.load_grab(args.Grab_aug_cfg, split) 
            # data_dict, new_name_list, length_list = self.load_grab(args.Grab_cfg, split) 
            length_list = length_list + length_list_
            new_name_list = new_name_list + new_name_list_
            for seq in new_name_list_:
                data_dict[seq] = data_dict_[seq]

            print(f"Loaded GRAB_aug {split} seqs", len(data_dict_))
        
                ### accumulate the dict
        self.data_dict = data_dict
        assert len(self.data_dict) > 0, "No samples were loaded"

        ### created a sorted len list,
        ### But why ?
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        ## setup the processing tools
        self.num_samples = len(self.data_dict)
        self.args = args

        self.length_arr = np.array(length_list)
        self.name_list = name_list
        self.nfeats = self.data_dict[name_list[0]]['motion'].shape[1]
        self.reset_max_len(self.max_length)

        if self.cache_file and self.dump_preprocessed_cache:
            print(f"Dumping preprocessed cache to {self.cache_file}")
            self._dump_preprocessed_cache(self.cache_file, split, dataset_name)


    def load_egovid5M(self, args, split, kwargs):

        ### get the dataset type, train, val or test
        ### load the index dict
        ### todo this could be also csv file, this is what typically used a
        if  ".json" in args.index_file:
            index_file = os.path.join(self.dataset_dir, args.index_file)
            all_seqs = json.load(open(index_file, "r")) # list of dicts

        if self.accumulated_ann_file_to_load is not None:
            ann_file = os.path.join(self.dataset_dir, self.accumulated_ann_file_to_load)
            if os.path.exists(ann_file):
                print(f"Loading accumulated annotation file {ann_file}")
                self.loaded_accumulated_ann = json.load(open(ann_file, "r"))
            else:
                print(f"Accumulated annotation file {ann_file} not found, using default annotations")

        ### handling the newest version of the dataset
        split_to_load = "train" if args.seqs_split.get("overfit", False) else split
        if isinstance(all_seqs, dict):
            all_seqs = all_seqs[split_to_load]

        seq_for_set = args.seqs_split[split_to_load]
        print(f"Split to load for set {split}: {split_to_load}")
        if type(seq_for_set) is not str:
            range = [int(x) for x in seq_for_set]
            assert len(range) <= len(all_seqs), "seq to be loaded > num of available seqs"
            sequence_to_be_loaded = all_seqs[range[0]:range[1]]
            self.sequence_to_be_loaded = sequence_to_be_loaded
        
        if len(sequence_to_be_loaded) == 0:
            print("No EGOVID5M Seq to load")
            return {}, [], []
        

        self.egovid5m_mapping = self.load_combo_group_dict(kwargs)

        total_num_seqs = len(all_seqs)
        to_load = len(self.sequence_to_be_loaded)
        loaded_data_dict = {}
        new_name_list = []
        length_list = []
        print(f"Total seqs {total_num_seqs}, seq to load {to_load}")
        for data_count, seq_dict in enumerate(tqdm(self.sequence_to_be_loaded, f"loading {split}")):
            loaded = self.load_egovid5M_seq(seq_dict) 

            if loaded is None:
                print(f"Skipping {seq_dict}")
                continue

            seq_name = loaded["seq_name"]
            if loaded["length"] < self.max_frames :
                length__ = loaded["length"]
                print(f"Skipping {seq_name} {length__} < {self.max_frames}")
                continue

            # print(seq_name)
            loaded_data_dict[seq_name] = loaded
            new_name_list.append(loaded["seq_name"])
            length_list.append(loaded['motion'].shape[0])

        return loaded_data_dict, new_name_list, length_list

    def load_egovid5M_seq(self, seq_dict):

        #### from the hawor file
        if isinstance(seq_dict, dict):
            hawor_file = seq_dict["hawor_recon_file"]
            recon_file = os.path.join(os.path.dirname(hawor_file), f"{self.dict_file_to_load}")
            seq_name = recon_file.split("/")[-2]
        elif isinstance(seq_dict, str):
            recon_file = os.path.join(seq_dict, f"{self.dict_file_to_load}")
            seq_name = recon_file.split("/")[-2]
        
        try:
            params_dict = np.load(recon_file, allow_pickle=True).item()
            loaded = {
                "seq_file": recon_file,
                "seq_name": seq_name
            }
        except:
            print(f"Error while loading {seq_name}")
            return None

        loaded = self.add_hand_params(loaded, params_dict)    
        loaded = self.handle_seq_length(loaded)

        ### load the text prompt from the file
        ### run it through clip and output the embedding
        if self.loaded_accumulated_ann is not None:
            vlm_file = os.path.dirname(recon_file)
            annotations = self.loaded_accumulated_ann.get(vlm_file)
            anns_to_load = [anns_to_load] if type(self.ann_file_to_load) == str else self.ann_file_to_load            
            summary = []
            for ann_to_load in anns_to_load:
                anns = annotations.get(ann_to_load.split("/")[-1].split(".")[0], None)
                summary.append(anns)
            loaded['sentence'] = summary
        elif type(self.ann_file_to_load) == str: # legacy code
            if "fastest_1000_recon_exp" in self.dataset_dir:
                f_level = recon_file.split("/")[-4]
                s_level = recon_file.split("/")[-3]
                vlm_file = os.path.join(self.vlm_annotation_path,  f_level, s_level, seq_name, self.ann_file_to_load)
                annotations = json.load(open(vlm_file, "r"))
            elif "clean_10K_recon" in self.dataset_dir or  "clean_10K_10sec" in self.dataset_dir: # legacy code
                f_level = recon_file.split("/")[-3]
                seq_name = recon_file.split("/")[-2]
                vlm_file = os.path.join(self.vlm_annotation_path,  f_level, seq_name, self.ann_file_to_load)
                annotations = json.load(open(vlm_file, "r"))
            else:
                raise("Enter a valid dataset")
            
            summary = annotations["summary"]
            loaded['sentence'] = [summary]
        ### list of the model
        elif "list" in str(type(self.ann_file_to_load)):
            summary = []
            f_level = recon_file.split("/")[-3]
            seq_name = recon_file.split("/")[-2]
            for ann_file in self.ann_file_to_load:
                vlm_file = os.path.join(self.vlm_annotation_path,  f_level, seq_name, ann_file)
                annotations = json.load(open(vlm_file, "r"))
                summary.append(annotations["summary"])
            loaded['sentence'] = summary
        else:
            raise("Error")

        
        # ### previosuly only choose the coarse summary
        # ## modified for Diffusion training model
        # if type(summary) == list:
        #     summary_embed = []
        #     for summ_ in summary:
        #         summary_embed.append(self.model_clip.encode_text(clip.tokenize([summ_]).to(device)).cpu().detach().numpy())
        # else:
        #     summary_embed = [self.model_clip.encode_text(clip.tokenize([summary]).to(device)).cpu().detach().numpy()]

        summary_embed = np.zeros([1, 64])
        loaded['sentence_vec'] =  np.zeros_like(summary_embed)
        loaded['intent_embedding'] = np.zeros_like(summary_embed)
        loaded['seq_name_embed'] = np.zeros_like(summary_embed)
        ### so far hasn't been used
        # loaded["detailed_summary"] = annotations["hand_details"]

        ### extra thingy added to make the motion decoder work
        loaded["motion"] = self.batch_to_mano_pose(loaded)
        loaded["length"] = loaded["motion"].shape[0]
        loaded["text"] =  self.process_sentence(loaded)

        if self.egovid5m_mapping is not None:
            seq="/".join(recon_file.split("/")[:-1])
            action_group = self.egovid5m_mapping["seq_2_group"][seq]
            one_hot = np.zeros([self.num_egovid_len])
            action_group_idx = self.all_egovid5m_group_mapping[action_group]
            one_hot[action_group_idx] = 1
            loaded["action_group"] = action_group
            loaded["intent_vec"] = one_hot

        return loaded
    
    def load_combo_group_dict(self, kwargs):
        combo_group_file = kwargs.get("seq_2_action_mapping", None)
        egovidr5m_action_mapping = None
        if combo_group_file is not None:
            combo_group_file = os.path.join(self.dataset_dir, combo_group_file)
            egovidr5m_action_mapping = json.load(open(combo_group_file, "r"))
            self.num_egovid_len = len(egovidr5m_action_mapping["group_2_seq"])
            self.all_egovid5m_group_mapping = {  k:idx for idx, k in enumerate(list(egovidr5m_action_mapping["group_2_seq"].keys())) }

        return egovidr5m_action_mapping

    def load_grab(self, args, split):

        current_dirpath = os.path.join(args.dataset_dir, split)
        all_seqs = glob.glob(os.path.join(current_dirpath, '*.npz'), recursive = True) 
        total_num_seqs = len(all_seqs)

        if total_num_seqs > 6_000: # for hanfling agu
            all_seqs = all_seqs[::2]
            total_num_seqs = len(all_seqs)


        seq_for_set = args.seqs_split[split]
        if type(seq_for_set) is not str:
            range = [int(x) for x in seq_for_set]
            sequence_to_be_loaded = all_seqs[range[0]:range[1]]

        to_load = len(sequence_to_be_loaded)
        loaded_data_dict = {}
        new_name_list = []
        length_list = []
        print(f"Total seqs {total_num_seqs}, seq to load {to_load}")
        for data_count, seq_file in enumerate(tqdm(sequence_to_be_loaded)):
            seq_name = seq_file.split('/')[-1].split('.')[0]
            # print(seq_name)
            loaded = self.load_grab_seq(seq_file) 


            if loaded is None:
                print(f"Skipping {seq_file}")
                continue


            if loaded["length"] < self.max_frames :
                length__ = loaded["length"]
                print(f"Skipping {seq_name} {length__} < {self.max_frames}")
                continue

            loaded["seq_name"] = seq_name
            loaded_data_dict[seq_name] = loaded
            new_name_list.append(loaded["seq_name"])
            length_list.append(loaded['motion'].shape[0])

            
        return loaded_data_dict, new_name_list, length_list

    def load_grab_seq(self, seq_file):
        
        param_dict = torch.load(seq_file, weights_only=False)

        to_copy = ["seq_name","mano_params","sentence", "intent_embedding", "sentence_vec", "intent" ]
        loaded = { k: param_dict[k] for k in to_copy }
        loaded['seq_name_embed'] = np.zeros_like(loaded["intent_embedding"])

        ## modified for Diffusion training model
        if not isinstance(loaded['sentence_vec'], list):
            loaded['sentence_vec'] = [loaded['sentence_vec']]

        loaded["action_group"] = param_dict["intent"] + "_" + param_dict["obj_name"]

        if param_dict.get("intent_vec", None) is not None:
            loaded["intent_vec"] = np.array(param_dict["intent_vec"])

        loaded['rhand_global_orientaion'] = loaded['mano_params']['rhand_global_orientaion'].reshape(-1, 1, 6)
        loaded['rhand_transl'] = loaded['mano_params']['rhand_transl'].reshape(-1, 3)

        loaded['lhand_global_orientaion'] = loaded['mano_params']['lhand_global_orientaion'].reshape(-1, 1, 6)
        loaded['lhand_transl'] = loaded['mano_params']['lhand_transl'].reshape(-1, 3)

        # T = loaded['rhand_transl'].shape[0]
        # loaded['rhand_pose'] = loaded['mano_params']['rhand_pose_6D'].reshape(T, -1)
        # loaded['lhand_pose'] = loaded['mano_params']['lhand_pose_6D'].reshape(T, -1)

        loaded['lhand_pose'] =  self.mano_doubleX.convert_pca_to_jt_rot_in_6D(loaded["mano_params"]['lhand_pose'], right_hand=False) # T x 15 x 6
        loaded['rhand_pose'] =  self.mano_doubleX.convert_pca_to_jt_rot_in_6D(loaded["mano_params"]['rhand_pose']) # # T x 15 x 6
        

        ## downsample for faster training
        input_T = loaded['mano_params']['rhand_global_orientaion'].shape[0]

        if self.sampling_type == "max_sampling" and self.down_sample and self.max_frames < input_T:
            frames_per_pose_input = input_T // self.max_frames 
            sample_idx = np.arange(0, input_T, frames_per_pose_input)[:self.max_frames]
            for k in ["rhand_global_orientaion", "rhand_transl", "rhand_pose", "lhand_global_orientaion", "lhand_transl", "lhand_pose"]:
                loaded[k] = loaded[k][sample_idx].to("cpu")
        elif self.sampling_type == "uniform_sampling" and self.down_sample:
            sample_idx = np.arange(0, input_T, self.down_sample_factor)[:self.max_frames]
            for k in ["rhand_global_orientaion", "rhand_transl", "rhand_pose", "lhand_global_orientaion", "lhand_transl", "lhand_pose"]:
                loaded[k] = loaded[k][sample_idx].to("cpu")
        else:
            return None
        
        for k in ["rhand_global_orientaion", "rhand_transl", "rhand_pose", "lhand_global_orientaion", "lhand_transl", "lhand_pose"]:
                loaded[k] = loaded[k].to("cpu")
                        
        if self.shift_pos_to_origin:
            '''translate pose to origin'''
            mean_init_trans = torch.mean(torch.stack([loaded['rhand_transl'][0, :], loaded['lhand_transl'][0, :]]), dim=0).reshape(-1, 3)
            loaded['lhand_transl']  =  loaded['lhand_transl'] - mean_init_trans
            loaded['rhand_transl']  =  loaded['rhand_transl'] - mean_init_trans

        
        loaded = self.handle_seq_length(loaded)
        ### extra thingy added to make the motion decoder work
        loaded["motion"] = self.batch_to_mano_pose(loaded)
        loaded["length"] = loaded["motion"].shape[0]
        loaded["text"] =  self.process_sentence(loaded)
        return loaded


    def __len__(self):
        return self.num_samples

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def process_sentence(self, loaded):
    
        if not self.text_augumentation:
            all_sentences = loaded["sentence"]
            if type(all_sentences) == str:
                all_sentences = [all_sentences]

            text_dict_list = []
            for sentence in all_sentences:
                doc = self.nlp(sentence)
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
                    
                caption = ' '.join(word_list) # A person stops for a moment and then runs to the left.

                text_dict = {
                    'caption': caption if not self.caption_no_lemmatization else sentence,
                    'tokens': [f"{x}/{y}" for x, y in zip(word_list, pos_list)]
                }
                # print("text_dict:", text_dict["caption"])
                text_dict_list.append(text_dict)
        else:
            
            activeintent = loaded["intent"]
            object = loaded["obj_name"]

            intent_dict = {
                "<singularactiveintent>" : activeintent,
                "<pluralactiveintent>" : plural_dict[activeintent],
                "<singularpassiveintent>": passive_dict[activeintent],

            }
            
            possible_sentences = [
                "The person <singularactiveintent> the <object>.", ## original
                "The person <pluralactiveintent> the <object>.", ## original
                "The person <singularactiveintent> the <object>.", ## original
                "The person <pluralactiveintent> the <object>.", ## original

                "The person chooses to <singularactiveintent> the <object>.",
                "The person takes action to <singularactiveintent> the <object>.",
                "The person's goal is to <singularactiveintent> the <object>.",
                "The person proceeds to <singularactiveintent> the <object>.",

                # Passive Voice
                "The <object> is <singularpassiveintent> the person.",
                "An effort is made by the person to <singularactiveintent> the <object>.",
                "The goal of the person is to have the <singularactiveintent> the <object>.",
                # "The <object> is being <singularpassiveintent> the person.",

            ]
            
            text_dict_list = []
            curated_sentence = []
            for sentence in possible_sentences:
                sentence = sentence.replace("<object>", object)
                for key, value in intent_dict.items():
                    sentence = sentence.replace(key, value)

                doc = self.nlp(sentence)
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
                    
                caption = ' '.join(word_list) # A person stops for a moment and then runs to the left.
                curated_sentence.append(sentence)
                text_dict = {
                    'caption': sentence,
                    'tokens': [f"{x}/{y}" for x, y in zip(word_list, pos_list)]
                }

                text_dict_list.append(text_dict)
            
        return text_dict_list
    
    def add_hand_params(self, items, params_dict):
        
        pred_trans=params_dict["pred_trans"][:,1:-1].detach().cpu() # hs x 123 x 3
        pred_rot=params_dict["pred_rot"][:,1:-1].detach().cpu() # hs x 123 x 3
        pred_hand_pose=params_dict["pred_hand_pose"][:,1:-1].detach().cpu() # torch.Size([2, 123, 15, 3])
        
        ##@ downsample for faster training
        input_T = pred_trans.shape[1]
        if self.sampling_type == "max_sampling" and self.down_sample and self.max_frames < input_T:
            frames_per_pose_input = input_T // self.max_frames 
            sample_idx = np.arange(0, input_T, frames_per_pose_input)[:self.max_frames]

            pred_trans = pred_trans[:, sample_idx, :]
            pred_rot = pred_rot[:, sample_idx, :]
            pred_hand_pose = pred_hand_pose[:, sample_idx, :]

        elif self.sampling_type == "uniform_sampling" and self.down_sample:
            sample_idx = np.arange(0, input_T, self.down_sample_factor)[:self.max_frames]
            pred_trans = pred_trans[:, sample_idx, :]
            pred_rot = pred_rot[:, sample_idx, :]
            pred_hand_pose = pred_hand_pose[:, sample_idx, :]

        ## convert to 6D
        pred_rot_6d = axis_angle_to_6D(pred_rot).detach().cpu()
        pred_hand_pose_6D = axis_angle_to_6D(pred_hand_pose).detach().cpu() # torch.Size([2, 123, 15, 3])

        T = pred_trans.shape[1]
        items['rhand_global_orientaion'] = pred_rot_6d[1].reshape(T, 6)
        items['rhand_transl'] = pred_trans[1].reshape(T, 3)
        items['rhand_pose'] = pred_hand_pose_6D[1].reshape(T, 15, 6)

        items['lhand_global_orientaion'] = pred_rot_6d[0].reshape(T, 6)
        items['lhand_transl'] = pred_trans[0].reshape(T, 3)
        items['lhand_pose'] = pred_hand_pose_6D[0].reshape(T, 15, 6)


        ### shift to the origin for the params
        if  self.shift_pos_to_origin:      
            '''translate pose to origin'''
            # mean_init_trans = torch.mean([items['rhand_transl'][0, :], items['lhand_transl'][0, :]], dim=-1).reshape(-1, 3)
            mean_init_trans = torch.mean(torch.stack([items['rhand_transl'][0, :], items['lhand_transl'][0, :]]), dim=0).reshape(-1, 3)
            items['lhand_transl']  =  items['lhand_transl'] - mean_init_trans
            items['rhand_transl']  =  items['rhand_transl'] - mean_init_trans
       
        return items
        
    def batch_to_mano_pose(self, batch):


        T = batch["rhand_global_orientaion"].shape[0]
        if self.use_pca_for_hand_pose: # set number of PCA to 24, used for the GRAB dataset
            batch["rhand_pose"]  = self.mano_doubleX.convert_jt_rot_in_6D_to_pca(batch["rhand_pose"].reshape( T, -1).to("cuda"), right_hand=True).cpu()
            batch["lhand_pose"]  = self.mano_doubleX.convert_jt_rot_in_6D_to_pca(batch["lhand_pose"].reshape( T, -1).to("cuda"), right_hand=False).cpu()

        motion = torch.cat(
                [
                batch["rhand_global_orientaion"].reshape(T, -1),# Bs x T x 6 
                batch["rhand_pose"].reshape( T, -1),  # Bs, T x 90 (15J x 6)
                batch["rhand_transl"].reshape(T, -1),# Bs x T x 3
                
                batch["lhand_global_orientaion"].reshape(T, -1),  # Bs x T x 6
                batch[ "lhand_pose"].reshape(T, -1), # Bs, T x 90 (15J x 6)
                batch["lhand_transl"].reshape(T, -1) # Bs x T x 3
                ],
                dim=-1
            )
        # T x 198
        return motion.clone().detach().cpu().numpy()

    def handle_seq_length(self, items):
        
        # print(idx, T, self.max_frames)
        T = items['rhand_pose'].shape[0]
        if T == self.max_frames:  
            seq_mask = torch.ones(T, 1)
            items['seq_mask']  = seq_mask.cpu().numpy()

        elif T < self.max_frames:

            if self.sampling_type == "max_sampling":
                pass

            elif self.sampling_type == "uniform_sampling":

                buff = self.max_frames - T

                rhand_global_orientaion = torch.zeros(buff, 6)
                rhand_pose = torch.zeros(buff, 15, 6)
                rhand_transl = torch.zeros(buff, 3)

                lhand_global_orientaion = torch.zeros(buff, 6)
                lhand_pose = torch.zeros(buff, 15, 6)
                lhand_transl = torch.zeros(buff, 3)

                items['rhand_global_orientaion'] =  torch.concatenate(( items['rhand_global_orientaion'].reshape(T, 6), rhand_global_orientaion), dim=0)
                items['rhand_pose'] =  torch.concatenate(( items['rhand_pose'], rhand_pose), dim=0)
                items['rhand_transl'] =  torch.concatenate(( items['rhand_transl'], rhand_transl), dim=0)
                
                
                items['lhand_global_orientaion'] =  torch.concatenate(( items['lhand_global_orientaion'].reshape(T, 6), lhand_global_orientaion), dim=0)
                items['lhand_pose'] =  torch.concatenate(( items['lhand_pose'], lhand_pose), dim =0)
                items['lhand_transl'] =  torch.concatenate(( items['lhand_transl'], lhand_transl), dim=0)

                seq_mask = torch.ones(T, 1)
                buff_mask = torch.zeros(buff, 1)
                seq_mask = torch.cat((seq_mask, buff_mask), dim=0)
                items['seq_mask']  = seq_mask.cpu().numpy()
        
        elif T > self.max_frames:

            # raise(f"Seq len {T} > max frames:{self.max_frames}")
            items['rhand_global_orientaion'] = items['rhand_global_orientaion'][:self.max_frames]
            items['rhand_pose'] = items['rhand_pose'][:self.max_frames]
            items['rhand_transl'] = items['rhand_transl'][:self.max_frames]
            
            items['lhand_global_orientaion'] = items['lhand_global_orientaion'][:self.max_frames]
            items['lhand_pose'] = items['lhand_pose'][:self.max_frames]
            items['lhand_transl'] = items['lhand_transl'][:self.max_frames]

            seq_mask = torch.ones(self.max_frames, 1)
            items['seq_mask']  = seq_mask.cpu().numpy()

        return items
    
    def __getitem__(self, idx):
        items = self.data_dict[idx]

        T = items['rhand_pose'].shape[0]
        # print(idx, T, self.max_frames)
        if T < self.max_frames:
            buff = self.max_frames - T

            rhand_global_orientaion = torch.zeros(buff, 1, 6)
            rhand_pose = torch.zeros(buff, 15, 6)
            rhand_transl = torch.zeros(buff, 3)

            lhand_global_orientaion = torch.zeros(buff, 1, 6)
            lhand_pose = torch.zeros(buff, 15, 6)
            lhand_transl = torch.zeros(buff, 3)
            items['rhand_global_orientaion'] =  torch.concatenate(( items['rhand_global_orientaion'], rhand_global_orientaion), dim=0)
            items['rhand_pose'] =  torch.concatenate(( items['rhand_pose'], rhand_pose), dim=0)
            items['rhand_transl'] =  torch.concatenate(( items['rhand_transl'], rhand_transl), dim=0)
            
            items['lhand_global_orientaion'] =  torch.concatenate(( items['lhand_global_orientaion'], lhand_global_orientaion), dim=0)
            items['lhand_pose'] =  torch.concatenate(( items['lhand_pose'], lhand_pose), dim =0)
            items['lhand_transl'] =  torch.concatenate(( items['lhand_transl'], lhand_transl), dim=0)

            seq_mask = torch.ones(T, 1)
            buff_mask = torch.zeros(buff, 1)
            seq_mask = torch.cat((seq_mask, buff_mask), dim=0)
            items['seq_mask']  = seq_mask
        
        elif T > self.max_frames:

            items['rhand_global_orientaion'] = items['rhand_global_orientaion'][:self.max_frames]
            items['rhand_pose'] = items['rhand_pose'][:self.max_frames]
            items['rhand_transl'] = items['rhand_transl'][:self.max_frames]
            
            items['lhand_global_orientaion'] = items['lhand_global_orientaion'][:self.max_frames]
            items['lhand_pose'] = items['lhand_pose'][:self.max_frames]
            items['lhand_transl'] = items['lhand_transl'][:self.max_frames]

            items['seq_len'] = self.max_frames

        return items
    
    def _dump_preprocessed_cache(self, cache_path, split, dataset_name):
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        meta = {
            "version": int(self.cache_version),
            "dataset": dataset_name,
            "dataset_dir": self.dataset_dir,
            "split": split,
            "dict_file_to_load": self.dict_file_to_load,
            "sampling_type": self.sampling_type,
            "max_frames": int(self.max_frames),
            "max_length": int(self.max_length),
            "down_sample": bool(self.down_sample),
            "use_pca_for_hand_pose": bool(self.use_pca_for_hand_pose),
        }

        with h5py.File(cache_path, "a") as handle:
            datasets_group = handle.require_group("datasets")
            dataset_group = datasets_group.require_group(dataset_name)
            if split in dataset_group:
                del dataset_group[split]
            split_group = dataset_group.create_group(split)
            split_group.attrs["meta"] = json.dumps(meta)

            split_group.create_dataset("name_list", data=np.array(self.name_list, dtype="S"))
            split_group.create_dataset("length_arr", data=self.length_arr)

            sequences_group = split_group.create_group("sequences")
            for seq_name in self.name_list:
                seq_group = sequences_group.create_group(seq_name)
                payload = self.data_dict[seq_name]
                for key, value in payload.items():
                    self._write_hdf5_value(seq_group, key, value)

    def _load_preprocessed_cache(self, cache_path, split, dataset_name):
        with h5py.File(cache_path, "r") as handle:
            datasets_group = handle.get("datasets")
            if datasets_group is None or dataset_name not in datasets_group:
                raise ValueError(f"Dataset '{dataset_name}' not present in cache {cache_path}")

            dataset_group = datasets_group[dataset_name]
            if split not in dataset_group:
                raise ValueError(f"Split '{split}' not found in cache {cache_path}")

            split_group = dataset_group[split]
            meta_json = split_group.attrs.get("meta")
            if meta_json is None:
                raise ValueError(f"Cache {cache_path} missing metadata for split '{split}'")

            meta = json.loads(meta_json)
            if meta.get("version") != int(self.cache_version):
                raise ValueError(
                    f"Cache version mismatch: expected {self.cache_version}, "
                    f"found {meta.get('version')} at {cache_path}"
                )
            if meta.get("dict_file_to_load") != self.dict_file_to_load:
                raise ValueError(
                    "Cached data was generated with a different dict_file_to_load. "
                    "Set regenerate_preprocessed_cache=True to rebuild."
                )

            raw_names = split_group["name_list"][...]
            name_list = [
                name.decode("utf-8") if isinstance(name, (bytes, np.bytes_)) else str(name)
                for name in raw_names
            ]
            length_arr = split_group["length_arr"][...]

            sequences_group = split_group["sequences"]
            data_dict = {}

            
            for seq_name in name_list:
                seq_group = sequences_group[seq_name]
                payload = {}
                for key, value in seq_group.items():
                    payload[key] = self._read_hdf5_value(value)
                data_dict[seq_name] = payload

            self.name_list = tuple(name_list)
            self.length_arr = np.array(length_arr)
            self.data_dict = data_dict
            self.num_samples = len(self.name_list)
            self.nfeats = self.data_dict[self.name_list[0]]["motion"].shape[1]

    @staticmethod
    def _write_hdf5_value(group, key, value):
        if isinstance(value, torch.Tensor):
            dataset = group.create_dataset(key, data=value.detach().cpu().numpy(), compression="gzip")
            dataset.attrs["payload_type"] = "tensor"
        elif isinstance(value, np.ndarray):
            dataset = group.create_dataset(key, data=value, compression="gzip")
            dataset.attrs["payload_type"] = "ndarray"
        elif isinstance(value, (str, bytes)):
            dataset = group.create_dataset(key, data=np.string_(value))
            dataset.attrs["payload_type"] = "str"
        elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            dataset = group.create_dataset(key, data=value)
            dataset.attrs["payload_type"] = "scalar"
        else:
            dataset = group.create_dataset(key, data=np.string_(json.dumps(value)))
            dataset.attrs["payload_type"] = "json"

    @staticmethod
    def _read_hdf5_value(dataset):
        payload_type = dataset.attrs.get("payload_type")
        if payload_type == "tensor":
            return torch.from_numpy(dataset[...])
        if payload_type == "ndarray":
            return dataset[...]
        if payload_type == "str":
            raw = dataset[()]
            return raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
        if payload_type == "scalar":
            raw = dataset[()]
            if isinstance(raw, np.ndarray):
                raw = raw.item()
            return raw
        if payload_type == "json":
            raw = dataset[()]
            if isinstance(raw, (bytes, np.bytes_)):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        return dataset[...]


def collate_fn(data):
    return data

# from torch.utils.data._utils.collate import default_collate
# class hand_dataset():
#     def __init__(self, **kwargs):
#         super().__init__()

#         self.batch_size = kwargs.get("batch_size", 1)
#         self.num_workers = kwargs.get("num_workers", 0)
#         self.shuffle = kwargs.get("shuffle", True)
#         overfit = kwargs.get("overfit", None) # used for debugging only
#         self.fps = kwargs.get("fps", 8) 
#         self.dataset = kwargs.get("dataset")
#         self.max_frames = kwargs.get("max_frames", 160)
#         self.dataset_dir = kwargs.get("dataset_dir")
    
#         # def load data
#         self.train_data = LoadData("train", **kwargs)
#         if overfit is not None:
#             print("Overfitting to ", overfit, " train samples")
#             self.valid_data = self.train_data
#         else:
#             self.valid_data = LoadData("val", **kwargs)

#         if kwargs.get("load_test", False) and overfit is not None:
#             self.test_data = self.train_data
#         elif kwargs.get("load_test", False) and overfit is None:
#             self.test_data = LoadData("test", **kwargs)
#         else:
#             self.test_data = None

#         print("total train samples: ", len(self.train_data))
#         print("total valid samples: ", len(self.valid_data))
#         if self.test_data is not None:
#             print("total test samples: ", len(self.test_data))

#         # need for the pytorhch lightning modeule to work
#         self.dataset_configs = {}
#         self.dataset_configs["train"] = self.train_data
#         self.train_dataloader = self._train_dataloader
#         self.dataset_configs["validation"] = self.valid_data
#         self.val_dataloader = self._val_dataloader
#         self.dataset_configs["test"] = self.test_data
#         self.test_dataloader = self._test_dataloader

#         seq_len_in_sec = 4
#         if kwargs.get("down_sample", False):
#             new_fps = self.max_frames // seq_len_in_sec
#             print(f"\nOld FPS: {self.fps} new fps {new_fps}")
#             self.fps = new_fps

#         #### self load the mean dict
#         self.mean = None
#         self.std = None
#         if kwargs.get("mean_std_dict", False):
#             mean_dict_file = kwargs.get("mean_std_dict")
#             full_path = os.path.join(self.dataset_dir, mean_dict_file)
#             assert os.path.exists(full_path), f"{full_path} does not exist"
#             stat_dict = np.load(full_path, allow_pickle=True).item()

#             self.mean = stat_dict["mean"].reshape(1, 1, -1)
#             self.std = stat_dict["std"].reshape(1, 1, -1)



#     def _train_dataloader(self):
#         return data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, 
#                                collate_fn=self._colate, num_workers=self.num_workers)


#     def _val_dataloader(self):
#         return data.DataLoader(dataset=self.valid_data, batch_size=1, shuffle=False, collate_fn=self._colate, num_workers=self.num_workers)

#     def _test_dataloader(self):
#         return data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False, num_workers=self.num_workers)
    
#     def _colate(self, batch):
#         return default_collate(batch)

#     def norm_to_metric_space(self, x):
#         return (x * self.std.to(x.device)) + self.mean.to(x.device)

#     def metric_space_to_norm(self, x):
#         return (x - self.mean.to(x.device)) / self.std.to(x.device)  

    

# class hand_test_dataset(hand_dataset):
#     def __init__(self, **kwargs):

#         self.batch_size = kwargs.get("batch_size", 1)
#         self.num_workers = kwargs.get("num_workers", 0)
#         self.shuffle = kwargs.get("shuffle", True)
#         # def load data
#         self.test_data = LoadData("test", **kwargs)
#         self.train_data = []
#         self.valid_data = []


#         print("total train samples: ", len(self.train_data))
#         print("total valid samples: ", len(self.valid_data))
#         if self.test_data is not None:
#             print("total test samples: ", len(self.test_data))

#         # need for the pytorhch lightning modeule to work
#         self.dataset_configs = {}
#         self.dataset_configs["train"] = self.train_data
#         self.train_dataloader = self._train_dataloader
#         self.dataset_configs["validation"] = self.valid_data
#         self.val_dataloader = self._val_dataloader
#         self.dataset_configs["test"] = self.test_data
#         self.test_dataloader = self._test_dataloader
