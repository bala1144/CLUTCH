import numpy as np
import os
import random
import torch
import time
from mGPT.config import instantiate_from_config
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
import mGPT.render.matplot.plot_3d_global as plot_3d
from mGPT.hand.body_models.mano_xx import mano_full_pose_to_mano_params, batch_to_mano_full_pose

class MotionGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae != None:
            self.vae = instantiate_from_config(motion_vae)

        # Instantiate motion-language model
        self.lm = instantiate_from_config(lm)

        # Freeze the motion tokenizer for lm training
        if 'lm' in self.hparams.stage:
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))

        ##
        self.create_visulizer(cfg)
        self.cfg = cfg


    def create_visulizer(self, cfg):
        self.vis_interval = cfg.TRAIN.vis_interval
        self.render_helper = None
        self.plot_helper = None
        mano_use_pca = False if self.datamodule.nfeats == 198 else True
        self.mano_pca_comp = 24
        from mGPT.hand.body_models.mano_xx import MANO_doubleX
        self.mano_doubleX = MANO_doubleX(
            batch_size=20,
            use_pca=mano_use_pca,
            num_pca_comps=self.mano_pca_comp,
            flat_hand_mean=True,
        )

        try:
            from mGPT.hand.visualizer.grab_visualizer import grab_mesh_viewer
            from mGPT.hand.visualizer.plot_visulizer import Plot_visulizer
        except Exception as exc:
            print(f"Skipping visualization helpers: {exc}")
        else:
            self.render_helper = grab_mesh_viewer()
            self.plot_helper = Plot_visulizer(self.datamodule.name)

        # if cfg.get("TEST_FOLDER_EXP", None) is not None:
        #     ## running test model
        #     self.test_out_dir = os.path.join(cfg.TEST_FOLDER_EXP, "videos")
        #     os.makedirs(self.test_out_dir, exist_ok=True)
        #     print(f"\ntest_video_vis:  {self.test_out_dir}")

        if cfg.get("FOLDER_EXP", None) is not None:

            self.train_out_dir = os.path.join(cfg.FOLDER_EXP, "train_progress_vis")
            os.makedirs(self.train_out_dir, exist_ok=True)
            print(f"\ntrain_progress_vis:  {self.train_out_dir}")

            self.val_out_dir = os.path.join(cfg.FOLDER_EXP, "val_progress_vis")
            os.makedirs(self.val_out_dir, exist_ok=True)
            print(f"\nval_progress_vis:  {self.val_out_dir}")

        else:
            print("SKIPPING train_progress_vis, val_progress_vis dir")

        ## ADDED ON JAN 22
        self.fps = self.datamodule.hparams.fps
        self.skip_frames = 1


    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]

        # Forward
        # texts = ['Generate motion: ' + text for text in texts]
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(texts)):
            if task not in ["t2m", "m2t"]:
                raise NotImplementedError
            motion = self.vae.decode(outputs[i])
            lengths.append(motion.shape[1])

            if motion.shape[1] > max_len:
                max_len = motion.shape[1]

            if task in ["t2m", "m2t"]:
                feats_rst_lst.append(motion)

        feats_rst = torch.zeros(
            (len(feats_rst_lst), max_len, motion.shape[-1])).to(self.device)

        # padding and concat
        for i in range(len(feats_rst_lst)):
            feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]

        # Recover joints for evaluation
        joints_rst = self.feats2joints(feats_rst)

        # return set
        outputs = {
            "texts": output_texts,
            "feats": feats_rst,
            "joints": joints_rst,
            "length": lengths
        }

        return outputs

    def train_lm_forward(self, batch):

        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        all_captions = batch['all_captions']
        if self.hparams.condition == 'caption':
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]

        # LLM Forward
        outputs = self.lm(texts, tokens_ref, lengths, tasks)

        return {'outputs': outputs}
    
    @torch.no_grad()
    def train_t2m_forward_nograd(self, batch):

        feats_ref = batch["motion_ref"]
        feats_idx = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]

        min_len = lengths.copy()

        ## reconstruct the ref motion from motion tokens   
        feat_ref_from_idx = torch.zeros_like(feats_ref) # 1 x T x 66
        for i in range(len(texts)):
            outputs = torch.clamp(feats_idx[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)
            motion = self.vae.decode(outputs.to(torch.long))
            feat_ref_from_idx[i:i + 1] = motion[:]

        assert feats_ref.shape == feat_ref_from_idx.shape
        feats_ref = feat_ref_from_idx

        # Forward
        outputs = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks)
    
        
        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        for i in range(len(texts)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1: # if len > 1, we have valid tokens
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], feats_ref[i].shape[0])
            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :min_len[i]]

        joints_ref = self.feats2joints_local(feats_ref)
        joints_rst = self.feats2joints_local(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            #
            "m_ref": feats_ref, 
            "m_rst": feats_rst,
            ## joitns
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len,
            # "length": lengths
            "outputs": outputs
        }

        return rs_set

    @torch.no_grad()
    def val_t2m_forward(self, batch, is_mm=False):
        
        # return self.train_t2m_forward_nograd(batch)

        feats_ref = batch["motion_ref"]
        # feats_idx = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]


        if is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            # instructions = pjoin(self.datamodule.hparams.data_root,
            #                      'template_instructions.json')
            # instructions = json.load(open(instructions, 'r'))
            # tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        # if self.hparams.condition == 'caption':
        #     tasks = [{
        #         'input': ['<Caption_Placeholder>'],
        #         'output': ['']
        #     }] * len(texts)

        # if self.hparams.cfg.DATASET.TASK_PATH:
        #     instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
        #     instructions = json.load(open(instructions, 'r'))
        #     tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        if True: # modify this later
            instructions = pjoin(self.datamodule.hparams.data_root, 'template_pretrain.json')
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        # Forward
        min_len = lengths.copy()
        outputs = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks)

        feats_rst = torch.zeros_like(feats_ref)
        for i in range(len(texts)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)
            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], feats_ref[i].shape[0])
            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :min_len[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints_local(feats_ref)
        joints_rst = self.feats2joints_local(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set
    
    @torch.no_grad()
    def val_m2t_forward(self, batch):

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        all_captions = batch['all_captions']

        if self.cfg.EVAL.get("DatasetEval", "") == "Text2MotionDatasetCB" or len(feats_ref.shape) == 2: 
            motion_tokens = feats_ref
            lengths_tokens = lengths
        else:
             # Motion Encode
            motion_tokens = []
            lengths_tokens = []
            for i in range(len(feats_ref)):
                motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
                motion_tokens.append(motion_token[0])
                lengths_tokens.append(motion_token.shape[1])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,lengths=lengths_tokens,task="m2t", stage='test')

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "t_ref": all_captions,
            # "t_ref": texts,
            "t_pred": outputs,
            "length": lengths_tokens
        }

        return rs_set

    @torch.no_grad()
    def feats2joints_local(self, feats):
        
        feats_ = self.datamodule.denormalize(feats)
        mano_parmas = mano_full_pose_to_mano_params(feats_)
        lh_output, rh_output = self.mano_doubleX(**mano_parmas)
        joints_with_tip = torch.cat((lh_output.joints_w_tip, rh_output.joints_w_tip), dim=2) # Bs, T, 42 x 3
        return joints_with_tip

    def allsplit_step(self, split: str, batch, batch_idx):
        
        """
        For train:
        ==========
        1. irrespective of task, rs_set = self.train_lm_forward(batch)
        2. loss = self._losses['losses_' + split].update(rs_set)

        For val:
        ========
        The problem with val in MotionGPT is that, during training the task is predefined to t2m, and other task related options in the function are only used during testing to compute metric
        
        1. Run self.train_lm_forward(batch) with torch.no_grad() to compute the LM loss generalization or not
        2. For t2m metrics, rs_set = self.val_t2m_forward(batch) and update the metrics
        3. For m2t metrics, rs_set = self.val_m2t_forward(batch)
    
        """

        loss = None
        if self.hparams.stage in ["lm_instruct", "lm_pretrain"] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)

            ### runs t2m on the train input
            if (self.current_epoch+1) % self.vis_interval == 0 and batch_idx == 0:
                rs_set_local = self.val_t2m_forward(batch)
                ## following is replaced with val_t2m_forward, since val and train uses the same quantized dataloader
                ## rs_set_local = self.train_t2m_forward_nograd(batch)
                self.visualize_results(rs_set_local, self.train_out_dir, batch)

        # Compute the metrics
        if split in ["val"]:

            # print(f"running val task {self.hparams.task} at:", self.current_epoch)
            if self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl"]:
                
                ## compute the cross entrophy loss on the model
                ### this block is just added to compute cross entrophy loss
                if self.cfg.EVAL.get("DatasetEval", "") == "Text2MotionDatasetCB":        
                    with torch.no_grad():
                        rs_set = self.train_lm_forward(batch)
                        loss = self._losses['losses_' + split].update(rs_set)

                
                ##### Text-to-Motion
                if "Text-to-Motion" in self.cfg.DATASET.get("tasks_to_load", []):
                    ### Run the t2m validation, visualize and evalute the metric  
                    rs_set = self.val_t2m_forward(batch)
                    
                    lengths = batch['length']
                    metric = "Egovid5M_Metrics"
                    self.hparams.metrics_dict = [metric]
                    getattr(self.metrics, metric).update(
                        feats_ref=rs_set["m_ref"],
                        feats_rst=rs_set["m_rst"],
                        joints_ref=rs_set["joints_ref"],
                        joints_rst=rs_set["joints_rst"],
                        lengths_ref=lengths,
                        lengths_rst=rs_set['length'],
                        word_embs=None,
                        pos_ohot=None,
                        text_lengths=None,
                    )

                    # visualize results for the first 'n' batches
                    if batch_idx < 2 and  self.current_epoch > 0:
                        self.visualize_results(rs_set, self.val_out_dir, batch)

                 ##### Motion-to-Text
                if "Motion-to-Text" in self.cfg.DATASET.get("tasks_to_load", []):

                    ### Run the m2t validation, visualize and evalute the metric 
                    rs_set = self.val_m2t_forward(batch)
                    metric = "Egovid5M_M2TMetrics"
                    self.hparams.metrics_dict.append(metric)
                    getattr(self.metrics, metric).update(
                        feats_ref=rs_set["m_ref"],
                        pred_texts=rs_set["t_pred"],
                        gt_texts=batch["all_captions"],
                        lengths=rs_set['length'],
                        word_embs=batch.get("word_embs", None),
                        pos_ohot=batch.get("pos_ohot", None),
                        text_lengths=batch.get("text_len", None),
                    )

                    if batch_idx < 2 and  self.current_epoch > 0:
                        self.visualize_results(rs_set, self.val_out_dir, batch)

        # return forward output rather than loss during test
        if split in ["test"] and self.hparams.task == "t2m":
            return rs_set["joints_rst"], rs_set["length"], rs_set[
                    "joints_ref"]

        return loss   

    def visualize_results(self, rs_set, out_dir, batch):

        ### x_t
        # self.mano_doubleX = self.mano_doubleX.to("cpu")
        x_t = self.datamodule.denormalize(rs_set["m_ref"])
        xt_mano_full_pose = x_t[:1] # first sample
        xt_mano_params = mano_full_pose_to_mano_params(xt_mano_full_pose)
        curr_xt = self.mano_doubleX.get_scene_verts_from_batch(**xt_mano_params)[0] # returns numpy array

        model_output = self.datamodule.denormalize(rs_set["m_rst"])
        pred_mano_full_pose = model_output[:1] # extract the first seq
        pred_mano_params = mano_full_pose_to_mano_params(pred_mano_full_pose) # outdict
        curr_pred = self.mano_doubleX.get_scene_verts_from_batch(**pred_mano_params)[0]

        seq_name = batch["seq_name"][0]
        curr_name = seq_name + "_%06d"%self.current_epoch
        if batch.get("text", None) is not None:
            text = batch["text"][0]
        else:
            text = curr_name

        if batch.get("motion_mask", None) is not None:
            motion_mask = batch.get("motion_mask")
            seq_len = torch.nonzero(motion_mask, as_tuple=False).max().item() + 1
            curr_pred = curr_pred[:seq_len]
            curr_xt = curr_xt[:seq_len]
        
        try:
            out_file = self.render_helper.visualize_hands_with_gt(out_dir,
                                                            curr_name,
                                                            curr_pred,
                                                            curr_xt,
                                                            self.mano_doubleX.faces,
                                                            fps=self.fps,
                                                            text=text
                                                            )
            print("visualize the prediction at %s"%out_file)

        except (OSError, IOError) as e:
            print(f"Error while rendering this file {curr_name}")
            print(f"Error: {e}")

        ### this is good for now, need to be fixed later
        out_file = self.plot_helper.visualize_results_as_plots(rs_set["m_rst"], rs_set["m_ref"], batch,  out_dir, curr_name, None)

    def visualize_test_results(self, rs_set, out_dir, batch):

        model_output = self.datamodule.denormalize(rs_set["m_rst"])
        pred_mano_full_pose = model_output[:1] # extract the first seq
        pred_mano_params = mano_full_pose_to_mano_params(pred_mano_full_pose) # outdict
        curr_pred = self.mano_doubleX.get_scene_verts_from_batch(**pred_mano_params)[0]

        seq_name = batch["seq_name"][0]
        curr_name = seq_name
        if batch.get("text", None) is not None:
            text = batch["text"][0]
        else:
            text = curr_name
            
        
        out_file = self.render_helper.visualize_hands(out_dir,
                                                        curr_name,
                                                        curr_pred,
                                                        self.mano_doubleX.faces,
                                                        fps=self.fps,
                                                        text=text
                                                        )
        print("visualize the prediction at %s"%out_file)

        # ### this is good for now, need to be fixed later
        # out_file = self.plot_helper.visualize_results_as_plots(rs_set["m_rst"], rs_set["m_ref"], batch,  out_dir, curr_name, None)

        return curr_pred, curr_name


    def dump_as_npy(self, rs_set, out_dir, batch):

        """
        Dump: as "npy" file
        """

        model_output = self.datamodule.denormalize(rs_set["m_rst"])
        pred_mano_full_pose = model_output[:1] # extract the first seq
        pred_mano_params = mano_full_pose_to_mano_params(pred_mano_full_pose) # outdict
        curr_pred = self.mano_doubleX.get_scene_verts_from_batch(**pred_mano_params)[0]

        seq_name = batch["seq_name"][0]
        curr_name = seq_name
        if batch.get("sentence", None) is not None:
            text = batch["sentence"][0]
        else:
            text = curr_name
            
        out_dict = dict(
            pred_verts=curr_pred,
            pred_mano_params={ k:v.cpu().numpy() for k, v in pred_mano_params.items()},
            pred_mano_full_pose=pred_mano_full_pose.cpu().numpy(),
            text=text,
        )

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, seq_name+".npy")
        np.save(out_file, out_dict, allow_pickle=True)
        print("Outdict : %s"%out_file)
