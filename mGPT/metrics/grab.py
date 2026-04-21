from typing import List
import os
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
from .utils import *
from mGPT.config import instantiate_from_config
from omegaconf import OmegaConf

### added for comuting the GRAB metrics
# from mdm_grab.metrics.interpenetration_volume import compute_intersect_vox_on_seq, compute_intersect_vox_with_depth_on_seq, max_interpenetration_depth_on
# from mdm_grab.utils.temporal_trimesh import temporal_trimesh
from mGPT.hand.utils.temporal_dict import temporal_dict
from tqdm import tqdm
from mGPT.hand.metrics.evaluate_statistical_metrics import *
from mGPT.hand.body_models.mano_xx import MANO_doubleX, to_cpu, to_numpy, mano_full_pose_to_mano_params


class GRABMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname='GRAB',
                 top_k=3,
                 R_size=32,
                 diversity_times=100,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "matching, fid, and diversity scores"
        self.top_k = top_k
        self.R_size = R_size
        self.text = 'lm' in cfg.TRAIN.STAGE and cfg.model.params.task == 't2m'
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = ['Recon',
                        'AC_div', 'AC_multi_mod', 'AC_gt_div', 'AC_gt_multi_mod', 'WT_div', 'WT_multi_mod', 'FID'
                        ]

        for metric in self.metrics:
            self.add_state(metric,default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Chached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)

                ###
        self.add_state("recmotion_labels", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_labels", default=[], dist_reduce_fx=None)

        ###
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)
        
        ##
        self.add_state("gtmotion_feat", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_feat", default=[], dist_reduce_fx=None)

        ##
        self.add_state("gtmotion_joints", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_joints", default=[], dist_reduce_fx=None)


        # T2M Evaluator
        self._get_t2m_evaluator(cfg)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """

        model_cfg = "/path/to/checkpoints/ac_01_train_S2O/args.yml"
        ckpt_path = "/path/to/checkpoints/ac_01_train_S2O/model.pt"
        model_cfg = OmegaConf.load(model_cfg).motion_model
        self.action_classifier = instantiate_from_config(model_cfg)
        self.action_classifier.init_from_ckpt(ckpt_path)
        self.action_classifier.eval()
        for p in self.action_classifier.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def compute(self, sanity_flag):
        
        scale = 100.0
        pitch = 1.0

        count = self.count.item()
        count_seq = self.count_seq.item()

        # Init metrics dict
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        
        # Jump in sanity check stage
        if sanity_flag:
            return metrics

        # Cat cached batches and shuffle
        shuffle_idx = torch.randperm(count_seq)

        all_gt_motions = torch.concatenate(self.gtmotion_feat, axis=0)
        all_pred_motions = torch.concatenate(self.recmotion_feat, axis=0)
 
        bs = all_gt_motions.shape[0]
        recon = torch.norm(all_pred_motions.reshape(bs, -1) - all_gt_motions.reshape(bs,-1), p=2, dim=-1).mean(0)
        metrics["Recon"] = recon

        y_pred =  torch.concatenate(self.recmotion_labels, axis=0)
        y_gt = torch.concatenate(self.gtmotion_labels, axis=0)

        # y_pred = self.recmotion_labels
        # y_gt = self.gtmotion_labels
        
        pred_activations = torch.concatenate(self.recmotion_embeddings, axis=0)
        gt_activations = torch.concatenate(self.gtmotion_embeddings, axis=0)

        JTS_pred = torch.concatenate(self.recmotion_joints, axis=0)
        JTS_gt =torch.concatenate(self.gtmotion_joints, axis=0)

        seqwise_loss_dict = temporal_dict()

        #### 
        divers = calculate_diversity_(pred_activations, y_pred, 29)
        seqwise_loss_dict.add_dict({"div":divers}, tag="AC_")

        multimodality = calculate_multimodality_(pred_activations, y_pred, 29)
        seqwise_loss_dict.add_dict({"multi_mod":multimodality}, tag="AC_")


        divers = calculate_diversity_(gt_activations, y_gt, 29)
        seqwise_loss_dict.add_dict({"div":divers}, tag="AC_gt_")

        multimodality = calculate_multimodality_(gt_activations, y_gt, 29)
        seqwise_loss_dict.add_dict({"multi_mod":multimodality}, tag="AC_gt_")


        ## all 6 joints
        action_labels_pred = torch.cat(y_pred, dim=0) if type(y_pred) == list else y_pred
        left_writst_joint = JTS_pred[:, :, 1, :]
        right_writst_joint = JTS_pred[:, :, 21, :]
        wrist_JT = torch.cat([left_writst_joint, right_writst_joint], dim=-1) # N x T X 6
        assert wrist_JT.shape[-1] == 6

        WT_divers = calculate_diversity_(wrist_JT, action_labels_pred, 29)
        seqwise_loss_dict.add_dict({"div": WT_divers}, tag="WT_")

        WT_multimodality = calculate_multimodality_(wrist_JT, action_labels_pred, 29)
        seqwise_loss_dict.add_dict({"multi_mod": WT_multimodality}, tag="WT_")

        mu, cov = calculate_activation_statistics_np(pred_activations.cpu().numpy() * scale)
        gt_mu, gt_cov = calculate_activation_statistics_np(gt_activations.cpu().numpy() * scale)
        fid_met = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        seqwise_loss_dict.add_dict({"FID": fid_met}, tag="")

        average_dict = seqwise_loss_dict.average()
        for k, v in average_dict.items():
            metrics[k] = v
        
        
        # Reset
        self.reset()

        return {**metrics}

    @torch.no_grad()
    def update(self,
               feats_ref: Tensor, # Bs x T x 66(hands params)
               feats_rst: Tensor,  # Bs x T x 66(hands params)
                joints_ref: Tensor, # Bs x T x 66(hands params)
               joints_rst: Tensor,  # Bs x T x 66(hands params)
               lengths_ref: List[int], # [T] * Bs
               lengths_rst: List[int], # [T] * Bs
               word_embs: Tensor = None,
               pos_ohot: Tensor = None,
               text_lengths: Tensor = None):
        
        self.count += sum(lengths_ref)
        self.count_seq += len(lengths_ref)


        cache = [0] * feats_ref.shape[0]
        joints_cache = [0] * feats_ref.shape[0]
        for i in range(feats_ref.shape[0]):
            cache[i] = feats_ref[i:i + 1]
            joints_cache[i] = joints_ref[i:i + 1]
        self.gtmotion_feat.extend(cache)
        self.gtmotion_joints.extend(joints_cache)

        activation_cache, label_cache = self.get_motion_features(joints_ref)
        self.gtmotion_embeddings.extend(activation_cache)
        self.gtmotion_labels.extend(label_cache)

        cache = [0] * feats_rst.shape[0]
        joints_cache = [0] * feats_ref.shape[0]
        for i in range(feats_rst.shape[0]):
            cache[i] = feats_rst[i:i + 1]
            joints_cache[i] = joints_rst[i:i + 1]

        self.recmotion_feat.extend(cache)
        self.recmotion_joints.extend(joints_cache)

        activation_cache, label_cache = self.get_motion_features(joints_rst)
        self.recmotion_embeddings.extend(activation_cache)
        self.recmotion_labels.extend(label_cache)


    def get_motion_features(self, joints: Tensor):
        
        jt = joints

        # jt = joints.detach().cpu()
        # for p in self.action_classifier.parameters():
        #     p.to("cpu")
        
        num_seq = joints.shape[0]
        activation_cache = [0] * joints.shape[0]
        label_cache = [0] * joints.shape[0]

        for idx in range(num_seq):
            
            # c_d = feats[idx:idx+1].device
            # mano_parmas = mano_full_pose_to_mano_params(feats[idx:idx+1])
            # # self.mano = self.mano.to(c_d)
            # lh_output, rh_output= self.mano.get_hands_from_payload(**mano_parmas)
            # joints_with_tip = torch.cat((lh_output.joints_w_tip, rh_output.joints_w_tip), dim=1).detach()
            # joint_cache[idx] = joints_with_tip
            
            # device = next(self.action_classifier.parameters()).device
            # label_cache[idx], activation_cache[idx] = self.action_classifier.predict(jt[idx].to(device))
            label_cache[idx], activation_cache[idx] = self.action_classifier.predict(jt[idx])

        return activation_cache, label_cache
