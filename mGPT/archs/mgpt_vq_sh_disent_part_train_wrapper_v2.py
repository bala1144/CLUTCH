###
# Partially from https://github.com/Mael-zys/T2M-GPT
###
from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict
import torch.nn.functional as F
from .mgpt_vq_single_hand_disent_part_train import VQVae
from copy import deepcopy
    
class VQVae_wrapper(nn.Module):

    def __init__(self,
                #  nfeats: int,
                #  quantizer: str = "ema_reset",
                #  code_num=512,
                #  code_dim=512,
                #  output_emb_width=512,
                #  down_t=3,
                #  stride_t=2,
                #  width=512,
                #  depth=3,
                #  dilation_growth_rate=3,
                #  norm=None,
                #  activation: str = "relu",
                #  train_model:list = ["trajectory", "hand_pose"],
                 **kwargs) -> None:

        super().__init__()

        """
        In this version the issue with number of codebooks is fixed 
        
        """

        self.code_dim = kwargs.get("code_dim")
        self.nfeats = kwargs.get("nfeats")
        self.single_hand_feats = kwargs.get("nfeats")//2
        self.down_t = kwargs.get("down_t")

        ### traject
        self.traj_feats = 9
        self.hand_pose_feats = self.single_hand_feats - self.traj_feats
        self.transl_feats = 3
        self.train_model = kwargs.get("train_model")

        self.code_per_frame = 0

        ### used for later 
        self.traj_code_start = 0
        self.traj_code_end =  kwargs["code_num"] // 2
        self.hp_code_start = kwargs["code_num"] // 2
        self.hp_code_end =  self.hp_code_start  + (kwargs["code_num"] // 2)
        self.hp_code_num = kwargs["code_num"]
        
        if "trajectory" in self.train_model:
            traj_kwargs = deepcopy(kwargs)

            traj_kwargs["train_model"] = "trajectory"
            traj_kwargs["code_num"] =  kwargs["code_num"] // 2
            self.traj_model = VQVae(**traj_kwargs)
            self.code_per_frame+=2
        

        if "hand_pose" in self.train_model:
            hp_kwargs = deepcopy(kwargs)
            hp_kwargs["train_model"] = "hand_pose"
            hp_kwargs["code_num"] =  kwargs["code_num"] // 2
            self.hp_model = VQVae(**hp_kwargs)
            self.code_per_frame+=2
        
    def preprocess(self, x):

        """
        x = Bs x T x self.nfeats(198 or 66 or 90)
        
        """

        ### convert both hands to single hands
        lh_hand = x[:, :, :self.single_hand_feats ]
        rh_hand = x[:, :, self.single_hand_feats: ]
        x = torch.cat([lh_hand, rh_hand], dim=0)

        ### (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)

        return x

    def postprocess(self, traj, hp, x_t=None):

        if traj is None:
            gt_orien = x_t[:, :6, :]
            gt_tran = x_t[:, -3:, :]
            traj = torch.cat([gt_orien, gt_tran], dim=1)# bs x 9 x T
        
        if hp is None:
            if x_t is not None:
                hp = x_t[:, 6:-3, :]  # bs x 24 x T or bs x 90 x T 
            else:
                Bs = traj.shape[0]
                T = traj.shape[-1]
                hp = torch.zeros((Bs, self.hand_pose_feats , T)).to(traj.device) 
        
        ### (bs, Jx3, T) ->  (bs, T, Jx3)
        hp = hp.permute(0, 2, 1) # bs X T x 24
        traj = traj.permute(0, 2, 1) # bs x T x 9

        ## decompose
        orientation = traj[:,:,:6]
        trans = traj[:,:,-3:]
        x = torch.cat([orientation, hp, trans], dim=-1)

        ###
        Bs, _ , _ = x.shape
        lh_hand = x[:Bs//2] # bs x T x 33
        rh_hand = x[Bs//2:]
        x = torch.cat([lh_hand, rh_hand], dim=-1)

        assert x.shape[-1] == self.nfeats
        return x
    
    def forward(self, features:Tensor):

        interleaved_code_idx, _ = self.encode(features)
        x_out =  self.decode(interleaved_code_idx)
        return x_out, torch.tensor([1e30]), torch.tensor([1e30]), _

    def decode_lm_embed_to_motion(self, lm_embed:Tensor ):

        """
        featurs: Bs x T/4 x 64

        144 / *
        
        """

        Bs, T, d = lm_embed.shape
        lm_embed = lm_embed.reshape(Bs, -1, self.code_per_frame,  d) 
        lh_traj_feat, lh_hp_feat, rh_traj_feat, rh_hp_feat = lm_embed[:, :, 0], lm_embed[:, :, 1], lm_embed[:, :, 2], lm_embed[:, :, 3] # each component 

        # feat to traj
        traj_feat = torch.cat([lh_traj_feat, rh_traj_feat], dim=0) # 2*Bs x T x d
        traj_idx = self.traj_model.traj_quantizer.quantize(traj_feat).view(Bs*2, -1)

        print("")
        
        traj_d = self.traj_model.traj_quantizer.dequantize(traj_idx) ## (Bs*2 x T x C)
        traj_out = self.traj_model.traj_decoder(traj_d.permute(0, 2, 1).contiguous()) 
        
        # feat to motion
        hp_feat = torch.cat([lh_hp_feat, rh_hp_feat], dim=0 )# 2*Bs x T x d
        hp_idx = self.hp_model.handpose_quantizer.quantize(hp_feat).view(Bs*2, -1)
        hp_d = self.hp_model.handpose_quantizer.dequantize(hp_idx) ## (Bs*2 x T x C)
        hp_out = self.hp_model.handpose_decoder(hp_d.permute(0, 2, 1).contiguous()) ## (Bs*2 x 24 x T) 

        x_out = self.postprocess(traj_out, hp_out)
        # return x_out, 
        return x_out

    def decode_from_gumbel_softmax_logits(self, motion_one_hot:Tensor ):

        """
        featurs: Bs x T/4 x 64

        144 / *
        
        """

        Bs, T, d = motion_one_hot.shape
        motion_one_hot = motion_one_hot.reshape(Bs, -1, self.code_per_frame,  d) 
        lh_traj_feat, lh_hp_feat, rh_traj_feat, rh_hp_feat = motion_one_hot[:, :, 0], motion_one_hot[:, :, 1], motion_one_hot[:, :, 2], motion_one_hot[:, :, 3] # each component 

        # feat to traj
        traj_feat = torch.cat([lh_traj_feat, rh_traj_feat], dim=0) # 2*Bs x T x d
        traj_feat = traj_feat[:, :, self.traj_code_start:self.traj_code_end]
        traj_d = traj_feat @ self.traj_model.traj_quantizer.codebook
        traj_out = self.traj_model.traj_decoder(traj_d.permute(0, 2, 1).contiguous()) 
        
        # feat to motion
        hp_feat = torch.cat([lh_hp_feat, rh_hp_feat], dim=0 )# 2*Bs x T x d
        hp_feat = hp_feat[:, :, self.hp_code_start:self.hp_code_end]
        hp_d = hp_feat @ self.hp_model.handpose_quantizer.codebook
        hp_out = self.hp_model.handpose_decoder(hp_d.permute(0, 2, 1).contiguous()) ## (Bs*2 x 24 x T) 

        x_out = self.postprocess(traj_out, hp_out)
        # return x_out, 
        return x_out
    
    def get_tokens_from_logits(self, motion_one_hot:Tensor ):

        """
        featurs: Bs x T/4 x 64

        144 / *
        
        """

        Bs, T, d = motion_one_hot.shape
        motion_one_hot = motion_one_hot.reshape(Bs, -1, self.code_per_frame,  d) 
        lh_traj_feat, lh_hp_feat, rh_traj_feat, rh_hp_feat = motion_one_hot[:, :, 0], motion_one_hot[:, :, 1], motion_one_hot[:, :, 2], motion_one_hot[:, :, 3] # each component 

        # feat to traj
        lh_traj_idx = torch.argmax(F.softmax(lh_traj_feat[:, :, self.traj_code_start:self.traj_code_end]), dim=-1).reshape(Bs, -1, 1)
        rh_traj_idx = torch.argmax(F.softmax(rh_traj_feat[:, :, self.traj_code_start:self.traj_code_end]), dim=-1).reshape(Bs, -1, 1)
        lh_hp_idx = torch.argmax(F.softmax(lh_hp_feat[:, :, self.hp_code_start:self.hp_code_end]), dim=-1).reshape(Bs, -1, 1)
        rh_hp_idx = torch.argmax(F.softmax(rh_hp_feat[:, :, self.hp_code_start:self.hp_code_end]), dim=-1).reshape(Bs, -1, 1)


        ## Bs x d_T x [lh-traj, lh-hp, rh-traj, rh-hp ]
        code_idx = torch.cat([lh_traj_idx, lh_hp_idx, rh_traj_idx, rh_hp_idx], dim=-1) # (Bs x d_t x 4)
        ### [lh-traj, lh-hp, rh-traj, rh-hp ] * T
        interleaved_code_idx = code_idx.reshape(Bs, -1)

        # traj_feat = torch.cat([lh_traj_feat, rh_traj_feat], dim=0) # 2*Bs x T x d
        # traj_feat = traj_feat[:, :, self.traj_code_start:self.traj_code_end]
        # traj_idx = F.softmax(traj_feat[:, :, self.traj_code_start:self.traj_code_end]).argmax(dim=-1)

        
        # # feat to motion
        # hp_feat = torch.cat([lh_hp_feat, rh_hp_feat], dim=0 )# 2*Bs x T x d
        # hp_feat = hp_feat[:, :, self.hp_code_start:self.hp_code_end]
        # hp_idx = F.softmax(hp_feat).argmax(dim=-1)

        # return x_out, 
        return interleaved_code_idx



    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        ### get the trajectory model
        Bs, T, _ = features.shape # Bs x T x 66
        traj_x_in = self.traj_model.preprocess(features)  # Bs*2 x 33 x T
        traj_encoder = self.traj_model.encoder(traj_x_in).permute(0, 2, 1) # Bs*2 x T/2 x C
        traj_idx = self.traj_model.traj_quantizer.quantize(traj_encoder).view(Bs*2, -1, 1) # (Bs * 2) x d_T
        lh_traj_idx, rh_traj_idx = traj_idx[:Bs],  traj_idx[Bs:] # Bs x d_T, Bs x d_T
            # rh_traj_idx = lh_traj_idx ## dont know why the hell I need this

        ### get the hand motion model
        traj_x_in = self.hp_model.preprocess(features)  # Bs*2 x 33 x T
        traj_encoder = self.hp_model.encoder(traj_x_in).permute(0, 2, 1) # Bs*2 x T/2 x C
        hp_idx = self.hp_model.handpose_quantizer.quantize(traj_encoder).view(Bs*2, -1, 1)
        ## adding hand pose offset
        hp_idx = hp_idx + self.hp_code_start 
        lh_hp_idx, rh_hp_idx = hp_idx[:Bs],  hp_idx[Bs:] # Bs x d_T, Bs x d_T
            # rh_hp_idx = lh_hp_idx ## dont know why the hell I need this

        ## Bs x d_T x [lh-traj, lh-hp, rh-traj, rh-hp ]
        code_idx = torch.cat([lh_traj_idx, lh_hp_idx, rh_traj_idx, rh_hp_idx], dim=-1) # (Bs x d_t x 4)
 
        ### [lh-traj, lh-hp, rh-traj, rh-hp ] * T
        interleaved_code_idx = code_idx.reshape(Bs, -1)

        # latent, dist
        return interleaved_code_idx, _

    def decode(self, z: Tensor, x_encoder:Tensor=None):

        """
        z: interleaved idx
        """

        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        
        ## z.shape
        if z.shape[-1] % self.code_per_frame > 0: ### make it even number of z
            pad = self.code_per_frame - (z.shape[-1] % self.code_per_frame)
            z = torch.cat([z, torch.zeros_like(z[:, :pad]).to(z.device)], dim=-1)

        Bs, T = z.shape ## z.shape
        z = z.reshape(Bs, -1, self.code_per_frame) ## (Bs x T x 4) ## remove the interleave idx
        #### decompose the hand wise and modality wise tokens
        lh_traj_idx, lh_hp_idx, rh_traj_idx, rh_hp_id = z[:, :, 0], z[:, :, 1], z[:, :, 2], z[:, :, 3] # Bs X T

        #### Decode the trajectory
        ## decode the trajector
        traj_idx = torch.cat([lh_traj_idx, rh_traj_idx], dim=0)
        traj_idx = traj_idx.clamp(min=0, max=self.traj_model.num_codebook-1) 
        traj_d = self.traj_model.traj_quantizer.dequantize(traj_idx) ## (Bs*2 x T x C)
        traj_out = self.traj_model.traj_decoder(traj_d.permute(0, 2, 1).contiguous()) ## (Bs*2 x 9 x T)
        
        #### Decode the handpose quantizer
        hp_idx = torch.cat([lh_hp_idx, rh_hp_id], dim=0)
        ### added for handling offset
        hp_idx = hp_idx - self.hp_code_start
        hp_idx = hp_idx.clamp(min=0, max=self.hp_model.num_codebook-1) ### handling the negative numbers
        hp_d = self.hp_model.handpose_quantizer.dequantize(hp_idx) ## (Bs*2 x T x C)
        hp_out = self.hp_model.handpose_decoder(hp_d.permute(0, 2, 1).contiguous()) ## (Bs*2 x 24 x T) 

        x_out = self.postprocess(traj_out, hp_out)
        # return x_out, 
        return x_out
    
    
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('audio_encoder.')]
    