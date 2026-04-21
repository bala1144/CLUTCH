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

class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 train_model:list = ["trajectory", "hand_pose"],
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim
        self.nfeats = nfeats
        self.single_hand_feats = nfeats//2

        ### traject
        self.traj_feats = 9
        self.hand_pose_feats = self.single_hand_feats - self.traj_feats
        self.transl_feats = 3
        self.train_model = train_model

        self.encoder = Encoder(self.single_hand_feats,
                        output_emb_width,
                        down_t,
                        stride_t,
                        width,
                        depth,
                        dilation_growth_rate,
                        activation=activation,
                        norm=norm)

        self.traj_quantizer = None
        self.code_per_frame = 0
        self.num_codebook = code_num

        if "trajectory" in self.train_model:
                    
            self.traj_decoder = Decoder(self.traj_feats,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm)
            
            self.traj_quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
            self.code_per_frame+=2

        self.handpose_quantizer = None
        if "hand_pose" in self.train_model:

            self.handpose_decoder = Decoder(self.hand_pose_feats,
                                output_emb_width,
                                down_t,
                                stride_t,
                                width,
                                depth,
                                dilation_growth_rate,
                                activation=activation,
                                norm=norm)
            
            self.handpose_quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
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
                print("warning: using zeros for hp")
        
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

    def forward(self, features: Tensor):
        
        # Preprocess
        x_in = self.preprocess(features)

        # Encode
        x_encoder = self.encoder(x_in)

        if "trajectory" in self.train_model:
            # trajectory decoder
            traj_quantized, traj_loss, traj_perplexity = self.traj_quantizer(x_encoder)
            traj_decoder = self.traj_decoder(traj_quantized)
        else:
            traj_decoder = None # use the goround truth 
            traj_loss = 0    
            traj_perplexity = 0   


        if "hand_pose" in self.train_model:
            # hand pose quantization
            hp_quantized, hand_pose_loss, hp_perplexity = self.handpose_quantizer(x_encoder)
            hp_decoder = self.handpose_decoder(hp_quantized)
        else:
            hp_decoder = None
            hand_pose_loss = 0
            hp_perplexity = 0
            
        # loss
        loss = hand_pose_loss + traj_loss
        x_out = self.postprocess(traj_decoder, hp_decoder, x_in)
        

        ### trajctory
        if "trajectory" in self.train_model:
            x_quantized = traj_quantized
        else: 
            x_quantized = hp_quantized

        # x_quantized = torch.cat([traj_quantized, hp_quantized], dim=-1)
        perplexity = traj_perplexity + hp_perplexity

        return x_out, loss, perplexity, x_quantized

    def patch_forward(self, x_quantized:Tensor,  features: Tensor):
        
        # # Preprocess
        # x_in = self.preprocess(features)

        # # Encode
        # x_encoder = self.encoder(x_in)

        # # quantization
        # x_quantized, loss, perplexity = self.quantizer(x_encoder)

        # x_quantized = Bs x cb_dim x T
        Bs, _, T = x_quantized.shape
        patch = x_quantized.permute(0, 2, 1).reshape(-1, 1, self.code_dim).permute(0, 2, 1).contiguous() # (Bs*T) x cb_dim x 1

        # decoder
        x_decoder = self.decoder(patch) # (Bs*T, self.nfeats x 1 )
       
        ### undo the patching
        x_out = self.postprocess(x_decoder)
        x_ref = features.reshape(Bs*T, -1, self.nfeats).contiguous()

        return x_out, x_ref

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        Bs, T, _ = features.shape # Bs x T x 66
        x_in = self.preprocess(features)  # Bs*2 x 33 x T
        x_encoder = self.encoder(x_in).permute(0, 2, 1) # Bs*2 x T/2 x C

        if self.traj_quantizer is not None:
            traj_idx = self.traj_quantizer.quantize(x_encoder).view(Bs*2, -1, 1) # (Bs * 2) x d_T
            lh_traj_idx, rh_traj_idx = traj_idx[:Bs],  traj_idx[Bs:] # Bs x d_T, Bs x d_T
            # rh_traj_idx = lh_traj_idx ## dont know why the hell I need this

        if self.handpose_quantizer is not None:
            hp_idx = self.handpose_quantizer.quantize(x_encoder).view(Bs*2, -1, 1)
            lh_hp_idx, rh_hp_idx = hp_idx[:Bs],  hp_idx[Bs:] # Bs x d_T, Bs x d_T
            # rh_hp_idx = lh_hp_idx ## dont know why the hell I need this

        if self.traj_quantizer is not None and self.handpose_quantizer is not None:
            ## Bs x d_T x [lh-traj, lh-hp, rh-traj, rh-hp ]
            code_idx = torch.cat([lh_traj_idx, lh_hp_idx, rh_traj_idx, rh_hp_idx], dim=-1) # (Bs x d_t x 4)
        elif self.handpose_quantizer is not None:
            code_idx = torch.cat([lh_hp_idx, rh_hp_idx], dim=-1) 
        elif self.traj_quantizer is not None:
            code_idx = torch.cat([lh_traj_idx, rh_traj_idx], dim=-1) 
        else:
            raise("Both quantizer cannot be None")
 
        ### [lh-traj, lh-hp, rh-traj, rh-hp ] * T
        interleaved_code_idx = code_idx.reshape(Bs, -1)

        # latent, dist
        return interleaved_code_idx, x_encoder

    def decode(self, z: Tensor, x_encoder:Tensor=None):

        """
        z: interleaved idx
        """

        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        
        ## z.shape
        if z.shape[-1] % self.code_per_frame > 0: ### make it even number of z
            pad = z.shape[-1] % self.code_per_frame
            z = torch.cat([z, torch.zeros_like(z[:, :pad]).to(z.device)], dim=-1)

        Bs, T = z.shape ## z.shape
        z = z.reshape(Bs, -1, self.code_per_frame) ## (Bs x T x 4) ## remove the interleave idx

        #### decompose the hand wise and modality wise tokens
        if self.traj_quantizer is not None and self.handpose_quantizer is not None:
            lh_traj_idx, lh_hp_idx, rh_traj_idx, rh_hp_id = z[:, :, 0], z[:, :, 1], z[:, :, 2], z[:, :, 3] # Bs X T
        elif self.handpose_quantizer is not None:
            lh_hp_idx, rh_hp_id = z[:, :, 0], z[:, :, 1]
        elif self.traj_quantizer is not None:
            lh_traj_idx, rh_traj_idx = z[:, :, 0], z[:, :, 1]

        #### Decode the trajectory
        if self.traj_quantizer is not None:
            ## decode the trajector
            traj_idx = torch.cat([lh_traj_idx, rh_traj_idx], dim=0)
            traj_d = self.traj_quantizer.dequantize(traj_idx) ## (Bs*2 x T x C)
            traj_out = self.traj_decoder(traj_d.permute(0, 2, 1).contiguous()) ## (Bs*2 x 9 x T)
        else:
            traj_out = None
        
        #### Decode the handpose quantizer
        if self.handpose_quantizer is not None:
            hp_idx = torch.cat([lh_hp_idx, rh_hp_id], dim=0)
            hp_d = self.handpose_quantizer.dequantize(hp_idx) ## (Bs*2 x T x C)
            hp_out = self.handpose_decoder(hp_d.permute(0, 2, 1).contiguous()) ## (Bs*2 x 24 x T) 
        else:
            hp_out = None

        x_out = self.postprocess(traj_out, hp_out)

        # ##### used for debugging
        # x_d = traj_d + hp_d
        # traj_commit = F.mse_loss(x_encoder, traj_d.detach())
        # hp_commit = F.mse_loss(x_encoder, hp_d.detach())
        # commit = traj_commit + hp_commit
        
        # return x_out, 
        return x_out
    
    def debug(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        Bs, _, _ = features.shape # Bs x T x 66

        # convert the 2  hands to single hand
        x_in = self.preprocess(features)  # Bs*2 x 33 x T
        x_encoder_ = self.encoder(x_in) # Bs*2 x cb_dim x d_t

        # replaceing the preprocessing step
        x_encoder_ = x_encoder_.permute(0, 2, 1).contiguous() # Bs*2 x d_t x cb_dim
        x_encoder = x_encoder_.view(-1, x_encoder_.shape[-1])  # Bs*2*d_t x cb_dim

        # # Init codebook if not inited
        # if self.training and not self.traj_quantizer.init:
        #     self.traj_quantizer.init_codebook(x_encoder)
        #     self.handpose_quantizer.init_codebook(x_encoder)


        traj_idx = self.traj_quantizer.quantize(x_encoder) # (Bs * 2 * d_T)
        hp_idx = self.handpose_quantizer.quantize(x_encoder) # (Bs * 2 * d_T)

        
        ### decode the trajector
        # traj_idx = torch.cat([lh_traj_idx, rh_traj_idx], dim=0)
        traj_d = self.traj_quantizer.dequantize(traj_idx) ## (Bs*2*T x C)
        # traj_d = x_encoder + (traj_d - x_encoder).detach()
        # Update embeddings
        if self.training:
            perplexity = self.traj_quantizer.update_codebook(x_encoder, traj_idx)
        traj_d = traj_d.view(Bs*2, -1, self.code_dim) ## (Bs*2 x T x C)
        traj_out = self.traj_decoder(traj_d.permute(0, 2, 1).contiguous()) ## (Bs*2 x 9 x T)

        # Update embeddings
        if self.training:
            perplexity = self.handpose_quantizer.update_codebook(x_encoder, hp_idx)
        # hp_idx = torch.cat([lh_hp_idx, rh_hp_idx], dim=0)
        # hp_idx =  hp_idx.view(Bs*2, -1, self.code_dim) # 
        hp_d = self.handpose_quantizer.dequantize(hp_idx) ## (Bs*2*T x C)
        hp_d = x_encoder + (hp_d - x_encoder).detach()


        hp_d = hp_d.view(Bs*2, -1, self.code_dim) ## (Bs*2 x T x C)
        hp_out = self.handpose_decoder(hp_d.permute(0, 2, 1).contiguous()) ## (Bs*2 x 24 x T) 

        x_out = self.postprocess(traj_out, hp_out)
        # return x_out

        ##### used for debugging
        x_d = traj_d + hp_d
        traj_commit = F.mse_loss(x_encoder_, traj_d.detach())
        hp_commit = F.mse_loss(x_encoder_, hp_d.detach())
        commit = traj_commit + hp_commit

        # ### passthrough
        # x_d = x_encoder_ + (traj_d - x_encoder_).detach()
        # x_d = x_encoder_ + (traj_d - x_encoder_).detach()
        
        # return x_out, 
        return x_out, commit, 0, x_d
    
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('audio_encoder.')]
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        print(keys)
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        unmatched_keys = self.load_state_dict(sd, strict=False)
        print(f"\nRestored from {path}\n")
        print("unmatched keys")
        print(unmatched_keys)


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
