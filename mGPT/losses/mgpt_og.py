import torch
import torch.nn as nn
from .base import BaseLosses


class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit


class Temporal_smooth_loss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, gt, **kwargs):
        loss = torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))
        return loss


class GPTLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS

        # Define losses
        losses = []
        params = {}
        if stage == "vae":

            if cfg.LOSS.LAMBDA_FEATURE > 1e-5:
                losses.append("recons_feature")
                params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            if cfg.LOSS.LAMBDA_VELOCITY > 1e-5:
                losses.append("recons_velocity")
                params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

            if cfg.LOSS.LAMBDA_COMMIT > 1e-5:
                losses.append("vq_commit")
                params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT

            if cfg.LOSS.get("LAMBDA_PATCH_RECON", None) is not None and cfg.LOSS.get("LAMBDA_PATCH_RECON") > 1e-5:
                losses.append("recons_patch")
                params['recons_patch'] = cfg.LOSS.LAMBDA_PATCH_RECON

            if cfg.LOSS.get("LAMBDA_3D_RECON", None) is not None and cfg.LOSS.get("LAMBDA_3D_RECON") > 1e-5:
                losses.append("recons_3D")
                params['recons_3D'] = cfg.LOSS.LAMBDA_3D_RECON
            
            if cfg.LOSS.get("LAMBDA_JOINT_RECON", None) is not None and cfg.LOSS.get("LAMBDA_JOINT_RECON") > 1e-5:
                losses.append("recons_joints")
                params['recons_joints'] = cfg.LOSS.LAMBDA_JOINT_RECON

            if cfg.LOSS.get("LAMBDA_TEMPORAL_SMOOTHNESS", None) is not None and cfg.LOSS.LAMBDA_TEMPORAL_SMOOTHNESS > 1e-5:
                losses.append("temporal_smooth")
                params['temporal_smooth'] = cfg.LOSS.LAMBDA_TEMPORAL_SMOOTHNESS

            losses.append("vq_perplexity")
            params['vq_perplexity'] = 1.0


        elif stage in ["lm_pretrain", "lm_instruct"]:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS

        elif stage in ["classifier"]:
            losses.append("cls_loss")
            params['cls_loss'] = cfg.LOSS.LAMBDA_CLS

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == 'recons':
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[1] in ['commit', 'loss', 'gpt', 'm2t2m', 't2m2t', 'perplexity']:
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                losses_func[loss] = nn.CrossEntropyLoss
            elif loss.split('_')[1] in  ["smooth"]:
                # losses_func[loss] = lambda t_1, t_0: torch.mean(torch.abs(nn.L1Loss(t_1, t_0, reduction='None')))
                losses_func[loss] = Temporal_smooth_loss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints, **kwargs)

    def update(self, rs_set):
        '''Update the losses'''
        total: float = 0.0

        if self.stage in ["vae"]:
            
            if "recons_feature" in self.losses:
                total += self._update_loss("recons_feature", rs_set['m_rst'],rs_set['m_ref'])
                # print("recons_feature")

            if "temporal_smooth" in self.losses:
                recons_temp_smooth = self._update_loss("temporal_smooth", rs_set['m_rst'], rs_set['m_ref'])
                total += recons_temp_smooth
                # print(recons_temp_smooth)
                # print("Recon_Temp_smooth")

            if "recons_joints" in self.losses:
                total += self._update_loss("recons_joints", rs_set['joints_rst'], rs_set['joints_ref'])

            if "recons_3D" in self.losses:
                total += self._update_loss("recons_3D", rs_set['hands_verts_rst'], rs_set['hands_verts_ref'])
            
            if "recons_velocity" in self.losses:
                nfeats = rs_set['m_rst'].shape[-1]
                if nfeats in [263, 135 + 263, 251]:
                    if nfeats == 135 + 263:
                        vel_start = 135 + 4
                    elif nfeats == 263 or nfeats == 251 :
                        vel_start = 4
                    total += self._update_loss(
                        "recons_velocity",
                        rs_set['m_rst'][..., vel_start:(self.num_joints - 1) * 3 +
                                        vel_start],
                        rs_set['m_ref'][..., vel_start:(self.num_joints - 1) * 3 +
                                        vel_start])
                elif nfeats == 66:
                    vel_rst = rs_set['m_rst'][:, 1:] - rs_set['m_rst'][:, :-1]
                    vel_ref = rs_set['m_ref'][:, 1:] - rs_set['m_ref'][:, :-1]
                    total += self._update_loss("recons_velocity", vel_rst,  vel_ref)
                else:
                    if self._params['recons_velocity'] != 0.0:
                        raise NotImplementedError(
                            "Velocity not implemented for nfeats = {})".format(nfeats))
            
            if "vq_commit" in self.losses:
                total += self._update_loss("vq_commit", rs_set['loss_commit'], rs_set['loss_commit'])
            
            if "vq_perplexity" in self.losses:
                self._update_loss("vq_perplexity", rs_set['perplexity'],rs_set['perplexity'])

            ## test the patch recon loss
            if "recons_patch" in self.losses:
                # print()
                # print("m_patch_rst", rs_set['m_patch_rst'].shape)
                # print()
                total += self._update_loss("recons_patch", 
                                           rs_set['m_patch_rst'],
                                           rs_set['m_patch_ref'])
                
                


        if self.stage in ["lm_pretrain", "lm_instruct"]:
            total += self._update_loss("gpt_loss", rs_set['outputs'].loss,
                                       rs_set['outputs'].loss)
            
        if self.stage in ["classifier"]:
            total += self._update_loss("cls_loss", rs_set['loss'],rs_set['loss'])

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total
