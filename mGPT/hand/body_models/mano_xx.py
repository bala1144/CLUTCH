from mGPT.hand.body_models.smplx_custom import *
import mGPT.hand.body_models.smplx_custom as smplx
import pickle
from mGPT.hand.utils.transformations import aa2d6, d62aa, rotmat2d6
from mGPT.utils.assets import get_asset_path, get_mano_model_path
import trimesh
import torch

to_cpu = lambda tensor: tensor.detach().cpu().numpy()
to_numpy = lambda tensor: tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
to_torch = lambda array, dtype: torch.from_numpy(array).to(dtype) if isinstance(array, np.ndarray) else array.to(dtype)

def batch_to_mano_full_pose(batch):

    batch_size, T =  batch["rhand_global_orientaion"].shape[0],  batch["rhand_global_orientaion"].shape[1]
    input_params = torch.cat(
            [
            batch["rhand_global_orientaion"].reshape(batch_size, T, -1),# Bs x T x 6 
            batch["rhand_pose"].reshape(batch_size, T, -1),  # Bs, T x 90 (15J x 6)
            batch["rhand_transl"].reshape(batch_size, T, -1),# Bs x T x 3
            
            batch["lhand_global_orientaion"].reshape(batch_size, T, -1),  # Bs x T x 6
            batch[ "lhand_pose"].reshape(batch_size, T, -1), # Bs, T x 90 (15J x 6)
            batch["lhand_transl"].reshape(batch_size, T, -1) # Bs x T x 3
            ],
            dim=-1
        )

    # T x 198
    return input_params
    
def mano_full_pose_to_mano_params(pred_params):

    batch_size, T =  pred_params.shape[0], pred_params.shape[1]

    if pred_params.shape[-1] == 198:

        rhand_global_orientaion = pred_params[:, :, :6].reshape(batch_size, T, 6) # Bs, T x 15J x 6
        rhand_pose = pred_params[:, :, 6:96].reshape(batch_size, T, -1, 6)
        rhand_transl = pred_params[:, :, 96:99].reshape(batch_size, T, 3)

        l_hand = pred_params[:, :, 99:]
        lhand_global_orientaion = l_hand[:, :, 0:6].reshape(batch_size, T, 6) # Bs, T x 15J x 6
        lhand_pose = l_hand[:, :, 6:96].reshape(batch_size, T, -1, 6)
        lhand_transl = l_hand[:, :, 96:99].reshape(batch_size, T, 3)

    elif  pred_params.shape[-1] == 66:

        rhand_global_orientaion = pred_params[:, :, :6].reshape(batch_size, T, -1, 6) # Bs, T x 15J x 6
        rhand_pose = pred_params[:, :, 6:30].reshape(batch_size, T, 24)
        rhand_transl = pred_params[:, :, 30:33].reshape(batch_size, T, 3)

        l_hand = pred_params[:, :, 33:]
        lhand_global_orientaion = l_hand[:, :, 0:6].reshape(batch_size, T, -1, 6) # Bs, T x 15J x 6
        lhand_pose = l_hand[:, :, 6:30].reshape(batch_size, T, 24)
        lhand_transl = l_hand[:, :, 30:33].reshape(batch_size, T, 3)
    

    elif  pred_params.shape[-1] == 75:

        rhand_global_orientaion = pred_params[:, :, :6].reshape(batch_size, T, -1, 6) # Bs, T x 15J x 6
        rhand_pose = pred_params[:, :, 6:30].reshape(batch_size, T, 24)
        rhand_transl = pred_params[:, :, 30:33].reshape(batch_size, T, 3)

        l_hand = pred_params[:, :, 33:]
        lhand_global_orientaion = l_hand[:, :, 0:6].reshape(batch_size, T, -1, 6) # Bs, T x 15J x 6
        lhand_pose = l_hand[:, :, 6:30].reshape(batch_size, T, 24)
        lhand_transl = l_hand[:, :, 30:33].reshape(batch_size, T, 3)

        obj_pose = pred_params[:, :, 66:] # Bs, T x 15J x 6
        global_orient_obj = obj_pose[:, :, :6].reshape(batch_size, T, 6 )
        transl_obj = obj_pose[:, :, 6:].reshape(batch_size, T, 3)

        scene_params = {
        "rhand_global_orientaion": rhand_global_orientaion,
        "rhand_pose": rhand_pose,
        "rhand_transl": rhand_transl,

        "lhand_global_orientaion": lhand_global_orientaion,
        "lhand_pose": lhand_pose,
        "lhand_transl": lhand_transl,

        "global_orient_obj": global_orient_obj,
        "transl_obj": transl_obj

        }  

        return scene_params
        


    mano_parmas = {
        "rhand_global_orientaion": rhand_global_orientaion,
        "rhand_pose": rhand_pose,
        "rhand_transl": rhand_transl,
        "lhand_global_orientaion": lhand_global_orientaion,
        "lhand_pose": lhand_pose,
        "lhand_transl": lhand_transl
    }    
    return mano_parmas


class MANO_doubleX(torch.nn.Module):
    def __init__(self,
                model_path=get_mano_model_path(),
                use_pca=True,
                flat_hand_mean=True,
                batch_size=20,
                num_pca_comps=45,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 **kwargs):
        super().__init__()

        self.num_pca_comps = num_pca_comps
        self.device = device

        self.right_hand = smplx.create(model_path=model_path,
                             model_type='mano',
                             num_pca_comps=num_pca_comps,
                             use_pca=use_pca,
                             flat_hand_mean=flat_hand_mean,
                             batch_size=batch_size,
                             is_rhand=True,
                             **kwargs)

        # self.left_hand = MANOLayer(model_path=os.path.join(model_path,"mano"),
        self.left_hand = smplx.create(model_path=model_path,
                        model_type='mano',
                        num_pca_comps=num_pca_comps,
                        use_pca=use_pca,
                        flat_hand_mean=flat_hand_mean,
                        is_rhand=False,
                        batch_size=batch_size, **kwargs).to(device)
        
        self.right_hand = self.right_hand.to(device)
        self.left_hand = self.left_hand.to(device)
        self.faces = self.load_doublex_faces()
        self.use_pca = use_pca
    
    def load_doublex_faces(self):
        faces_path = get_asset_path("Mano_doublex_faces_water_tight.npy")
        faces = np.load(faces_path, allow_pickle=True).item()["faces"]
        return faces
    

    def forward(self, rhand_global_orientaion=None, rhand_transl=None, rhand_pose=None, lhand_global_orientaion=None, lhand_transl=None, lhand_pose=None, **kwargs):
        """
        model prediction are mostly in 6D space
        """

        Bs, T, _ = lhand_transl.shape

        if rhand_pose.shape[-1] == 90 or not self.use_pca: # this is bug version, just keeping it here for legacy
            rhand_pose = d62aa(rhand_pose).reshape(-1, 15*3).to(self.device)
            lhand_pose = d62aa(lhand_pose).reshape(-1, 15*3).to(self.device)
        else:
            rhand_pose = rhand_pose.reshape(-1,self.num_pca_comps).to(self.device)
            lhand_pose = lhand_pose.reshape(-1,self.num_pca_comps).to(self.device)


        rh_output = self.right_hand.batch_forward( 
            hand_pose=rhand_pose,
            global_orient=d62aa(rhand_global_orientaion).reshape(-1,3).to(self.device),
            transl=rhand_transl.reshape(-1,3).to(self.device),
        )
        rh_output.vertices = rh_output.vertices.reshape(Bs, T, 778, 3)
        rh_output.joints = rh_output.joints.reshape(Bs, T, -1, 3)
        rh_output.joints_w_tip = rh_output.joints_w_tip.reshape(Bs, T, -1, 3)
        
        
        lh_output = self.left_hand.batch_forward(
            hand_pose=lhand_pose,
            global_orient=d62aa(lhand_global_orientaion).reshape(-1,3).to(self.device),
            transl=lhand_transl.reshape(-1,3).to(self.device),
        )

        lh_output.vertices = lh_output.vertices.reshape(Bs, T, 778, 3)
        lh_output.joints = lh_output.joints.reshape(Bs, T, -1, 3)
        lh_output.joints_w_tip = lh_output.joints_w_tip.reshape(Bs, T, -1, 3)
    

        return lh_output, rh_output


    def get_scene_verts_from_batch(self, rhand_global_orientaion=None, rhand_transl=None, rhand_pose=None, lhand_global_orientaion=None, lhand_transl=None, lhand_pose=None, **kwargs):
        """
        model prediction are mostly in 6D space
        """

        Bs, T, _ = lhand_transl.shape

        ### 90 = 15 x 6D
        if rhand_pose.shape[-1] == 90 or not self.use_pca: # this is bug version, just keeping it here for legacy
            rhand_pose = d62aa(rhand_pose).reshape(-1, 15*3).to(self.device)
            lhand_pose = d62aa(lhand_pose).reshape(-1, 15*3).to(self.device)
        else:
            rhand_pose = rhand_pose.reshape(-1,self.num_pca_comps).to(self.device)
            lhand_pose = lhand_pose.reshape(-1,self.num_pca_comps).to(self.device)

        rh_output = self.right_hand.batch_forward( 
            hand_pose=rhand_pose,
            global_orient=d62aa(rhand_global_orientaion).reshape(-1,3).to(self.device),
            transl=rhand_transl.reshape(-1,3).to(self.device),
        )
        
        lh_output = self.left_hand.batch_forward(
            hand_pose=lhand_pose,
            global_orient=d62aa(lhand_global_orientaion).reshape(-1,3).to(self.device),
            transl=lhand_transl.reshape(-1,3).to(self.device),
        )

        verts = np.concatenate([to_cpu(lh_output.vertices), to_cpu(rh_output.vertices)], axis=1)
        return verts.reshape(Bs, T, 2 * 778, 3)


    def get_scene_verts_from_payload(self, **kwargs):
        device = self.device
        if kwargs.get("rh_hand_pose", None) is not None: ## rthis is a bug version keeping it here for legacy
            rh_output = self.right_hand.batch_forward( 
                hand_pose=d62aa(kwargs['rh_hand_pose']).to(device=device).reshape(-1,45),
                global_orient=d62aa(kwargs['rh_global_orient']).to(device=device).reshape(-1,3),
                transl=to_torch(kwargs['rh_transl'], torch.float32).to(device=device),
            )
            
            lh_output = self.left_hand.batch_forward(
                    hand_pose=d62aa(kwargs['lh_hand_pose']).to(device=device).reshape(-1,45),
                    global_orient=d62aa(kwargs['lh_global_orient']).to(device=device).reshape(-1,3),
                    transl=to_torch(kwargs['lh_transl'], torch.float32).to(device=device),
            )
        else:
            rh_output = self.right_hand.batch_forward( 
                hand_pose=kwargs['rhand_pose'].to(device=device).reshape(-1,self.num_pca_comps),
                global_orient=d62aa(kwargs['rhand_global_orientaion']).to(device=device).reshape(-1,3),
                transl=to_torch(kwargs['rhand_transl'], torch.float32).to(device=device),
            )
            
            lh_output = self.left_hand.batch_forward(
                    hand_pose=kwargs['lhand_pose'].to(device=device).reshape(-1,self.num_pca_comps),
                    global_orient=d62aa(kwargs['lhand_global_orientaion']).to(device=device).reshape(-1,3),
                    transl=to_torch(kwargs['lhand_transl'], torch.float32).to(device=device),
            )

        verts = np.concatenate([to_cpu(lh_output.vertices), to_cpu(rh_output.vertices)], axis=1)
        return verts
    
    def get_hands_from_payload(self, **kwargs):
        device = self.device
        if kwargs.get("rh_hand_pose", None) is not None: ## rthis is a bug version keeping it here for legacy
            rh_output = self.right_hand( 
                hand_pose=d62aa(kwargs['rh_hand_pose']).to(device=device).reshape(-1,45),
                global_orient=d62aa(kwargs['rh_global_orient']).to(device=device).reshape(-1,3),
                transl=to_torch(kwargs['rh_transl'], torch.float32).to(device=device),
            )
            
            lh_output = self.left_hand(
                    hand_pose=d62aa(kwargs['lh_hand_pose']).to(device=device).reshape(-1,45),
                    global_orient=d62aa(kwargs['lh_global_orient']).to(device=device).reshape(-1,3),
                    transl=to_torch(kwargs['lh_transl'], torch.float32).to(device=device),
            )
        else:
            rh_output = self.right_hand( 
                hand_pose=kwargs['rhand_pose'].to(device=device).reshape(-1,self.num_pca_comps),
                global_orient=d62aa(kwargs['rhand_global_orientaion']).to(device=device).reshape(-1,3),
                transl=to_torch(kwargs['rhand_transl'], torch.float32).to(device=device).reshape(-1,3),
            )
            
            lh_output = self.left_hand(
                    hand_pose=kwargs['lhand_pose'].to(device=device).reshape(-1,self.num_pca_comps),
                    global_orient=d62aa(kwargs['lhand_global_orientaion']).to(device=device).reshape(-1,3),
                    transl=to_torch(kwargs['lhand_transl'], torch.float32).to(device=device).reshape(-1,3),
            )

        return lh_output, rh_output

    def convert_pca_to_jt_rot_in_6D(self, hand_pose, right_hand=True):

        if not torch.is_tensor(hand_pose):
            hand_pose = torch.from_numpy(hand_pose).float().to(self.device)
        
        if right_hand:
            hand_pose_aa = torch.einsum('bi,ij->bj', [hand_pose, self.right_hand.hand_components])
        else:
            hand_pose_aa = torch.einsum('bi,ij->bj', [hand_pose, self.left_hand.hand_components])

        hand_pose_6D = aa2d6(hand_pose_aa.reshape(-1, 15, 3)).reshape(hand_pose.shape[0], 15, 6)
        return hand_pose_6D


    # def convert_jt_rot_in_6D_to_pca(self, hand_pose_6D, right_hand=True):

    #     if not torch.is_tensor(hand_pose_6D):
    #         hand_pose_6D = torch.from_numpy(hand_pose_6D).float().to(self.device)

    #     if hand_pose_6D.shape[-1] // 15 == 3:
    #         hand_pose_in_aa = d62aa(hand_pose_6D.reshape(-1, 15, 6)).reshape(-1, 15*3)
    #     else:
    #         hand_pose_in_aa = hand_pose_6D.reshape(-1, 15*3)
        

    #     if right_hand:
    #         hand_pose_in_aa = hand_pose_in_aa - self.right_hand.pose_mean.view(1, 48)[:, -45:]
    #         # hand_pose_pca = torch.einsum('bj,ji->bi', [hand_pose_in_aa, self.right_hand.hand_components.T])
    #         hand_pose_pca = torch.einsum('bi,ji->bj', hand_pose_in_aa, self.right_hand.hand_components)
    #     else:
    #         hand_pose_in_aa = hand_pose_in_aa - self.left_hand.pose_mean.view(1, 48)[:, -45:]
    #         # hand_pose_pca = torch.einsum('bj,ji->bi', [hand_pose_in_aa, self.left_hand.hand_components.T])

    #         hand_pose_pca = torch.einsum('bi,ji->bj', hand_pose_in_aa, self.left_hand.hand_components)

    #     # hand_pose_6D = aa2d6(hand_pose_aa.reshape(-1, 15, 3)).reshape(hand_pose.shape[0], 15, 6)
        
    #     return hand_pose_pca

    def convert_jt_rot_in_6D_to_pca(self, hand_pose_6D, right_hand=True):
        if not torch.is_tensor(hand_pose_6D):
            hand_pose_6D = torch.from_numpy(hand_pose_6D).float().to(self.device)

        B = hand_pose_6D.shape[0]

        # Convert from 6D to axis-angle
        if hand_pose_6D.shape[-1] == 6 * 15 or hand_pose_6D.shape[-1] == 6:
            hand_pose_in_aa = d62aa(hand_pose_6D.reshape(-1, 15, 6)).reshape(B, -1)  # [B, 45]
        else:
            hand_pose_in_aa = hand_pose_6D.reshape(B, -1)

        if right_hand:
            pose_mean_hand = self.right_hand.pose_mean.reshape(-1)[3:].view(1, 45)  # [1, 45]
            centered = hand_pose_in_aa - pose_mean_hand
            hand_pose_pca = torch.einsum('bi,ji->bj', centered, self.right_hand.hand_components)  # [B, N_PCA]
        else:
            pose_mean_hand = self.left_hand.pose_mean.reshape(-1)[3:].view(1, 45)
            centered = hand_pose_in_aa - pose_mean_hand
            hand_pose_pca = torch.einsum('bi,ji->bj', centered, self.left_hand.hand_components)

        return hand_pose_pca
