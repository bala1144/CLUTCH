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

        rhand_global_orientaion = pred_params[:, :, :6].reshape(batch_size, T, -1, 6) # Bs, T x 15J x 6
        rhand_pose = pred_params[:, :, 6:96].reshape(batch_size, T, -1, 6)
        rhand_transl = pred_params[:, :, 96:99].reshape(batch_size, T, 3)

        l_hand = pred_params[:, :, 99:]
        lhand_global_orientaion = l_hand[:, :, 0:6].reshape(batch_size, T, -1, 6) # Bs, T x 15J x 6
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


class MANO_doubleX(smplx.MANO):
    def __init__(self,
                model_path=get_mano_model_path(),
                use_pca=True,
                flat_hand_mean=True,
                batch_size=20,
                num_pca_comps=45,
                 **kwargs):
        super(MANO_doubleX, self).__init__(model_path+"/mano", **kwargs)

        self.num_pca_comps = num_pca_comps
        self.right_hand = smplx.create(model_path=model_path,
                             model_type='mano',
                             num_pca_comps=num_pca_comps,
                             use_pca=use_pca,
                             flat_hand_mean=flat_hand_mean,
                             batch_size=batch_size,
                             is_rhand=True,
                             **kwargs)

        self.left_hand = smplx.create(model_path=model_path,
                        model_type='mano',
                        num_pca_comps=num_pca_comps,
                        use_pca=use_pca,
                        flat_hand_mean=flat_hand_mean,
                        is_rhand=False,
                        batch_size=batch_size, **kwargs)
        
        self.faces = self.load_doublex_faces()

    def name(self) -> str:
        return 'MANO_xx'
    
    def load_doublex_faces(self):
        faces_path = get_asset_path("Mano_doublex_face.npy")
        faces = np.load(faces_path, allow_pickle=True).item()["faces"]
        return faces

    def forward(self, rhand_global_orientaion=None, rhand_transl=None, rhand_pose=None, lhand_global_orientaion=None, lhand_transl=None, lhand_pose=None, **kwargs):
        """
        model prediction are mostly in 6D space
        """

        Bs, T, _ = lhand_transl.shape

        if rhand_pose.shape[-1] == 90: # this is bug version, just keeping it here for legacy
            rhand_pose = d62aa(rhand_pose)
            lhand_pose = d62aa(lhand_pose)
    
        rh_output = self.right_hand.batch_forward( 
            hand_pose=rhand_pose.reshape(-1,self.num_pca_comps),
            global_orient=d62aa(rhand_global_orientaion).reshape(-1,3),
            transl=rhand_transl.reshape(-1,3),
        )
        rh_output.vertices = rh_output.vertices.reshape(Bs, T, 778, 3)
        rh_output.joints = rh_output.joints.reshape(Bs, T, -1, 3)
        rh_output.joints_w_tip = rh_output.joints_w_tip.reshape(Bs, T, -1, 3)
        
        
        lh_output = self.left_hand.batch_forward(
            hand_pose=lhand_pose.reshape(-1,self.num_pca_comps),
            global_orient=d62aa(lhand_global_orientaion).reshape(-1,3),
            transl=lhand_transl.reshape(-1,3),
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

        if rhand_pose.shape[-1] == 90: # this is bug version, just keeping it here for legacy
            rhand_pose = d62aa(rhand_pose)
            lhand_pose = d62aa(lhand_pose)
    
        rh_output = self.right_hand( 
            hand_pose=rhand_pose.reshape(-1,self.num_pca_comps),
            global_orient=d62aa(rhand_global_orientaion).reshape(-1,3),
            transl=rhand_transl.reshape(-1,3),
        )
        
        lh_output = self.left_hand(
            hand_pose=lhand_pose.reshape(-1,self.num_pca_comps),
            global_orient=d62aa(lhand_global_orientaion).reshape(-1,3),
            transl=lhand_transl.reshape(-1,3),
        )

        verts = np.concatenate([to_cpu(lh_output.vertices), to_cpu(rh_output.vertices)], axis=1)
        return verts.reshape(Bs, T, 2 * 778, 3)


    def get_scene_verts_from_payload(self, **kwargs):


        if kwargs.get("rh_hand_pose", None) is not None: ## rthis is a bug version keeping it here for legacy
            rh_output = self.right_hand( 
                hand_pose=d62aa(kwargs['rh_hand_pose']).reshape(-1,45),
                global_orient=d62aa(kwargs['rh_global_orient']).reshape(-1,3),
                transl=to_torch(kwargs['rh_transl'], torch.float32),
            )
            
            lh_output = self.left_hand(
                    hand_pose=d62aa(kwargs['lh_hand_pose']).reshape(-1,45),
                    global_orient=d62aa(kwargs['lh_global_orient']).reshape(-1,3),
                    transl=to_torch(kwargs['lh_transl'], torch.float32),
            )
        else:
            rh_output = self.right_hand( 
                hand_pose=kwargs['rhand_pose'].reshape(-1,self.num_pca_comps),
                global_orient=d62aa(kwargs['rhand_global_orientaion']).reshape(-1,3),
                transl=to_torch(kwargs['rhand_transl'], torch.float32),
            )
            
            lh_output = self.left_hand(
                    hand_pose=kwargs['lhand_pose'].reshape(-1,self.num_pca_comps),
                    global_orient=d62aa(kwargs['lhand_global_orientaion']).reshape(-1,3),
                    transl=to_torch(kwargs['lhand_transl'], torch.float32),
            )

        verts = np.concatenate([to_cpu(lh_output.vertices), to_cpu(rh_output.vertices)], axis=1)
        return verts
    
    def get_hands_from_payload(self, **kwargs):


        if kwargs.get("rh_hand_pose", None) is not None: ## rthis is a bug version keeping it here for legacy
            rh_output = self.right_hand( 
                hand_pose=d62aa(kwargs['rh_hand_pose']).reshape(-1,45),
                global_orient=d62aa(kwargs['rh_global_orient']).reshape(-1,3),
                transl=to_torch(kwargs['rh_transl'], torch.float32),
            )
            
            lh_output = self.left_hand(
                    hand_pose=d62aa(kwargs['lh_hand_pose']).reshape(-1,45),
                    global_orient=d62aa(kwargs['lh_global_orient']).reshape(-1,3),
                    transl=to_torch(kwargs['lh_transl'], torch.float32),
            )
        else:
            rh_output = self.right_hand( 
                hand_pose=kwargs['rhand_pose'].reshape(-1,self.num_pca_comps),
                global_orient=d62aa(kwargs['rhand_global_orientaion']).reshape(-1,3),
                transl=to_torch(kwargs['rhand_transl'], torch.float32).reshape(-1,3),
            )
            
            lh_output = self.left_hand(
                    hand_pose=kwargs['lhand_pose'].reshape(-1,self.num_pca_comps),
                    global_orient=d62aa(kwargs['lhand_global_orientaion']).reshape(-1,3),
                    transl=to_torch(kwargs['lhand_transl'], torch.float32).reshape(-1,3),
            )

        return lh_output, rh_output
