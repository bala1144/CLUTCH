import trimesh
import numpy as np
from mGPT.hand.utils.temporal_trimesh import temporal_trimesh


def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    
    return volume

def compute_intersect_vox_on_seq(lh_mesh:temporal_trimesh, rh_mesh:temporal_trimesh, obj_mesh:temporal_trimesh):

    assert len(lh_mesh) == len(rh_mesh) == len(obj_mesh)

    rh_vols = [intersect_vox(obj_mesh[i], rh_mesh[i]) for i in range(len(obj_mesh))]
    lh_vols = [intersect_vox(obj_mesh[i], lh_mesh[i]) for i in range(len(obj_mesh))]

    out_dict = {}
    out_dict["sum"] = sum(rh_vols + lh_vols)
    out_dict["mean"] = np.mean(rh_vols+lh_vols)
    all_vols = rh_vols + rh_vols
    non_zero_vols = [vol+ 1e-12 for vol in all_vols if vol > 0.0000000001]
    out_dict["non_zero_vols"] = np.mean(non_zero_vols)
    
    return out_dict

##################################################
######### Compute with depth  ####################
##################################################
def intersect_vox_with_depth(obj_mesh, hand_mesh, pitch=0.01):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)

    if volume > 1e-8:
        (result_close, result_distance, _, ) = trimesh.proximity.closest_point(obj_mesh, hand_mesh.vertices)
        max_depth = result_distance.max()
    else:
        max_depth = 0
    
    return volume, max_depth

def compute_intersect_vox_with_depth_on_seq(lh_mesh:temporal_trimesh, rh_mesh:temporal_trimesh, obj_mesh:temporal_trimesh, pitch:float=0.01, scale:float=1.0):

    assert len(lh_mesh) == len(rh_mesh) == len(obj_mesh)

    # rh_vols_with_depth = [intersect_vox_with_depth(obj_mesh[i], rh_mesh[i]) for i in range(len(obj_mesh))]
    # lh_vols_with_depth = [intersect_vox_with_depth(obj_mesh[i], lh_mesh[i]) for i in range(len(obj_mesh))]

    rh_vols_with_depth = [intersect_vox_with_depth(obj_mesh[i], rh_mesh[i], pitch) for i in range(len(obj_mesh))]
    lh_vols_with_depth = [intersect_vox_with_depth(obj_mesh[i], lh_mesh[i], pitch) for i in range(len(obj_mesh))]


    ### compute penetration
    rh_vols  = [vol[0] for vol in rh_vols_with_depth]
    lh_vols  = [vol[0] for vol in lh_vols_with_depth]

    out_dict = {}
    # out_dict["sum_vol"] = sum(rh_vols + lh_vols)
    out_dict["mean_vol"] = np.mean(rh_vols+lh_vols)
    all_vols = rh_vols + rh_vols
    non_zero_vols = [vol+ 1e-12 for vol in all_vols if vol > 0.0000000001]
    # out_dict["non_zero_vols"] = np.mean(non_zero_vols)

    ### compute depth
    rh_depth  = [vol[1] for vol in rh_vols_with_depth]
    lh_depth  = [vol[1] for vol in lh_vols_with_depth]

    # out_dict["sum_depth"] = sum(rh_depth + lh_depth)
    out_dict["mean_depth"] = np.mean(rh_depth+lh_depth)
    all_depths = rh_depth + lh_depth
    non_zero_depths = [depth + 1e-12 for depth in all_depths if depth > 0.0000000001]
    # out_dict["non_zero_depth"] = np.mean(non_zero_depths)
    
    return out_dict


########### Max interpenetration depth  #################
def max_interpenetration_depth(mesh1, mesh2):
    """Compute the maximum interpenetration depth between two meshes."""

    # Compute the closest points on mesh2 for each vertex of mesh1
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(mesh2, mesh1.vertices)

    # Find the vertices of mesh1 that are inside mesh2
    inside_mesh2 = distances < 0

    # Compute the maximum penetration depth
    if np.any(inside_mesh2):
        max_depth = -np.min(distances[inside_mesh2])
    else:
        max_depth = 0.0

    return max_depth


def max_interpenetration_depth_on_seq(lh_mesh:temporal_trimesh, rh_mesh:temporal_trimesh, obj_mesh:temporal_trimesh):

    assert len(lh_mesh) == len(rh_mesh) == len(obj_mesh)

    ### h2o
    rh_depth = [max_interpenetration_depth(obj_mesh[i], rh_mesh[i]) for i in range(len(obj_mesh))]
    lh_depth = [max_interpenetration_depth(obj_mesh[i], lh_mesh[i]) for i in range(len(obj_mesh))]

    out_dict = {}
    all_depths = rh_depth + lh_depth
    out_dict["mean_o2h"] = np.mean(rh_depth+lh_depth)
    nz_depths = [vol+ 1e-12 for vol in all_depths if vol > 0.0000000001]
    out_dict["nz_o2h"] = np.mean(nz_depths)
    
    ### o2h
    rh_depth = [max_interpenetration_depth(rh_mesh[i], obj_mesh[i]) for i in range(len(obj_mesh))]
    lh_depth = [max_interpenetration_depth(lh_mesh[i], obj_mesh[i]) for i in range(len(obj_mesh))]

    all_depths = rh_depth + lh_depth
    out_dict["mean_h2o"] = np.mean(rh_depth+lh_depth)
    nz_depths = [vol+ 1e-12 for vol in all_depths if vol > 0.0000000001]
    out_dict["nz_h2o"] = np.mean(nz_depths)
    
    return out_dict
