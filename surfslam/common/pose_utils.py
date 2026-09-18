"""
File: src/common/pose_utils.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""


import numpy as np
import pytorch3d.transforms
import torch
import pandas as pd
from scipy.spatial.transform import Rotation


## Converts a 4x4 transformation matrix to the se(3) twist vector
# Inspired by a similar NICE-SLAM function.
# @param transformation_matrix: A pytorch 4x4 homogenous transformation matrix
# @param device: The device for the output
# @returns: A 6-tensor [x,y,z,r_x,r_y,r_z]
def transform_to_tensor(transformation_matrix, device=None):

    gpu_id = -1
    if isinstance(transformation_matrix, np.ndarray):
        if transformation_matrix.get_device() != -1:
            if transformation_matrix.requires_grad:
                transformation_matrix = transformation_matrix.detach()
            transformation_matrix = transformation_matrix.detach().cpu()
            gpu_id = transformation_matrix.get_device()
        elif transformation_matrix.requires_grad:
            transformation_matrix = transformation_matrix.detach()
        transformation_matrix = transformation_matrix.numpy()
    elif not isinstance(transformation_matrix, torch.Tensor):
        raise ValueError((f"Invalid argument of type {type(transformation_matrix).__name__}"
                          "passed to transform_to_tensor (Expected numpy array or pytorch tensor)"))

    R = transformation_matrix[:3, :3]
    T = transformation_matrix[:3, 3]

    rot = pytorch3d.transforms.matrix_to_axis_angle(R)

    tensor = torch.cat([T, rot]).float()
    if device is not None:
        tensor = tensor.to(device)
    elif gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


## Converts a tensor produced by transform_to_tensor to a transformation matrix
# Inspired by a similar NICE-SLAM function.
# @param transformation_tensors: se(3) twist vectors
# @returns a 4x4 homogenous transformation matrix
def tensor_to_transform(transformation_tensors):

    N = len(transformation_tensors.shape)
    if N == 1:
        transformation_tensors = torch.unsqueeze(transformation_tensors, 0)
    Ts, rots = transformation_tensors[:, :3], transformation_tensors[:, 3:]
    rotation_matrices = pytorch3d.transforms.axis_angle_to_matrix(rots)
    RT = torch.cat([rotation_matrices, Ts[:, :, None]], 2)
    if N == 1:
        RT = RT[0]

    H_row = torch.zeros_like(RT[..., 0, :])
    H_row[..., 3] = 1
    RT = torch.cat((RT, H_row[..., None, :]), dim=-2)
    return RT

def build_poses_from_df(df: pd.DataFrame, zero_origin=False):
    data = torch.from_numpy(df.to_numpy(dtype=np.float64))

    ts = data[:,0]
    xyz = data[:,1:4]
    quat = data[:,4:]

    rots = torch.from_numpy(Rotation.from_quat(quat).as_matrix())
    
    poses = torch.cat((rots, xyz.unsqueeze(2)), dim=2)

    homog = torch.Tensor([0,0,0,1]).tile((poses.shape[0], 1, 1)).to(poses.device)

    poses = torch.cat((poses, homog), dim=1)

    if zero_origin:
        rot_inv = poses[0,:3,:3].T
        t_inv = -rot_inv @ poses[0,:3,3]
        start_inv = torch.hstack((rot_inv, t_inv.reshape(-1, 1)))
        start_inv = torch.vstack((start_inv, torch.tensor([0,0,0,1.0], device=start_inv.device)))
        poses = start_inv.unsqueeze(0) @ poses

    return poses.float(), ts
