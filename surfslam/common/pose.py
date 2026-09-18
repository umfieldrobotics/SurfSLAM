"""
File: src/common/pose.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

from dataclasses import dataclass
from typing import Optional, Union

import pytorch3d.transforms
import torch

from common.pose_utils import tensor_to_transform, transform_to_tensor


class Pose:
    """ Class to define a possibly optimizable pose.
    Poses are represented as a 7-tuple [x,y,z,q_x,q_y,q_z,q_w]
    """

    ## Constructor
    # @param transformation_matrix: 4D Homogenous transformation matrix to turn into a pose
    # @param pose_tensor: An alternate argument to @p transformation matrix, allows you to
    #                     specify the 6-vector representation directly
    # @param fixed: Specifies whether or not a gradient should be computed.
    def __init__(self, transformation_matrix: torch.Tensor = torch.eye(4),
                 pose_tensor: torch.Tensor = None,
                 fixed: bool = None,
                 requires_tensor: bool = False,
                 cov: torch.Tensor | None = None):

        if fixed is None:
            if transformation_matrix is None:
                fixed = not pose_tensor.requires_grad
            else:
                fixed = not transformation_matrix.requires_grad
        
        if pose_tensor is not None:
            self._pose_tensor = pose_tensor
            self._pose_tensor.requires_grad_(not fixed)
            transformation_matrix = tensor_to_transform(self._pose_tensor).float()
        elif requires_tensor:
            # We do this copy back and forth to support computing gradients on the
            # resulting pose tensor. 
            self._pose_tensor = transform_to_tensor(transformation_matrix).float()
            self._pose_tensor.requires_grad_(not fixed)
            transformation_matrix = tensor_to_transform(self._pose_tensor).float()
        else:
            self._pose_tensor = None
            transformation_matrix = transformation_matrix.float()

        self._transformation_matrix = transformation_matrix
        self._transformation_matrix.requires_grad_(not fixed)
        
        self.covariance = cov

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.get_transformation_matrix())

    ## Moves the pose in-place to the specified device, and @returns a reference to the current pose.
    def to(self, device: Union[str, int]) -> "Pose":
        if self._pose_tensor is not None:
            self._pose_tensor = self._pose_tensor.to(device)
        
        if self.covariance is not None:
            self.covariance = self.covariance.to(device)
            
        self._transformation_matrix = self._transformation_matrix.to(device)
        return self

    ## Returns a copy of the current pose that is detached from the computation graph.
    # @returns a new pose. 
    def detach(self) -> "Pose":
        return Pose(self.get_transformation_matrix().detach())

    ## Load in a setting dict of form {xyz: [x,y,z], "orientation": [w,x,y,z]} to a Pose
    # @returns a Pose representing the 
    def from_settings(pose_dict: dict, fixed: bool = None) -> "Pose":
        xyz = torch.Tensor(pose_dict['xyz'])
        quat = torch.Tensor(pose_dict['orientation'])
        cov = pose_dict.get('covariance', None)
        if cov is not None:
            cov = torch.Tensor(cov)

        axis_angle = pytorch3d.transforms.quaternion_to_axis_angle(quat)
        tensor = torch.cat((xyz, axis_angle))
                
        if 'fixed' in pose_dict and fixed is None:
            fixed = pose_dict['fixed']
        elif fixed is None:
            fixed = True
            
        return Pose(pose_tensor=tensor, fixed=fixed, cov=cov)
        
    ## Converts the current Pose to a dict and @returns the pose as a dict.
    def to_settings(self) -> dict:
        translation = self.get_translation().detach().cpu()
        xyz = [translation[i].item() for i in range(3)]

        quat = pytorch3d.transforms.matrix_to_quaternion(self.get_rotation().detach().cpu())
        
        if self.covariance is None:
            cov = None
        else:
            cov = self.covariance.tolist()

        fixed_status = not (self._pose_tensor is not None and self._pose_tensor.requires_grad)

        return {
            "xyz": xyz,
            "orientation": quat,
            "covariance": cov,
            "fixed": fixed_status,
        }

    ## @returns a copy of the current pose.
    def clone(self, fixed=None, requires_tensor=False) -> "Pose":
        if fixed is None:
            fixed = not self.get_transformation_matrix().requires_grad
        return Pose(self.get_transformation_matrix().clone(), fixed=fixed, 
                    requires_tensor=requires_tensor,
                    cov=None if self.covariance is None else self.covariance.clone())

    # Performs matrix multiplication on matrix representations of the given poses, and returns the result
    def __mul__(self, other: "Pose") -> "Pose":
        T = self.get_transformation_matrix() @ other.get_transformation_matrix()

        if self.covariance is None and other.covariance is None:
            return Pose(T)

        # Sigma = Sigma_1  +  Ad(T1) * Sigma_2 * Ad(T1)^T
        if self.covariance is None:
            Sigma = other.covariance.clone()
            # Transform only other’s cov
            Ad1 = self.adjoint()
            Sigma = Ad1 @ Sigma @ Ad1.transpose(0, 1)
        elif other.covariance is None:
            Sigma = self.covariance.clone()
        else:
            Ad1 = self.adjoint()
            Sigma = self.covariance.clone() + Ad1 @ other.covariance @ Ad1.transpose(0, 1)

        return Pose(T, cov=Sigma)

    def inv(self) -> "Pose":
        T_inv = self.get_transformation_matrix().inverse()

        if self.covariance is None:
            return Pose(T_inv)

        Ad_inv = Pose(T_inv).adjoint()
        Sigma_inv = Ad_inv @ self.covariance @ Ad_inv.transpose(0, 1)

        return Pose(T_inv, cov=Sigma_inv)

    ## Gets the matrix representation of the pose. Only pytorch operations are used, so gradients are preserved.
    # @returns a 4x4 homogenous transformation matrix
    def get_transformation_matrix(self) -> torch.Tensor:
        if self._pose_tensor is None or not self._pose_tensor.requires_grad:
            return self._transformation_matrix

        return tensor_to_transform(self._pose_tensor)

    ## Gets the underlying 7-tensor. Should basically never be used.
    def get_pose_tensor(self) -> torch.Tensor:
        if self._pose_tensor is None:
            self._pose_tensor = transform_to_tensor(self.get_transformation_matrix())
        return self._pose_tensor

    ## Gets the translation component of the pose
    def get_translation(self) -> torch.Tensor:
        if self._pose_tensor is not None:
            return self._pose_tensor[:3]
        return self.get_transformation_matrix()[:3, 3]

    ## Returns the rotation as a rotation matrix
    def get_rotation(self) -> torch.Tensor:
        return self.get_transformation_matrix()[:3,:3]

    def adjoint(self) -> torch.Tensor:
        T = self.get_transformation_matrix()
        R = T[:3, :3]
        t = T[:3, 3]

        tx = torch.tensor(
            [[0.0, -t[2], t[1]],
            [t[2], 0.0, -t[0]],
            [-t[1], t[0], 0.0]],
            dtype=R.dtype,
            device=R.device,
        )

        Adj = torch.zeros(6, 6, dtype=R.dtype, device=R.device)
        Adj[:3, :3] = R
        Adj[3:, 3:] = R
        Adj[3:, :3] = tx @ R

        return Adj
    
    
    
@dataclass
class RegistrationResult:
    timestamp_a: float
    timestamp_b: float
    T_a_b: Pose
    id_a: Optional[int] = None # asscoate keyframes to the timestamps 
    id_b: Optional[int] = None # asscoate keyframes to the timestamps
