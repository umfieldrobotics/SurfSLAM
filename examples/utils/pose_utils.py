"""
File: examples/utils.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import numpy as np
import torch
import pytorch3d.transforms

from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import os, sys


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)

from common.pose import Pose
from tbnms.tbnms_calibration import TBNMSCalibration

class TransformRecord:
    def __init__(self, t, xyz, quat):
        self.t = float(t)
        self.xyz = np.asarray(xyz, dtype=float)
        self.quat = np.asarray(quat, dtype=float)

class SimpleTransformBuffer:
    def __init__(self):
        self.records = []
        self._t = None
        self._interp_xyz = None
        self._slerp = None

    def insert(self, t, pose):
        xyz = pose[:3, 3]
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        self.records.append(TransformRecord(t, xyz, quat))

    def finalize(self):
        self.records.sort(key=lambda r: r.t)
        self._t = np.array([r.t for r in self.records])
        xyz = np.stack([r.xyz for r in self.records])
        quat = np.stack([r.quat for r in self.records])

        # translation interp
        self._interp_xyz = interp1d(
            self._t, xyz, axis=0, kind="linear", fill_value="extrapolate"
        )

        # rotation slerp
        key_rots = Rotation.from_quat(quat)
        self._slerp = Slerp(self._t, key_rots)

    def lookup(self, t, exact=False) -> torch.Tensor:
        if self._t is None:
            raise RuntimeError("Call finalize() before lookup().")
        t = float(t)
        
        # If exact mode is enabled, check if there's a close reference pose
        if exact:
            closest_diff = np.min(np.abs(self._t - t))
            if closest_diff > 0.1:
                return None
        
        xyz = self._interp_xyz(t)
        rot = self._slerp(t)
        R = rot.as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = xyz
        
        return torch.from_numpy(T)

def build_buffer_from_poses(poses, gt_timestamps):
    buf = SimpleTransformBuffer()
    for pose, ts in zip(poses, gt_timestamps):
        buf.insert(ts, pose)
    buf.finalize()
    return buf, list(map(float, gt_timestamps))


def msg_to_transformation_mat(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    xyz = torch.Tensor([trans.x, trans.y, trans.z]).reshape(3, 1)
    quat = torch.Tensor([rot.w, rot.x, rot.y, rot.z])
    rotmat = pytorch3d.transforms.quaternion_to_matrix(quat)
    
    T = torch.hstack((rotmat, xyz))
    T = torch.vstack((T, torch.Tensor([0, 0, 0, 1])))
    return T.float()

def load_calibration(dataset_family: str, calib_path: str):
    if dataset_family.lower() == 'tbnms':
        return TBNMSCalibration(calib_path)
    print("Warning: Supplied dataset has no calibration configured. Don't enable the camera!")
    return None