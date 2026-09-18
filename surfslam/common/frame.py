"""
File: src/common/frame.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

from typing import Union

import torch

from common.sensors import Image, StereoImage
from common.pose import Pose


class Frame:
    """Frame Class representing the atomic unit of optimization in Loner SLAM

    A Frame consists of an image, and lidar points that occured at nearby times.

    It also stores tensors used for computing the poses of cameras and lidars.
    """

    ## Constructor
    # @param image: The RGB Image
    # @param T_lidar_to_camera: Pose object with extrinsic calibration
    def __init__(
        self, stereo_image: StereoImage = None, T_imu_to_camera: Pose = None, frame_id=-1,
        depth_image=None
    ) -> None:

        self.stereo_image: StereoImage = stereo_image
        self.depth_image: Image | None = depth_image

        self._imu_to_camera: Pose = T_imu_to_camera

        self._camera_pose: Pose = None
        self._gt_camera_pose: Pose = None

        self._id = frame_id
        self._associated_imu_timestamp = None

    def get_id(self) -> int:
        return int(self._id)

    ## Creates a deepcopy of the frame.
    # @returns a deepcopy of the current frame
    def clone(self) -> "Frame":

        attrs = [
            "stereo_image",
            "depth_image",
            "_imu_to_camera",
            "_camera_pose",
            "_gt_camera_pose",
            "_associated_imu_timestamp",
        ]

        new_frame = Frame()
        for attr in attrs:
            old_attr = getattr(self, attr)
            new_attr = old_attr.clone() if hasattr(old_attr, 'clone') else old_attr
            setattr(new_frame, attr, new_attr)

        # Preserve the frame id for downstream consumers (e.g., registration)
        new_frame._id = self._id

        return new_frame

    def __str__(self):
        im_str = "None" if self.stereo_image is None else self.stereo_image.timestamp
        if isinstance(im_str, torch.Tensor):
            im_str = im_str.item()

        return f"<Frame {self._id}: (t={float(self.get_time()):.3f})>"

    def __repr__(self):
        return self.__str__()

    ## Moves all items in the frame to the specified device, in-place. Also returns the current frame.
    # @param device: Target device, as int (GPU) or string (CPU or GPU)
    def to(self, device: Union[int, str]) -> "Frame":
        if self.stereo_image is not None:
            self.stereo_image.to(device)

        for pose in [self._imu_to_camera, self._camera_pose, self._gt_camera_pose]:
            if pose is not None:
                pose.to(device)

        return self

    ## Detaches the current frame from the computation graph
    # @returns a reference to self.
    def detach(self) -> "Frame":
        self._camera_pose.detach()

        return self

    ## Gets the timestamp of the start of the scan
    def get_time(self) -> float:
        return self.stereo_image.timestamp

    ## @returns the Pose of the camera at the time the image was captured
    def get_camera_pose(self) -> Pose:
        return self._camera_pose

    def state_dict(self) -> dict:
        return {
            "camera_pose": (
                self._camera_pose.to_settings()
                if self._camera_pose is not None
                else None
            ),
            "gt_camera_pose": (
                self._gt_camera_pose.to_settings()
                if self._gt_camera_pose is not None
                else None
            ),
            "imu_to_camera": (
                self._imu_to_camera.to_settings()
                if self._imu_to_camera is not None
                else None
            ),
            "id": self._id,
            "stereo_image": (
                self.stereo_image.state_dict()
                if self.stereo_image is not None
                else None
            ),
            "associated_imu_timestamp": self._associated_imu_timestamp,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "Frame":
        frame = cls()
        frame._camera_pose = (
            Pose.from_settings(state_dict["camera_pose"])
            if state_dict["camera_pose"] is not None
            else None
        )
        frame._gt_camera_pose = (
            Pose.from_settings(state_dict["gt_camera_pose"])
            if state_dict["gt_camera_pose"] is not None
            else None
        )
        frame._imu_to_camera = (
            Pose.from_settings(state_dict["imu_to_camera"])
            if state_dict["imu_to_camera"] is not None
            else None
        )
        frame._id = state_dict["id"]
        frame.stereo_image = (
            StereoImage.from_state_dict(state_dict["stereo_image"])
            if state_dict["stereo_image"] is not None
            else None
        )
        frame._associated_imu_timestamp = state_dict.get("associated_imu_timestamp", None)
        return frame
