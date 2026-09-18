"""
File: src/tracking/frame_synthesis.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

from typing import List, Union


from common.frame import Frame
from common.pose import Pose
from common.sensors import StereoImage
from common.settings import Settings

class FrameSynthesis:
    """ FrameSynthesis class to process streams of data and create frames.
    """

    ## Constructor
    # @param settings: Settings for the tracker (which includes frame synthesis)
    def __init__(self, settings: Settings, T_imu_to_camera: Pose) -> None:
        self._settings = settings

        self._t_imu_to_camera = T_imu_to_camera
        self._t_camera_to_imu = self._t_imu_to_camera.inv()

        # Frames that are fully built and ready to be processed
        self._completed_frames = []

        # used for decimating images
        self._prev_accepted_timestamp = float('-inf')

        # Minimum dt between frames
        self._frame_delta_t_sec = self._settings.stereo_delta_t_sec

    ## Reads data from the @p lidar_scan and adds the points to the appropriate frame(s)
    # @param lidar_scan: Set of lidar points to add to the in-progress frames(s)
    def process_stereo(self, stereo_image: StereoImage, gt_pose: Pose, associated_imu_ts: float = None) -> List[int]:
        scan_time = stereo_image.timestamp
        dt = self._frame_delta_t_sec - self._settings.frame_delta_t_sec_tolerance # type: ignore
        if scan_time - self._prev_accepted_timestamp >= dt:
            new_frame = Frame(stereo_image, self._t_imu_to_camera)
            new_frame._gt_camera_pose = gt_pose
            new_frame._associated_imu_timestamp = associated_imu_ts
            self._completed_frames.append(new_frame.clone())
            self._prev_accepted_timestamp = stereo_image.timestamp

    ## Check if a frame exists to be returned
    def has_frame(self) -> bool:
        return len(self._completed_frames) != 0

    ## Return and remove a newly synthesized Frame. If unavailable, returns None.
    def pop_frame(self) -> Union[Frame, None]:

        # note: this is done with queues to avoid potentially expensive copies
        # which would be needed to avoid active_frame getting overwritten.
        if len(self._completed_frames) == 0:
            return None

        return self._completed_frames.pop(0)

    def peek_frame(self) -> Union[Frame, None]:
        if len(self._completed_frames) == 0:
            return None

        return self._completed_frames[0]
