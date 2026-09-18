"""
File: src/tracking/tracker.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

import torch
import torch.multiprocessing as mp

from common.frame import Frame
from common.pose import Pose, RegistrationResult
from common.sensors import StereoImage, BackendData
from common.settings import Settings
from common.signals import Signal, StopSignal
from tracking.frame_synthesis import FrameSynthesis
from common.shared_state import SharedState

from tracking.backend import Backend
from scipy.spatial.transform import Rotation


class Tracker:
    """Tracker: Top-level Tracking module

    Given streams of RGB data and lidar scans, this class is responsible for creating
    Frame instances and estimating their poses.
    """

    # Constructor
    # @param settings: Top level settings for the entire Loner SLAM module. Needed for calib etc.
    # @param rgb_signal: A Signal which the Tracker creates a Slot for, used for fetching RGB frames
    # @param lidar_signal: Same as rgb_signal, but for lidar
    # @param frame_queue: A Signal which the Tracker emits to when it completes a frame
    def __init__(
        self,
        settings: Settings,
        stereo_signal: Signal,
        backend_signal: Signal,
        keyframe_signal: Signal,
        registration_result_signal: Signal,
    ) -> None:

        self._stereo_slot = stereo_signal.register()
        self._backend_slot = backend_signal.register()
        self._registration_result_signal = registration_result_signal.register()

        self._keyframe_signal = keyframe_signal
        self._settings = settings
        self._tracker_settings = settings.tracker

        self._t_imu_to_camera = Pose.from_settings(
            settings.calibration.imu_to_camera)

        self._keyframe_synthesizer = FrameSynthesis(
            self._tracker_settings.frame_synthesis, self._t_imu_to_camera
        )

        self._kf_stereo_to_imu_lookup: Dict[int, float] = {}

        # Used to indicate to an external process that I've processed the stop signal
        self._processed_stop_signal = mp.Value("i", 0)
        # Set to 0 from an external thread when it's time to actuall exit.
        self._term_signal = mp.Value("i", 0)

        # Track stop signals received from each slot
        self._backend_slot_stopped = False
        self._stereo_slot_stopped = False
        self._registration_slot_stopped = False
        self._emitted_downstream_stop = False  # Track if we've sent stop to keyframe_signal

        # Used for frame-to-frame ICP tracking
        self._reference_point_cloud = None
        self._reference_pose = Pose(fixed=True)
        self._reference_time = None

        self._frame_count = 0
        self._last_mapped_frame_time = None
        self._last_tracked_frame_time = 0
        
        self._keyframes: Dict[int, Frame] = {}

        cfg_path = (
            Path(os.path.dirname(__file__)).parent.parent
            / "cfg"
            / self._tracker_settings.backend_config_path)
        
        live_log_file = f"{self._tracker_settings.log_directory}/trajectory/surfslam_live.txt"
        os.makedirs(os.path.dirname(live_log_file))
        self._backend = Backend(
            cfg_path,
            live_log_file=live_log_file,
            traj_pose_queue_max_size=self._tracker_settings.get(
                "traj_pose_queue_max_size", None),
        )

        # TODO: fill this in
        self._t_imu_lidar = torch.eye(4)

        self._ready = False

        self._keyframe_times = []
        self._keyframe_time_tolerance = 0.05  # 50ms tolerance for timestamp matching

        # Registration trajectory tracking
        self._registration_trajectory_file = (
            f"{self._tracker_settings.log_directory}/registration_trajectory.txt"
        )
        # timestamp -> (T_world_camera, source: 'registration'|'turtlemap')
        self._registration_trajectory = {}

    def start_backend(self):
        self._backend.start()

    def save_backend_traj(self, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self._backend.save_traj(fname)

    def is_graph_initialized(self) -> bool:
        """Check if backend graph is initialized and ready for stereo processing."""
        return self._backend.is_graph_initialized()

    def update(self):

        tic = time.time()
        num_tracked = 0
        # Share backend init status if a shared state has been provided
        if hasattr(self, "_shared_state") and self._shared_state is not None:
            self._shared_state.backend_initialized.value = int(
                self._backend.is_initialized()
            )

        if self._processed_stop_signal.value:
            print("Not updating tracker: Tracker already done.")

        if self._backend_slot.has_value():
            n = len(self._backend_slot)
            for _ in range(n):
                data: BackendData = self._backend_slot.get_value()
                if data is None:  # Frame was dropped due to age
                    continue

                if isinstance(data, StopSignal):
                    print("Tracker got stop in TurltMap Slot")
                    self._backend_slot_stopped = True
                    if not self._emitted_downstream_stop:
                        self._keyframe_signal.emit(StopSignal())
                        self._emitted_downstream_stop = True
                    continue

                self._backend.process(data)
                ts = data.timestamp
                if (
                    self._tracker_settings.minimum_frame_time is None
                    or ts >= self._tracker_settings.minimum_frame_time
                ):
                    self._ready = True

        if self._stereo_slot.has_value():
            val: Tuple[StereoImage, Pose, float] = self._stereo_slot.get_value()

            if val is None:  # Frame was dropped due to age
                return

            if isinstance(val, StopSignal):
                print("Tracker got stop in Stereo Slot")
                self._stereo_slot_stopped = True
                if not self._emitted_downstream_stop:
                    self._keyframe_signal.emit(StopSignal())
                    self._emitted_downstream_stop = True
            else:
                new_stereo, new_gt_pose, associated_imu_ts = val

                # Check if graph is initialized before processing stereo
                if not self._backend.is_graph_initialized():
                    print(f"Skipping stereo frame at time {float(new_stereo.timestamp):.3f}s - "
                          "waiting for graph initialization")
                    return

                if self._ready:
                    self._keyframe_synthesizer.process_stereo(new_stereo, new_gt_pose, associated_imu_ts)
                else:
                    print(f"Skipping frame creation at time {float(new_stereo.timestamp)}"
                          "since tracker isn't initialized.")

        kf_triggered = False
        while self._keyframe_synthesizer.has_frame():
            keyframe = self._keyframe_synthesizer.peek_frame()

            ts = keyframe.get_time()

            # Safe to pop and process frame
            keyframe = self._keyframe_synthesizer.pop_frame()

            keyframe._id = self._frame_count
            self._frame_count += 1

            self._keyframe_signal.emit(keyframe.clone())
            self._keyframes[keyframe.get_id()] = keyframe
            self._kf_stereo_to_imu_lookup[keyframe.get_id()] = keyframe._associated_imu_timestamp
            self._last_tracked_frame_time = ts
            num_tracked += 1
            kf_triggered = True

        if self._registration_result_signal.has_value():
            n = len(self._registration_result_signal)
            for _ in range(n):
                registration_result: RegistrationResult | StopSignal = (
                    self._registration_result_signal.get_value()
                )  # type: ignore

                if registration_result is None:  # Result was dropped due to age
                    continue

                if isinstance(registration_result, StopSignal):
                    print("Tracker got stop in RegistrationResult Slot")
                    self._registration_slot_stopped = True
                    continue

                id_a = registration_result.id_a
                id_b = registration_result.id_b
                        
                # re-associate times to what the posegraph internally uses
                registration_result.timestamp_a = self._kf_stereo_to_imu_lookup.get(id_a, None) # type: ignore
                registration_result.timestamp_b = self._kf_stereo_to_imu_lookup.get(id_b, None) # type: ignore

                if registration_result.timestamp_a is None or registration_result.timestamp_b is None:
                    continue
                
                assert id_a is not None and id_b is not None

                stereo_mat = registration_result.T_a_b.get_transformation_matrix().cpu().numpy()
                covariance: torch.Tensor = registration_result.T_a_b.covariance
                reorder_indices = torch.tensor([3, 4, 5, 0, 1, 2])
                if covariance is not None:
                    # Reorder covariance from [tx, ty, tz, rx, ry, rz] to [rx, ry, rz, tx, ty, tz]
                    covariance = covariance[reorder_indices][:, reorder_indices]

                T_pg_a_t, T_pg_b_t, T_pg_between = None, None, None
                T_pg_b_t_cov = None

                if id_a is not None and id_b is not None:
                    T_pg_a_t = self._backend.get_pose_at_time(
                        registration_result.timestamp_a)
                    T_pg_b_t, T_pg_b_t_cov = self._backend.get_pose_at_time(
                        registration_result.timestamp_b, True, True)
                    
                if T_pg_a_t is not None and T_pg_b_t is not None:
                    T_pg_between = np.linalg.inv(T_pg_a_t) @ T_pg_b_t
                else:
                    print(
                        "[Registration] Posegraph pose unavailable for comparison."
                    )

                if T_pg_a_t is not None and T_pg_b_t is not None and \
                    T_pg_between is not None and T_pg_b_t_cov is not None:
                    # Compare registration and TurtleMap transforms
                    T_reg = registration_result.T_a_b.get_transformation_matrix().cpu().numpy()
                    comparison = self._compare_transforms(T_reg, T_pg_between)

                    dist = self._backend.compute_mahalanobis_distance(
                        Pose(torch.from_numpy(T_pg_a_t)),
                        Pose(torch.from_numpy(T_pg_b_t)),
                        registration_result.T_a_b,
                        torch.from_numpy(T_pg_b_t_cov),
                        min_diag_cov=0.01,
                    )
                    
                    is_loop_closure = registration_result.timestamp_b - registration_result.timestamp_a > 3

                    rejection_settings = self._tracker_settings.registration_rejection.loop_closure \
                        if is_loop_closure else self._tracker_settings.registration_rejection.frame_to_frame

                    if comparison['translation_error_m'] > rejection_settings.max_translation_error:
                        print(
                            f"Large translation error! Error is {comparison['translation_error_m']:.5f}. Mahal. Dist: {dist:.4f}")

                    dist_thresh = rejection_settings.stereo_meas_max_std_dev


                    accepted = (dist < dist_thresh and
                            comparison['translation_error_m'] < rejection_settings.max_translation_error and \
                            comparison['rotation_error_deg'] < rejection_settings.max_rotation_err_deg)

                    if accepted:
                        if is_loop_closure:
                            print(f"Accepted loop closure stereo transform with Mahalanobis distance {dist:.4f}")
                        self._backend.process_registration_result(
                            timestamp=0.0 if is_loop_closure else registration_result.timestamp_b,
                            transformation=stereo_mat,
                            prev_kf_timestamp=registration_result.timestamp_a,
                            curr_kf_timestamp=registration_result.timestamp_b
                        )
                    else:
                        if is_loop_closure:
                            print(f"Rejected loop closure stereo transform! Mahalanobis distance of {dist} exceeds threshold {dist_thresh}")
                        print(
                            f"Rejected stereo transform! Mahalanobis distance of {dist} exceeds threshold {dist_thresh}")

                    latest_kf_id = max(self._keyframes.keys())
                    latest_kf = self._keyframes[latest_kf_id]
                    ts = latest_kf.get_time()
                    

                    

                else:
                    print("Unable to process stereo... something wasn't available.")

        elif kf_triggered:
            # get the latest kf id from self._keyframes
            latest_kf_id = max(self._keyframes.keys())
            latest_kf = self._keyframes[latest_kf_id]
            ts = latest_kf.get_time()
        
        # Check if all slots have received stop signals
        if self._backend_slot_stopped and self._stereo_slot_stopped and self._registration_slot_stopped:
            print("Tracker received stop signals from all slots")
            self._processed_stop_signal.value = 1
            return

        toc = time.time()

        if num_tracked > 0 and self._tracker_settings.debug.log_times:
            with open(
                f"{self._tracker_settings.log_directory}/track_times.csv", "a+"
            ) as time_f:
                time_f.write(f"{toc - tic},{num_tracked}\n")
                
    def _compare_transforms(self,
                            T_reg: np.ndarray,
                            T_tmap: np.ndarray) -> dict:
        """Compare two 4x4 transformation matrices and return differences."""
        # Extract translations
        t_reg = T_reg[:3, 3]
        t_tmap = T_tmap[:3, 3]
        trans_diff = np.linalg.norm(t_reg - t_tmap)

        # Extract rotations and compute angular difference
        R_reg = T_reg[:3, :3]
        R_tmap = T_tmap[:3, :3]

        # Compute relative rotation
        R_diff = R_tmap.T @ R_reg

        # Convert to angle-axis to get rotation error
        rot_diff = Rotation.from_matrix(R_diff)
        angle_diff_deg = rot_diff.magnitude() * 180.0 / np.pi

        return {
            'translation_error_m': trans_diff,
            'rotation_error_deg': angle_diff_deg,
            't_reg': t_reg,
            't_tmap': t_tmap,
            'R_reg': R_reg,
            'R_tmap': R_tmap
        }

    def finish(self):
        pass

    # Run spins and processes incoming data while putting resulting frames into the queue
    def run(self, shared_state: SharedState) -> None:
        self._last_mapped_frame_time = shared_state.last_mapped_frame_time
        self._shared_state = shared_state

        self._backend.start()

        while not self._processed_stop_signal.value:
            self.update()

        print("Tracking Done. Waiting to terminate.")

        self.finish()

        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting tracking process.")
