"""
File: src/loner.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import time
from typing import Union
import threading

import torch
import torch.multiprocessing as mp
import os
import pickle
import yaml
import datetime
from torch.profiler import ProfilerActivity, profile


from common.pose import Pose
from common.signals import Signal, StopSignal
from common.sensors import StereoImage, BackendData
from pyturtlmap import ImuMeasurement, DvlMeasurement, BarometerMeasurement
from common.settings import Settings
from mapping.registration import Registration
from mapping.frame_tracker import FrameTracker
from mapping.stereo.stereo_matcher import StereoMatcher
from tracking.tracker import Tracker
from pathlib import Path
from common.shared_state import SharedState
from common.timing import log_process_timing
from typing import Dict, Any


class SurfSLAM:
    """ Top-level SLAM module.

    To run sychronously, call Start with sychronous=true, then pass in measurements in the
    same or a different thread. When you're done, call Stop()
    """

    def _get_signal_config(self, signal_name: str) -> Dict[str, Any]:
        """Get staleness configuration for a signal.

        Returns a dict with max_age_seconds, warn_on_drop, and summary_interval_seconds
        based on the signal_staleness config section.
        """
        staleness_cfg = self._settings.system.get("signal_staleness", {})

        if not staleness_cfg.get("enabled", True):
            return {"max_age_seconds": None, "warn_on_drop": False, "summary_interval_seconds": 10.0}

        signals_cfg = staleness_cfg.get("signals", {})
        default_cfg = staleness_cfg.get("default", {})
        signal_cfg = signals_cfg.get(signal_name, {})

        return {
            "max_age_seconds": signal_cfg.get("max_age_seconds", default_cfg.get("max_age_seconds")),
            "warn_on_drop": signal_cfg.get("warn_on_drop", default_cfg.get("warn_on_drop", False)),
            "summary_interval_seconds": staleness_cfg.get("summary_interval_seconds", 10.0),
        }

    def __init__(self, settings: Union[Settings, str]) -> None:
        if isinstance(settings, str):
            self._settings = Settings.load_from_file(settings)

        elif type(settings).__name__ == "Settings":  # Avoiding strange attrdict behavior
            self._settings = settings
        else:
            raise RuntimeError(
                f"Can't load settings of type {type(settings).__name__}")

        self._single_threaded = self._settings.system.single_threaded
        self._track_in_main_process = self._settings.system.track_in_main_process

        if not self._single_threaded:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as e:
                # Don't blame me, pytorch should've made this something other than a generic RuntimeError
                if str(e) == "context has already been set":
                    pass
                else:
                    raise e

        # The top-level module inserts RGB frames/Lidar, and the tracker reads them
        stereo_cfg = self._get_signal_config("stereo_input")
        self._stereo_signal = Signal(
            synchronous=True,
            single_process=self._single_threaded or self._track_in_main_process,
            name="stereo_input",
            signal_type="input",
            **stereo_cfg
        )

        backend_cfg = self._get_signal_config("backend_input")
        self._backend_signal = Signal(
            synchronous=True,
            single_process=self._single_threaded or self._track_in_main_process,
            name="backend_input",
            signal_type="input",
            **backend_cfg
        )

        # The tracker makes frames and emits them here
        frame_cfg = self._get_signal_config("frame")
        self._frame_signal = Signal(
            single_process=self._single_threaded,
            name="frame",
            signal_type="internal",
            **frame_cfg
        )

        # the StereoMatcher computes depths, adds them to the frames, and emits the frame here
        completed_frame_cfg = self._get_signal_config("completed_frame")
        self._completed_frame_signal = Signal(
            single_process=self._single_threaded,
            name="completed_frame",
            signal_type="internal",
            **completed_frame_cfg
        )

        # When the Registration process registers frames, it puts there results here
        registration_cfg = self._get_signal_config("registration_result")
        self._registration_result_signal = Signal(
            single_process=self._single_threaded,
            name="registration_result",
            signal_type="internal",
            **registration_cfg
        )
        
        # When the stereo matcher computes the depth, add them to the frames and emit the depth
        # this is a duplicate of completed_frame_signal, just ensuring that any mutatiuon of the frame
        # is not affecting other processes
        mapping_cfg = self._get_signal_config("mapping_input")
        self._mapping_input_signal = Signal(
            single_process=self._single_threaded,
            name="mapping_input",
            signal_type="internal",
            **mapping_cfg
        )
        
        # Final poses signal - sent at the end of the run with all optimized poses
        final_poses_cfg = self._get_signal_config("final_poses")
        self._final_poses_signal = Signal(
            single_process=self._single_threaded,
            name="final_poses",
            signal_type="internal",
            **final_poses_cfg
        )

        # Backend data signal - shares latest IMU/DVL/Barometer data with FrameTracker
        backend_to_mapper_cfg = self._get_signal_config("backend_to_mapper")
        self._backend_to_mapper_signal = Signal(
            synchronous=True,
            single_process=self._single_threaded,
            name="backend_to_mapper",
            signal_type="internal",
            **backend_to_mapper_cfg
        )

        # Buffer for latest backend measurements (emitted when stereo is processed)
        self._latest_imu = None
        self._latest_dvl = None
        self._latest_barometer = None

        # Placeholder for the Mapping and Tracking processes 
        self._tracker = None
        self._stereo_matcher = None
        self._registration = None
        self._mapper_process = None
        
        
        self._tracking_process = None
        self._stereo_process = None
        self._registration_process = None
        self._mapper_process = None

        # Process monitoring
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()

        # To initialize, call initialize
        self._initialized = False

        self._shared_state = SharedState()
        
    def set_start_time(self, start_time):
        self.start_time = start_time

    def _monitor_processes(self):
        """Monitor child processes and terminate all if any dies unexpectedly."""
        check_interval = 1.0  # Check every second

        while not self._stop_monitoring.is_set():
            time.sleep(check_interval)

            # Check each process that should be running (hasn't been told to stop yet)
            dead_processes = []

            # Only flag as dead if the process died AND hasn't processed the stop signal
            if (self._tracking_process is not None and
                not self._tracking_process.is_alive() and
                not self._tracker._processed_stop_signal.value):
                dead_processes.append("Tracking")

            if (self._stereo_process is not None and
                not self._stereo_process.is_alive() and
                not self._stereo_matcher._processed_stop_signal.value):
                dead_processes.append("StereoMatcher")

            if (self._registration_process is not None and
                not self._registration_process.is_alive() and
                not self._registration._processed_stop_signal.value):
                dead_processes.append("Registration")

            if dead_processes:
                print(f"\n!!! FATAL ERROR: Process(es) died unexpectedly: {', '.join(dead_processes)}")
                print("!!! Terminating all processes...")
                time.sleep(2) # let things settle down a bit
                os._exit(1)
                

    def initialize(self, dataset_path: str,
                         experiment_name: str = None,
                         config_idx: int = None,
                         trial_idx: int = None):
        
        self._initialized = True
        self._dataset_path = Path(dataset_path).resolve().as_posix()
        
        now = datetime.datetime.now()
        now_str = now.strftime("%y%m%d_%H%M%S")
        
        if "experiment_name" in self._settings:
            expname = self._settings.experiment_name
        else:
            expname = "experiment"
            
        self._experiment_name = f"{expname}_{now_str}"
        if experiment_name is None:
            self._log_directory = os.path.expanduser(f"{self._settings.system.log_dir_prefix}/{self._experiment_name}/")
        else:
            self._log_directory = os.path.expanduser(f"{self._settings.system.log_dir_prefix}/{experiment_name}/")
            if config_idx is not None:
                self._log_directory += f"config_{config_idx}/"
            if trial_idx is not None:
                self._log_directory += f"trial_{trial_idx}/"

        os.makedirs(self._log_directory, exist_ok=True)
        
        self._settings["experiment_name"] = self._experiment_name
        self._settings["dataset_path"] = self._dataset_path
        self._settings["log_directory"] = self._log_directory

        self._settings["mapper"]["experiment_name"] = self._experiment_name
        self._settings["mapper"]["log_directory"] = self._log_directory

        self._settings["tracker"]["experiment_name"] = self._experiment_name
        self._settings["tracker"]["log_directory"] = self._log_directory

        # Pass debug settings through
        if self._settings.debug.flags is not None:
            for key in self._settings.debug.flags:
                self._settings["debug"][key] = self._settings.debug.flags[key] and self._settings.debug.global_enabled

    def start(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Can't Start: System Uninitialized. You must call initialize first.")

        with open(f"{self._log_directory}/full_config.yaml", 'w+') as f:
            yaml.dump(self._settings, f)

        with open(f"{self._log_directory}/full_config.pkl", 'wb+') as f:
            pickle.dump(self._settings, f)

        if self._settings.debug.profile:
            prof_dir = f"{self._settings.log_directory}/profile"
            os.makedirs(prof_dir, exist_ok=True)
            
            self._profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{prof_dir}/tensorboard/"))
            
            self._profiler.start()
        
        self._settings["tracker"]["log_directory"] = self._log_directory
        self._settings["tracker"]["debug"] = self._settings.debug
        self._tracker = Tracker(self._settings,
                                self._stereo_signal,
                                self._backend_signal,
                                self._frame_signal,
                                self._registration_result_signal)
        
        self._settings["mapper"]["stereo_estimator"]["log_directory"] = self._log_directory
        self._settings["mapper"]["stereo_estimator"]["debug"] = self._settings.debug
        self._stereo_matcher = StereoMatcher(self._settings.mapper.stereo_estimator,
                                             self._settings.calibration,
                                             self._frame_signal,
                                             self._completed_frame_signal,
                                             self._mapping_input_signal)
        
        self._settings["mapper"]["registration"]["log_directory"] = self._log_directory
        self._settings["mapper"]["registration"]["debug"] = self._settings.debug
        self._registration = Registration(self._settings.mapper.registration,
                                          self._settings.calibration,
                                          self._completed_frame_signal,
                                          self._registration_result_signal)
        
        # Use FrameTracker instead of Mapper - defers pose association to end of run
        self._mapper = FrameTracker(self._settings,
                                    self._final_poses_signal,
                                    self._mapping_input_signal,
                                    self._backend_to_mapper_signal)

        print("Starting SurfSLAM")

        if self._single_threaded or self._track_in_main_process:
            self._tracker.start_backend()
            
        if self._single_threaded:
            self._stereo_matcher.start()

        if not self._single_threaded:

            # Start the children
            if not self._track_in_main_process:
                self._tracking_process = mp.Process(target=self._tracker.run, args=(self._shared_state,))
                self._tracking_process.daemon = True
                self._tracking_process.start()

            self._registration_process = mp.Process(target=self._registration.run, args=(self._shared_state,))
            self._registration_process.daemon = True
            self._registration_process.start()
            
            self._stereo_process = mp.Process(target=self._stereo_matcher.run, args=(self._shared_state,))
            self._stereo_process.daemon = True
            self._stereo_process.start()
            
            self._mapper_process = mp.Process(target=self._mapper.run, args=(self._shared_state,))
            self._mapper_process.daemon = True
            self._mapper_process.start()

            # Start the process monitor thread
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
            self._monitor_thread.start()


        if self._single_threaded and self._settings.debug.pytorch_detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

    def _collect_final_poses(self) -> Dict[int, Any]:
        """Collect all final optimized poses from the backend.
        
        Returns:
            Dict mapping keyframe IDs (1-indexed backend convention) to 4x4 pose matrices
        """
        import numpy as np
        
        final_poses = {}
        
        try:
            # Get the backend instance from the tracker
            backend = self._tracker._backend
            
            # Get the number of keyframes processed
            num_keyframes = backend._pg_backend.get_processed_keyframe_count()
            print(f"Backend has {num_keyframes} processed keyframes")
            
            # Collect poses for all keyframes (1-indexed in the backend)
            for kf_id in range(1, num_keyframes + 1):
                try:
                    pose = backend.get_pose_at_keyframe(kf_id)
                    if pose is not None:
                        # Ensure it's a numpy array
                        if hasattr(pose, 'cpu'):
                            pose = pose.cpu().numpy()
                        elif not isinstance(pose, np.ndarray):
                            pose = np.array(pose)
                        final_poses[kf_id] = pose
                except Exception as e:
                    print(f"Warning: Could not get pose for keyframe {kf_id}: {e}")
                    
        except Exception as e:
            print(f"Error collecting final poses: {e}")
            import traceback
            traceback.print_exc()
            
        return final_poses

    # Stop the processes running the mapping and tracking
    def stop(self):

        if not self._single_threaded:
            print("Stopping LONER SLAM Sub-Processes")

            self._stereo_signal.emit(StopSignal())
            self._backend_signal.emit(StopSignal())

            while not self._tracker._processed_stop_signal.value:
                if self._track_in_main_process:
                    self._tracker.update()
                else:
                    time.sleep(0.1)
            
            while not self._registration._processed_stop_signal.value:
                time.sleep(0.1)
                
            while not self._stereo_matcher._processed_stop_signal.value:
                time.sleep(0.1)
            
            # Collect and emit final poses from the backend before mapper processes them
            print("Collecting final poses from the backend...")
            final_poses = self._collect_final_poses()
            print(f"Emitting {len(final_poses)} final poses to FrameTracker")
            self._final_poses_signal.emit(final_poses)
                
            # Add timeout in case StopSignal isn't received
            mapper_wait_start = time.time()
            mapper_timeout = 60.0  # 60 second timeout (map accumulation takes time)
            while not self._mapper._processed_stop_signal.value:
                time.sleep(0.1)
                if time.time() - mapper_wait_start > mapper_timeout:
                    print(f"Warning: FrameTracker did not receive StopSignal after {mapper_timeout}s, forcing stop...")
                    # The mapper is in a separate process, we can't call stop() directly
                    # but we can set the signal value (though this won't save the map properly)
                    # Better to emit StopSignal directly on the signal
                    self._mapping_input_signal.emit(StopSignal())
                    self._final_poses_signal.emit(StopSignal())
                    time.sleep(2.0)  # Give it time to process
                    break
                
        if self._settings.debug.profile:
            self._profiler.stop()

        if not self._single_threaded:
            if not self._track_in_main_process:
                self._tracker._term_signal.value = True
                self._tracking_process.join()

            self._registration._term_signal.value = True
            self._registration_process.join()

            self._stereo_matcher._term_signal.value = True
            self._stereo_process.join()
            
            self._mapper._term_signal.value = True
            self._mapper_process.join()

            # Now that all processes have exited, stop the monitor thread
            self._stop_monitoring.set()
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=2.0)
        else:
            self._stereo_matcher.finish()
            self._tracker.finish()

            # Collect final poses and pass directly to mapper in single-threaded mode
            final_poses = self._collect_final_poses()
            self._mapper.stop(final_poses)

        # Persist results even if one of these wedges: a partial census beats none.
        try:
            self._tracker.save_backend_traj(f"{self._log_directory}/trajectory/surfslam")
        except Exception as exc:
            print(f"WARNING: failed to save backend trajectory: {exc}")
        try:
            self._save_statistics()
        except Exception as exc:
            print(f"WARNING: failed to save pipeline statistics: {exc}")
        print("SurfSLAM successfully terminated. Goodbye!")

    def save_map(self, filepath: str) -> None:
        """Save the reconstructed mesh to disk."""
        if self._mapper is None:
            raise RuntimeError("Mapper is not initialized; call start() first.")

        final_poses = self._collect_final_poses()
        self._mapper.save_map(filepath, final_poses)
        
    # For use in single-threaded system. 
    def _system_update(self):
        assert self._single_threaded, "_system_update should only be called in single-threaded mode"

        self._tracker.update()
        self._stereo_matcher.update()
        self._registration.update()
        self._mapper.update()

        if self._settings.debug.profile:
            self._profiler.step()

    def process_backend_data(self, backend_data: BackendData):
        self._backend_signal.emit(backend_data)

        # Buffer the latest measurement by type for emission with stereo frames
        if isinstance(backend_data, ImuMeasurement):
            self._latest_imu = backend_data
        elif isinstance(backend_data, DvlMeasurement):
            self._latest_dvl = backend_data
        elif isinstance(backend_data, BarometerMeasurement):
            self._latest_barometer = backend_data

        if self._single_threaded:
            self._system_update()

        elif self._track_in_main_process:
            self._tracker.update()
            
    def get_last_kf_timestamp(self):
        return self._tracker._backend.get_last_kf_timestamp()
            
    def process_stereo(self, stereo_img: StereoImage, gt_pose: Pose, associated_imu_ts: float = None) -> None:
        tic = time.perf_counter()
        self._stereo_signal.emit((stereo_img, gt_pose, associated_imu_ts))

        # Emit latest backend measurements to FrameTracker
        self._backend_to_mapper_signal.emit({
            "imu": self._latest_imu,
            "dvl": self._latest_dvl,
            "barometer": self._latest_barometer,
        })

        if self._single_threaded:
            self._system_update()
        elif self._track_in_main_process:
            self._tracker.update()

        toc = time.perf_counter()
        timestamp = getattr(stereo_img, "timestamp", None)
        if timestamp is not None:
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError):
                timestamp = str(timestamp)
        log_process_timing(
            getattr(self._settings, "log_directory", None),
            "process_stereo_total",
            toc - tic,
            metadata={"timestamp": timestamp},
        )
    
    def is_graph_initialized(self) -> bool:
        """Check if backend graph is initialized and ready for stereo processing."""
        if self._tracker is None:
            return False
        return self._tracker.is_graph_initialized()

    def _save_statistics(self):
        """Save frame processing and signal drop statistics to log directory."""
        stats_file = f"{self._log_directory}/pipeline_statistics.txt"

        with open(stats_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PIPELINE STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            # Frame processing stats
            stereo_input = self._shared_state.stereo_input_count.value
            stereo_processed = self._shared_state.stereo_processed_count.value
            registration_processed = self._shared_state.registration_processed_count.value

            f.write("Frame Processing Counts:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Stereo frames input:           {stereo_input}\n")
            f.write(f"  Stereo frames processed:       {stereo_processed}\n")
            f.write(f"  Registration frames processed: {registration_processed}\n")

            # Calculate drop rates
            if stereo_input > 0:
                stereo_drop_rate = 100.0 * (1 - stereo_processed / stereo_input)
                f.write(f"\n  Stereo processing drop rate:   {stereo_drop_rate:.2f}%\n")

            if stereo_processed > 0:
                reg_drop_rate = 100.0 * (1 - registration_processed / stereo_processed)
                f.write(f"  Registration drop rate:        {reg_drop_rate:.2f}%\n")

            if stereo_input > 0:
                overall_drop_rate = 100.0 * (1 - registration_processed / stereo_input)
                f.write(f"  Overall drop rate:             {overall_drop_rate:.2f}%\n")

            # Signal drop stats
            f.write("\n" + "=" * 80 + "\n")
            f.write("SIGNAL DROP STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            for signal_name, signal in [
                ("stereo_input", self._stereo_signal),
                ("backend_input", self._backend_signal),
                ("frame", self._frame_signal),
                ("completed_frame", self._completed_frame_signal),
                ("registration_result", self._registration_result_signal),
            ]:
                stats = signal.get_drop_stats()

                f.write(f"{signal_name}:\n")
                f.write(f"  Total dropped:     {stats['total_dropped']}\n")
                f.write(f"  Number of slots:   {stats['num_slots']}\n")
                f.write(f"  Max age (seconds): {stats['max_age_seconds']}\n")
                f.write("\n")

        print(f"\nPipeline statistics saved to: {stats_file}")

