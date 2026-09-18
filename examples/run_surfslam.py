#!/usr/bin/env python3

"""
File: examples/run_loner.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import argparse
import copy
import datetime
import os
import sys
import time
import random
import traceback
import warnings

from utils.h5_log_reader import H5LogReader
import pandas as pd
import torch
from pathlib import Path
import torch.multiprocessing as mp
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/surfslam")


from surfslam.surfslam import SurfSLAM
from surfslam.common.dataset_paths import load_config, resolve_config_path
from surfslam.common.pose import Pose
from surfslam.common.sensors import StereoImage
from surfslam.common.settings import Settings
from surfslam.common.pose_utils import build_poses_from_df
from examples.tbnms.tbnms_calibration import TBNMSCalibration
from examples.utils.pose_utils import *

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")



LIDAR_MIN_RANGE = 0.1 #http://www.oxts.com/wp-content/uploads/2021/01/Ouster-datasheet-revc-v2p0-os0.pdf


WARN_MOCOMP_ONCE = True
WARN_LIDAR_TIMES_ONCE = True

_LAST_TRIAL_LOGDIR = None


def run_trial(config, settings, settings_description = None, config_idx = None, trial_idx = None, dryrun: bool = False):
    
    if trial_idx is None:    
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(trial_idx)
    else:
        np.random.seed(trial_idx)
        torch.manual_seed(trial_idx)
        random.seed(trial_idx)


    im_scale_factor = settings.system.image_scale_factor

    h5_path = Path(os.path.expanduser(config["dataset"]))
    
    init = False

    init_clock = time.perf_counter()
    
    calibration: TBNMSCalibration = load_calibration(config["dataset_family"], config["calibration"])

    stereo_key = settings.system.h5_keys.stereo
    barometer_key = settings.system.h5_keys.barometer
    dvl_key = settings.system.h5_keys.dvl
    imu_key = settings.system.h5_keys.imu

    # okokok in this case lidar = "stereo points" from a system perspective. live with it :)

    if calibration is not None:
        settings["calibration"] = calibration.to_dict(im_scale_factor)
        
    settings["experiment_name"] = config["experiment_name"]
    settings["run_config"] = config
    surfslam = SurfSLAM(settings)
    gt_traj = config.get("groundtruth_traj", None)
    # Get ground truth trajectory. This is only used to construct the world cube.
    if gt_traj is not None:
        ground_truth_file = os.path.expanduser(config["groundtruth_traj"])
        ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ", comment="#")
        camera_poses, timestamps = build_poses_from_df(ground_truth_df, True)
        tf_buffer, timestamps = build_buffer_from_poses(camera_poses, timestamps)
    else:
        tf_buffer = None
        camera_poses = None

    if config_idx is None and trial_idx is None:
        ablation_name = None
    else:
        ablation_name = config["experiment_name"]

    surfslam.initialize(h5_path.as_posix(), ablation_name, config_idx, trial_idx)

    logdir = surfslam._log_directory

    if settings_description is not None and config_idx is not None:
        if trial_idx == 0:
            with open(f"{logdir}/../configuration.txt", 'w+') as desc_file:
                desc_file.write(settings_description)
        elif trial_idx is None:
            with open(f"{logdir}/configuration.txt", 'w+') as desc_file:
                desc_file.write(settings_description)
    
    if dryrun:
        return

    # Recorded so the driver can mark this trial's directory if it blows up.
    global _LAST_TRIAL_LOGDIR
    _LAST_TRIAL_LOGDIR = logdir

    surfslam.start()

    start_camera_pose = None

    start_clock = None # wall time
    start_time = None # bag time

    prev_scan_time = float('-inf')
    stereo_delta_t = settings.tracker.frame_synthesis.stereo_delta_t_sec
    imu_delta_t = settings.tracker.frame_synthesis.imu_delta_t_sec
    imu_kf_lookahead_suppression = settings.tracker.frame_synthesis.imu_kf_lookahead_suppression_sec

    # Keyframe timing state
    last_imu_kf_time = float('-inf')
    next_stereo_expected_time = float('-inf')

    # Pending stereo frame waiting for next IMU
    pending_stereo = None  # Will hold: (timestamp, leftRect, rightRect, gt_camera_pose)

    graph_initialized = False  # Track when backend graph is ready
    stereo_count = 0  # Track number of stereo frames processed

    if config['dataset_family'] == 'tbnms':
        log_reader = H5LogReader(h5_path, imu_key, dvl_key, barometer_key, stereo_key)
    else:
        raise ValueError(f"Unknown dataset family: {config['dataset_family']}")

    # Get time filtering configuration
    config_start_time = settings.system.get('start_time', None)
    config_end_time = settings.system.get('end_time', None)
    log_first_stamp = log_reader.first_stamp()
    log_last_stamp = log_reader.events()[-1].stamp
    
    last_time = None
    last_wall_time = None
    
    for event in log_reader.events():
        timestamp = copy.deepcopy(event.stamp)
        
        # Apply time filtering relative to log start
        time_offset = timestamp - log_first_stamp
        if config_start_time is not None and time_offset < config_start_time:
            print(f"Skipping event at {timestamp} (offset {time_offset}) before start_time {config_start_time}")
            continue
        if config_end_time is not None and time_offset > config_end_time:
            break
        
        if (not init) and timestamp and (tf_buffer is None or timestamp >= timestamps[0]):
            init = True
            start_time = timestamp
            next_stereo_expected_time = start_time + stereo_delta_t
            start_clock = time.perf_counter()
            
            with open(f"{logdir}/start_time.txt", 'w+') as start_time_file:
                start_time_file.write(f"{start_time}\n")
                
            surfslam.set_start_time(start_time)

        if not init:
            continue
        
        
        # timestamp -= start_time
        now = time.perf_counter()
        if last_time is not None:
            dt_ideal = timestamp - start_time
            t_elapsed = now - start_clock
            time.sleep(max(0, dt_ideal - t_elapsed))
        
        if config["duration"] is not None and timestamp - start_time > config["duration"]:
            break
        
        if event.type == "IMU":
            m = log_reader.imu()[event.index]
            m.timestamp = timestamp
            
            # Check if stereo is waiting for this IMU
            if pending_stereo is not None:
                # This IMU becomes the keyframe for the pending stereo
                m.make_keyframe = True
                last_imu_kf_time = timestamp
                m.skip_optimize = False

                # Send the IMU first
                surfslam.process_backend_data(m)

                # Now send the stereo with this IMU association
                stereo_ts, leftRect, rightRect, gt_pose = pending_stereo
                stereo_img = StereoImage(
                    torch.from_numpy(leftRect).permute(2,0,1),
                    torch.from_numpy(rightRect).permute(2,0,1),
                    stereo_ts
                )
                surfslam.process_stereo(stereo_img, Pose(gt_pose), timestamp)  # timestamp is IMU timestamp
                surfslam._shared_state.stereo_input_count.value += 1

                stereo_count += 1
                if stereo_count % 10 == 0:
                    elapsed = time.perf_counter() - start_clock
                    log_progress = (timestamp - log_first_stamp) / (log_last_stamp - log_first_stamp) * 100
                    print(f"Processed {stereo_count} stereo frames ({log_progress:.1f}% through log, {elapsed:.1f}s elapsed). {(timestamp - log_first_stamp):.1f} log time processed.")

                pending_stereo = None

            # Only check for timer-based keyframes if no pending stereo
            elif timestamp - last_imu_kf_time >= imu_delta_t:
                # Check lookahead: is stereo frame expected soon?
                time_until_stereo = next_stereo_expected_time - timestamp

                if time_until_stereo > imu_kf_lookahead_suppression:
                    # Safe to mark as keyframe - stereo is far enough away
                    m.make_keyframe = True
                    last_imu_kf_time = timestamp
                    m.skip_optimize=True

                # Send IMU
                surfslam.process_backend_data(m)
            
            else:
                # Not a keyframe, just send the IMU
                surfslam.process_backend_data(m)
        elif event.type == "DVL":
            m = log_reader.dvl()[event.index]
            m.timestamp = timestamp
            surfslam.process_backend_data(m)
        elif event.type == "PRESSURE":
            m = log_reader.pressure()[event.index]
            m.timestamp = timestamp
            surfslam.process_backend_data(m)
        elif event.type == "STEREO":
            # Check if graph is initialized (only needs to be checked once)
            if not graph_initialized:
                graph_initialized = surfslam.is_graph_initialized()
            
            # Only apply decimation AFTER graph is initialized
            # Before initialization, we need to pass all frames so IMU can initialize
            if graph_initialized and settings.tracker.frame_synthesis.decimate_on_load \
                and (timestamp - prev_scan_time < stereo_delta_t):
                continue
                
            prev_scan_time = timestamp
            next_stereo_expected_time = timestamp + stereo_delta_t  # Update lookahead prediction
             
            if tf_buffer is not None:
                try:
                    T_world_cam_gt = tf_buffer.lookup(event.stamp)
                    if T_world_cam_gt is None:
                        continue
                except Exception as e:
                    print(f"Failed to lookup transform. May have reached the end of the log. Exiting early. Error: {e}")
                    break
                if start_camera_pose is None:
                    start_camera_pose = T_world_cam_gt

                gt_camera_pose = start_camera_pose.inverse() @ T_world_cam_gt
            else:
                gt_camera_pose = torch.eye(4)
            
            m = log_reader.stereo()[event.index]
            _, left, right = m
            
            leftRect, rightRect = calibration.rectify_images(left, right, im_scale_factor)
            
            # HOLD the stereo - don't send it yet
            # Wait for next IMU to associate and mark as keyframe
            pending_stereo = (timestamp, leftRect, rightRect, gt_camera_pose)

        else:
            raise Exception("Should be unreachable")

        last_time = timestamp
        last_wall_time = now

    # Handle any leftover pending stereo frame
    if pending_stereo is not None:
        print(f"Warning: Stereo frame at {pending_stereo[0]} was pending but no more IMU data. Discarding.")
        pending_stereo = None

    surfslam.stop()
    end_clock = time.perf_counter()

    with open(f"{surfslam._log_directory}/runtime.txt", 'w+') as runtime_f:
        runtime_f.write(f"Execution Time (With Overhead): {end_clock - init_clock}\n")
        if start_clock is not None:
            runtime_f.write(f"Execution Time (Without Overhead): {end_clock - start_clock}\n")
        else:
            runtime_f.write("Execution Time (Without Overhead): n/a (never initialized)\n")

# Implements a single worker in a thread-pool model.
def _gpu_worker(config, gpu_id: int, job_queue: mp.Queue, dryrun: bool) -> None:

    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            return

        settings, description, config_idx, trial_idx = data
        run_trial(config, settings, description, config_idx, trial_idx, dryrun)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Loner SLAM on RosBag")
    parser.add_argument("configuration_path")
    parser.add_argument("experiment_name", nargs="?", default=None)
    parser.add_argument("--duration", help="How long to run for (in input data time, sec)", type=float, default=None)
    parser.add_argument("--gpu_ids", nargs="*", required=False, default = None, help="Which GPUs to use. Defaults to parallel if set")
    parser.add_argument("--num_repeats", type=int, required=False, default=1, help="How many times to run the experiment")
    parser.add_argument("--run_all_combos", action="store_true",default=False, help="If set, all combinations of overrides will be run. Otherwise, one changed at a time.")
    parser.add_argument("--overrides", type=str, default=None, help="File specifying parameters to vary for ablation study or testing")
    parser.add_argument("--dryrun", action="store_true",default=False, help="If set, generates output dirs and settings files but doesn't run anything.")
    parser.add_argument("--traj_queue_size", type=int, default=None,
                        help="Override tracker.traj_pose_queue_max_size (dense poses the backend retains for lookups)")
    parser.add_argument("--log_dir_root", type=str, default=None,
                        help="Set system.log_dir_prefix outright (overriding the config's own), so every "
                             "config in a sweep arm lands in the same tree")
    parser.add_argument("--maxrange", type=float, default=None, help="if set, overrides ray_range[1]")

    args = parser.parse_args()


    config = load_config(args.configuration_path)

    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name

    def override_maxrange(data, maxrange):
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "ray_range" and isinstance(v, list) and len(v) > 1:
                    data[k][1] = maxrange
                else:
                    override_maxrange(v, maxrange)
        elif isinstance(data, list):
            for item in data:
                override_maxrange(item, maxrange)

    if args.maxrange is not None:
        override_maxrange(config, args.maxrange)
        
    config["duration"] = args.duration

    baseline_settings_path = resolve_config_path(config['baseline'])

    if args.overrides is not None:
        settings_options, settings_descriptions = \
            Settings.generate_options(baseline_settings_path,
                                      args.overrides,
                                      args.run_all_combos,
                                      [config.get("changes", {})])
        
    else:
        settings_descriptions = [None]
        settings_options = [Settings.load_from_file(baseline_settings_path)]            

        if "changes" in config and config["changes"] is not None:
            settings_options[0].augment(config["changes"])


    for _settings in settings_options:
        if args.traj_queue_size is not None:
            _settings.tracker.traj_pose_queue_max_size = args.traj_queue_size
        if args.log_dir_root is not None:
            _settings.system.log_dir_prefix = args.log_dir_root

    failed_trials = []

    if len(settings_options) > 1 or args.num_repeats > 1:
        now = datetime.datetime.now()
        now_str = now.strftime("%y%m%d_%H%M%S")
        config["experiment_name"] += f"_{now_str}"

    if args.gpu_ids is not None and len(args.gpu_ids) > 1:
        mp.set_start_method('spawn')
        
        if len(settings_descriptions) > 1:
            config_idxs = range(len(settings_descriptions))
        else:
            config_idxs = [None]

        job_queue_data = zip(settings_options, settings_descriptions, config_idxs)

        job_queue = mp.Queue()
        for element in job_queue_data:
            if args.num_repeats == 1:
                job_queue.put(element + (None,))
            else:
                for trial_idx in range(args.num_repeats):
                    job_queue.put(element + (trial_idx,))
        
        for _ in args.gpu_ids:
            job_queue.put(None)

        # Create the workers
        gpu_worker_processes = []
        for gpu_id in args.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(config,gpu_id,job_queue,args.dryrun)))
            gpu_worker_processes[-1].start()

        # Sync
        for process in gpu_worker_processes:
            process.join()
        
    else:
        if args.gpu_ids is not None:
            gpu_id = str(args.gpu_ids[0])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        for config_idx, (settings, description) in enumerate(zip(settings_options, settings_descriptions)):
            if len(settings_options) == 1:
                config_idx = None
            for trial_idx in range(args.num_repeats):
                if args.num_repeats == 1:
                    trial_idx = None
                try:
                    run_trial(config, settings, description, config_idx, trial_idx, args.dryrun)
                except Exception:
                    failed_trials.append((config_idx, trial_idx))
                    print(f"\n!!! Trial failed (config={config_idx}, trial={trial_idx}); "
                          "continuing with the next one.", file=sys.stderr)
                    traceback.print_exc()
                    if _LAST_TRIAL_LOGDIR is not None:
                        try:
                            with open(f"{_LAST_TRIAL_LOGDIR}/FAILED.txt", 'w+') as f:
                                f.write(traceback.format_exc())
                        except OSError as exc:
                            print(f"(could not write FAILED.txt: {exc})", file=sys.stderr)

    if failed_trials:
        print(f"\n{len(failed_trials)} trial(s) failed: "
              + ", ".join(f"config={c} trial={t}" for c, t in failed_trials),
              file=sys.stderr)
