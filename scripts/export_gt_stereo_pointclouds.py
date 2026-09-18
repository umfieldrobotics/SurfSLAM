#!/usr/bin/env python3
"""
Generate per-stereo-frame point clouds using ground-truth poses.

This mirrors the configuration and data loading flow in examples/run_surfslam.py,
but instead of running the full SLAM stack it:
  - loads the GT trajectory
  - rectifies each stereo pair and runs DEFOM stereo inference to get depth
  - backprojects depth to a point cloud in the camera frame
  - transforms the cloud with the GT pose at that timestamp
  - saves one PLY per stereo frame to the requested directory

Example:
python scripts/export_gt_stereo_pointclouds.py cfg/tbnms/mono_full.yaml \\
    --output_dir outputs/gt_stereo_plys --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
from kornia.geometry.depth import depth_to_3d_v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "surfslam"))
sys.path.append(str(PROJECT_ROOT / "examples"))

from surfslam.common.pose_utils import build_poses_from_df  # noqa: E402
from surfslam.common.sensors import StereoImage  # noqa: E402
from surfslam.common.settings import Settings  # noqa: E402
from surfslam.mapping.stereo.stereo_estimation_defom import DefomStereoEstimator  # noqa: E402
from examples.utils.h5_log_reader import H5LogReader  # noqa: E402
from examples.utils.pose_utils import build_buffer_from_poses, load_calibration  # noqa: E402
from surfslam.common.dataset_paths import load_config, resolve_config_path  # noqa: E402


def _backproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor):
    """
    Convert a depth map into 3D points in the camera frame.

    Returns:
        points_cam: (N, 3) tensor of 3D points.
        valid_mask: (H*W,) boolean mask corresponding to valid pixels.
    """
    if depth.dim() == 3 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    elif depth.dim() == 3:
        depth = depth.mean(0)

    if depth.dim() != 2:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.as_tensor(intrinsics, device=depth.device, dtype=depth.dtype)
    else:
        intrinsics = intrinsics.to(device=depth.device, dtype=depth.dtype)

    pcd = depth_to_3d_v2(depth.unsqueeze(0), intrinsics.unsqueeze(0)).squeeze(0)
    pcd_flat = pcd.view(-1, 3)

    valid_mask = torch.isfinite(pcd_flat).all(dim=1) & (pcd_flat[:, 2] > 0)
    points_cam = pcd_flat[valid_mask]
    return points_cam, valid_mask


def _transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    homog = torch.cat([points, ones], dim=1)
    return (transform @ homog.T).T[:, :3]


def main():
    parser = argparse.ArgumentParser(description="Export GT-aligned stereo point clouds to PLY files.")
    parser.add_argument("configuration_path", help="Path to the run_surfslam-style YAML config.")
    parser.add_argument(
        "--output_dir",
        default=PROJECT_ROOT / "outputs" / "gt_stereo_plys",
        help="Directory to store PLY files (will be created if missing).",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for stereo inference.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional limit on number of stereo frames to export.")
    parser.add_argument("--start_time", type=float, default=None, help="Override settings.system.start_time (seconds from log start).")
    parser.add_argument("--end_time", type=float, default=None, help="Override settings.system.end_time (seconds from log start).")

    args = parser.parse_args()

    config = load_config(args.configuration_path)

    if "groundtruth_traj" not in config:
        raise RuntimeError("Configuration must provide a groundtruth_traj path for GT-aligned export.")

    settings_path = resolve_config_path(config["baseline"])
    settings = Settings.load_from_file(settings_path)
    if "changes" in config and config["changes"] is not None:
        settings.augment(config["changes"])

    im_scale_factor = settings.system.image_scale_factor
    calibration = load_calibration(config["dataset_family"], config["calibration"])
    if calibration is None:
        raise RuntimeError("A stereo calibration is required to run depth inference.")
    settings["calibration"] = calibration.to_dict(im_scale_factor)

    intrinsics = settings.calibration.camera_intrinsic.k

    stereo_estimator = DefomStereoEstimator(
        settings.mapper.stereo_estimator,
        settings.calibration,
        args.device,
    )

    h5_path = Path(os.path.expanduser(config["dataset"]))
    stereo_key = settings.system.h5_keys.stereo
    barometer_key = settings.system.h5_keys.barometer
    dvl_key = settings.system.h5_keys.dvl
    imu_key = settings.system.h5_keys.imu

    log_reader = H5LogReader(h5_path, imu_key, dvl_key, barometer_key, stereo_key)

    gt_file = Path(os.path.expanduser(config["groundtruth_traj"]))
    gt_df = pd.read_csv(gt_file, names=["timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w"], delimiter=" ")
    gt_poses, gt_timestamps = build_poses_from_df(gt_df, True)
    tf_buffer, tf_ts = build_buffer_from_poses(gt_poses, gt_timestamps)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_first_stamp = log_reader.first_stamp()
    start_filter = args.start_time if args.start_time is not None else settings.system.get("start_time", None)
    end_filter = args.end_time if args.end_time is not None else settings.system.get("end_time", None)

    stereo_delta_t = settings.tracker.frame_synthesis.stereo_delta_t_sec
    decimate_on_load = settings.tracker.frame_synthesis.decimate_on_load
    last_stereo_time = float("-inf")

    start_pose = None
    exported = 0

    for event in log_reader.events():
        timestamp = float(event.stamp)

        # Apply optional time filtering
        time_offset = timestamp - log_first_stamp
        if start_filter is not None and time_offset < start_filter:
            continue
        if end_filter is not None and time_offset > end_filter:
            break

        if event.type != "STEREO":
            continue

        if decimate_on_load and (timestamp - last_stereo_time) < stereo_delta_t:
            continue
        last_stereo_time = timestamp

        # Guard against extrapolating too far outside GT coverage
        if timestamp < tf_ts[0] or timestamp > tf_ts[-1]:
            continue

        T_world_cam = tf_buffer.lookup(timestamp)
        if T_world_cam is None:
            continue
        if start_pose is None:
            start_pose = T_world_cam

        rel_pose = torch.linalg.inv(start_pose) @ T_world_cam

        ts, left, right = log_reader.stereo()[event.index]
        left_rect, right_rect = calibration.rectify_images(left, right, im_scale_factor)

        left_torch = torch.from_numpy(left_rect).permute(2, 0, 1)
        right_torch = torch.from_numpy(right_rect).permute(2, 0, 1)
        stereo_img = StereoImage(left_torch, right_torch, ts)

        depth_img = stereo_estimator.infer(stereo_img)
        depth_tensor = depth_img.image

        points_cam, valid_mask = _backproject_depth(depth_tensor, intrinsics)
        if points_cam.numel() == 0:
            continue

        rel_pose = rel_pose.to(device=points_cam.device, dtype=points_cam.dtype)
        points_world = _transform_points(points_cam, rel_pose)

        points_np = points_world.detach().cpu().numpy()
        colors_np = left_rect.reshape(-1, 3)[valid_mask.cpu().numpy()] / 255.0

        if points_np.shape[0] == 0:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
        if colors_np.shape[0] == points_np.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors_np.astype(np.float64))

        out_path = output_dir / f"frame_{exported:06d}_ts_{timestamp:.3f}.ply"
        o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)

        exported += 1
        print(f"Saved {out_path}")

        if args.max_frames is not None and exported >= args.max_frames:
            break

    print(f"Finished writing {exported} PLYs to {output_dir}")


if __name__ == "__main__":
    main()
