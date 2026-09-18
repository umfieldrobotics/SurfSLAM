#!/usr/bin/env python3
"""
Project Point Cloud based on Gravity Direction
==============================================
Projects a point cloud to be flat in the gravity direction determined from IMU readings
at each timestamp of the ground truth TUM poses.

Usage:
    python project_pointcloud_gravity.py --gt_poses path/to/gt.tum --h5_file path/to/data.h5 --gt_pointcloud path/to/pointcloud.ply
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "surfslam"))

from examples.utils.h5_log_reader import H5LogReader
from examples.utils.pose_utils import load_calibration
from surfslam.common.dataset_paths import load_config, resolve_config_path
from surfslam.common.settings import Settings


def load_tum_poses(tum_path: str) -> List[Tuple[float, np.ndarray]]:
    """
    Load poses from TUM format file.
    
    Args:
        tum_path: Path to TUM format trajectory file
        
    Returns:
        List of (timestamp, 4x4 pose matrix) tuples
    """
    raw = np.loadtxt(str(tum_path))
    if raw.ndim == 1:
        raw = raw[None, :]

    poses = []
    for row in raw:
        timestamp = row[0]
        xyz = row[1:4]
        quat_xyzw = row[4:8]
        T = np.eye(4)
        T[:3, 3] = xyz
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        poses.append((timestamp, T))
    return poses


def find_closest_imu_reading(target_ts: float, imu_events, imu_times, log_reader, max_diff: float = 0.01):
    """
    Find the closest IMU reading to a target timestamp.
    
    Args:
        target_ts: Target timestamp
        imu_events: List of IMU events
        imu_times: Array of IMU timestamps
        log_reader: H5LogReader instance
        max_diff: Maximum time difference to accept
        
    Returns:
        Dictionary with IMU data or None if no match found
    """
    if len(imu_times) == 0:
        return None

    time_diffs = np.abs(imu_times - target_ts)
    min_idx = np.argmin(time_diffs)

    if time_diffs[min_idx] > max_diff:
        return None

    event = imu_events[min_idx]
    imu_data = log_reader.imu()[event.index]
    
    return {
        'timestamp': imu_data.timestamp,
        'accel_x': imu_data.ax,
        'accel_y': imu_data.ay,
        'accel_z': imu_data.az
    }


def compute_gravity_direction(imu_data: dict, T_world_cam: np.ndarray, R_cam_imu: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute gravity direction in world frame from IMU acceleration.
    
    Args:
        imu_data: Dictionary with accel_x, accel_y, accel_z
        T_world_cam: 4x4 transformation matrix from camera to world
        R_cam_imu: 3x3 rotation matrix from IMU to camera (optional)
        
    Returns:
        3D unit vector representing gravity direction in world frame
    """
    if imu_data is None:
        return None

    # Get acceleration in IMU frame
    accel_imu = np.array(
        [imu_data["accel_x"], imu_data["accel_y"], imu_data["accel_z"]],
        dtype=np.float64
    )
    accel_norm = float(np.linalg.norm(accel_imu))
    
    if accel_norm < 1e-9 or not np.isfinite(accel_norm):
        return None

    # Normalize to get direction
    accel_dir_imu = accel_imu / accel_norm
    
    # Gravity is opposite to acceleration direction
    gravity_dir_imu = -accel_dir_imu

    # Default IMU to camera rotation if not provided
    if R_cam_imu is None:
        R_cam_imu = np.array(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0]],
            dtype=np.float64
        )
    else:
        R_cam_imu = np.asarray(R_cam_imu, dtype=np.float64)
        if R_cam_imu.shape != (3, 3):
            raise ValueError("R_cam_imu must be 3x3.")

    # Transform to camera frame
    gravity_dir_cam = R_cam_imu @ gravity_dir_imu
    gravity_dir_cam /= np.linalg.norm(gravity_dir_cam)

    # Get rotation from world to camera
    T_world_cam = np.asarray(T_world_cam, dtype=np.float64)
    if T_world_cam.shape != (4, 4):
        raise ValueError("T_world_cam must be 4x4.")

    R_world_cam = T_world_cam[:3, :3]

    # Transform to world frame
    gravity_dir_world = R_world_cam @ gravity_dir_cam
    gravity_dir_world /= np.linalg.norm(gravity_dir_world)

    return gravity_dir_world


def compute_mean_gravity_direction(
    gt_poses: List[Tuple[float, np.ndarray]],
    imu_events,
    imu_times,
    log_reader,
    max_time_diff: float = 0.01
) -> np.ndarray:
    """
    Compute mean gravity direction across all pose timestamps.
    
    Args:
        gt_poses: List of (timestamp, pose) tuples
        imu_events: List of IMU events
        imu_times: Array of IMU timestamps
        log_reader: H5LogReader instance
        max_time_diff: Maximum time difference for IMU matching
        
    Returns:
        3D unit vector representing mean gravity direction in world frame
    """
    gravity_vectors = []
    
    for pose_ts, pose in gt_poses:
        imu_data = find_closest_imu_reading(
            pose_ts, imu_events, imu_times, log_reader, max_time_diff
        )
        
        if imu_data is not None:
            gravity_dir = compute_gravity_direction(imu_data, pose)
            if gravity_dir is not None:
                gravity_vectors.append(gravity_dir)
    
    if len(gravity_vectors) == 0:
        raise RuntimeError("No valid gravity vectors computed")
    
    # Compute mean and normalize
    mean_gravity = np.mean(gravity_vectors, axis=0)
    mean_gravity /= np.linalg.norm(mean_gravity)
    
    print(f"Computed mean gravity from {len(gravity_vectors)}/{len(gt_poses)} poses")
    print(f"Mean gravity direction: [{mean_gravity[0]:.4f}, {mean_gravity[1]:.4f}, {mean_gravity[2]:.4f}]")
    
    return mean_gravity


def compute_projection_transform(gravity_dir: np.ndarray, target_axis: np.ndarray = None) -> np.ndarray:
    """
    Compute transformation to align gravity direction with target axis.
    
    Args:
        gravity_dir: Current gravity direction (3D unit vector)
        target_axis: Target axis (default: [0, 0, -1] pointing down)
        
    Returns:
        4x4 transformation matrix
    """
    if target_axis is None:
        # Default: align gravity to point down along -Z axis
        target_axis = np.array([0.0, 0.0, -1.0])
    
    target_axis = target_axis / np.linalg.norm(target_axis)
    
    # Compute rotation to align gravity_dir with target_axis
    # Use Rodrigues' rotation formula
    v = np.cross(gravity_dir, target_axis)
    c = np.dot(gravity_dir, target_axis)
    
    # Check if vectors are already aligned or opposite
    if np.abs(c - 1.0) < 1e-9:
        # Already aligned
        R_align = np.eye(3)
    elif np.abs(c + 1.0) < 1e-9:
        # Opposite directions - rotate 180 degrees around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(gravity_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, gravity_dir) * gravity_dir
        perp = perp / np.linalg.norm(perp)
        R_align = 2 * np.outer(perp, perp) - np.eye(3)
    else:
        # General case
        s = np.linalg.norm(v)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    # Create 4x4 transformation
    T = np.eye(4)
    T[:3, :3] = R_align
    
    return T


def project_pointcloud(
    pcd: o3d.geometry.PointCloud,
    gravity_dir: np.ndarray,
    output_path: str = None
) -> o3d.geometry.PointCloud:
    """
    Project point cloud to be flat in gravity direction.
    
    Args:
        pcd: Input point cloud
        gravity_dir: Gravity direction in world frame
        output_path: Optional path to save projected point cloud
        
    Returns:
        Projected point cloud
    """
    # Get point cloud points as numpy array
    points = np.asarray(pcd.points)
    
    # Normalize gravity direction
    gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
    
    # Project points onto plane perpendicular to gravity
    # For each point p, compute: p_proj = p - (p · g) * g
    # This removes the component of each point along the gravity direction
    dots = np.dot(points, gravity_dir)  # Shape: (N,)
    projected_points = points - np.outer(dots, gravity_dir)
    
    # Create new point cloud with projected points
    pcd_projected = o3d.geometry.PointCloud()
    pcd_projected.points = o3d.utility.Vector3dVector(projected_points)
    
    # Copy colors if they exist
    if pcd.has_colors():
        pcd_projected.colors = pcd.colors
    
    # Copy normals if they exist
    if pcd.has_normals():
        pcd_projected.normals = pcd.normals
    
    if output_path:
        o3d.io.write_point_cloud(output_path, pcd_projected)
        print(f"Saved projected point cloud to: {output_path}")
    
    return pcd_projected


def main():
    parser = argparse.ArgumentParser(
        description="Project point cloud based on gravity direction from IMU")
    parser.add_argument("--gt_poses", type=str, required=True,
                        help="Path to ground truth TUM format poses")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="Path to HDF5 data file")
    parser.add_argument("--gt_pointcloud", type=str, required=True,
                        help="Path to ground truth point cloud")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config for H5 keys (if not provided, uses defaults)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for projected point cloud (default: input_projected.ply)")
    parser.add_argument("--max_time_diff", type=float, default=0.01,
                        help="Maximum time difference for IMU matching (seconds)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize original and projected point clouds")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        input_path = Path(args.gt_pointcloud)
        args.output = str(input_path.parent / f"{input_path.stem}_projected{input_path.suffix}")
    
    print("="*80)
    print("Point Cloud Gravity Projection")
    print("="*80)
    print(f"GT Poses:      {args.gt_poses}")
    print(f"H5 File:       {args.h5_file}")
    print(f"GT Pointcloud: {args.gt_pointcloud}")
    print(f"Output:        {args.output}")
    print()
    
    # Load ground truth poses
    print("Loading ground truth poses...")
    gt_poses = load_tum_poses(args.gt_poses)
    print(f"Loaded {len(gt_poses)} poses")
    
    # Load HDF5 data
    print("\nLoading HDF5 data...")
    if args.config:
        config = load_config(args.config)
        baseline_path = resolve_config_path(config['baseline'])
        settings = Settings.load_from_file(baseline_path)
        if 'changes' in config:
            settings.augment(config['changes'])
        h5_keys = settings.system.h5_keys
        
        log_reader = H5LogReader(
            Path(args.h5_file),
            h5_keys.imu,
            h5_keys.dvl,
            h5_keys.barometer,
            h5_keys.stereo
        )
    else:
        # Use default keys - need to load from default settings
        baseline_path = resolve_config_path("defaults.yaml")
        settings = Settings.load_from_file(baseline_path)
        h5_keys = settings.system.h5_keys
        
        log_reader = H5LogReader(
            Path(args.h5_file),
            h5_keys.imu,
            h5_keys.dvl,
            h5_keys.barometer,
            h5_keys.stereo
        )
    
    # Get IMU events
    print("Loading IMU data...")
    imu_events = [e for e in log_reader.events() if e.type == "IMU"]
    imu_times = np.array([e.stamp for e in imu_events])
    print(f"Loaded {len(imu_events)} IMU readings")
    
    # Compute mean gravity direction
    print("\nComputing gravity direction...")
    mean_gravity = compute_mean_gravity_direction(
        gt_poses, imu_events, imu_times, log_reader, args.max_time_diff
    )
    
    # Load point cloud
    print("\nLoading point cloud...")
    pcd = o3d.io.read_point_cloud(args.gt_pointcloud)
    print(f"Loaded point cloud with {len(pcd.points)} points")
    
    # Project point cloud
    print("\nProjecting point cloud...")
    pcd_projected = project_pointcloud(pcd, mean_gravity, args.output)
    
    print("\n" + "="*80)
    print("Projection complete!")
    print("="*80)
    
    # Visualize if requested
    if args.visualize:
        print("\nVisualizing results...")
        
        # Color original in blue, projected in red
        pcd_vis = pcd.paint_uniform_color([0.0, 0.0, 1.0])
        pcd_proj_vis = pcd_projected.paint_uniform_color([1.0, 0.0, 0.0])
        
        # Create coordinate frames
        coord_original = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        coord_projected = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        coord_projected.translate([5.0, 0.0, 0.0])  # Offset for visibility
        
        o3d.visualization.draw_geometries(
            [pcd_vis, pcd_proj_vis, coord_original, coord_projected],
            window_name="Original (Blue) vs Projected (Red)",
            width=1920,
            height=1080
        )


if __name__ == "__main__":
    main()
