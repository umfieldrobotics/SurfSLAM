#!/usr/bin/env python3
"""
Interpolate poses from a TUM trajectory file and compute the relative transform 
between two specified timestamps.

TUM format (per line):
    timestamp tx ty tz qx qy qz qw

Usage:
    python interpolate_tum_relative_transform.py <tum_file> <time_a> <time_b>
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def load_tum_trajectory(tum_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TUM trajectory file.
    
    Returns:
        timestamps: Array of timestamps (N,)
        positions: Array of positions (N, 3) - tx, ty, tz
        quaternions: Array of quaternions (N, 4) - qx, qy, qz, qw
    """
    data = np.loadtxt(str(tum_path), comments="#")
    
    if data.ndim == 1:
        data = data[None, :]
    
    if data.shape[1] != 8:
        raise ValueError(
            f"Expected 8 columns (timestamp + 3 translation + 4 quaternion), "
            f"got shape {data.shape}"
        )
    
    timestamps = data[:, 0]
    positions = data[:, 1:4]  # tx, ty, tz
    quaternions = data[:, 4:8]  # qx, qy, qz, qw
    
    return timestamps, positions, quaternions


def interpolate_pose(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    query_time: float
) -> np.ndarray:
    """
    Interpolate pose at a given timestamp.
    
    Args:
        timestamps: Array of timestamps (N,)
        positions: Array of positions (N, 3)
        quaternions: Array of quaternions (N, 4) - qx, qy, qz, qw
        query_time: Timestamp to interpolate at
        
    Returns:
        pose: 4x4 transformation matrix
    """
    if query_time < timestamps[0] or query_time > timestamps[-1]:
        raise ValueError(
            f"Query time {query_time:.6f} is outside trajectory range "
            f"[{timestamps[0]:.6f}, {timestamps[-1]:.6f}]"
        )
    
    # Find bounding indices
    if query_time == timestamps[0]:
        idx = 0
        alpha = 0.0
    elif query_time == timestamps[-1]:
        idx = len(timestamps) - 2
        alpha = 1.0
    else:
        idx = np.searchsorted(timestamps, query_time) - 1
        t0 = timestamps[idx]
        t1 = timestamps[idx + 1]
        alpha = (query_time - t0) / (t1 - t0)
    
    # Interpolate position (linear)
    pos0 = positions[idx]
    pos1 = positions[idx + 1]
    interp_position = pos0 + alpha * (pos1 - pos0)
    
    # Interpolate rotation (SLERP)
    quat0 = quaternions[idx]  # qx, qy, qz, qw
    quat1 = quaternions[idx + 1]
    
    # Convert to scipy Rotation format (scalar-last: x, y, z, w)
    rot0 = R.from_quat(quat0)
    rot1 = R.from_quat(quat1)
    
    # SLERP interpolation
    key_times = np.array([0.0, 1.0])
    key_rots = R.from_quat([quat0, quat1])
    slerp = Slerp(key_times, key_rots)
    interp_rotation = slerp([alpha])[0]
    
    # Build 4x4 transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = interp_rotation.as_matrix()
    pose[:3, 3] = interp_position
    
    return pose


def compute_relative_transform(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """
    Compute relative transform from pose_a to pose_b.
    
    T_b = T_a * T_ab
    T_ab = T_a^(-1) * T_b
    
    Args:
        pose_a: 4x4 transformation matrix at time a
        pose_b: 4x4 transformation matrix at time b
        
    Returns:
        relative_transform: 4x4 transformation matrix from a to b
    """
    return np.linalg.inv(pose_a) @ pose_b


def rotation_matrix_to_quaternion_xyzw(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (qx, qy, qz, qw)."""
    rot = R.from_matrix(rotation_matrix)
    return rot.as_quat()  # Returns [qx, qy, qz, qw]


def print_transform(transform: np.ndarray, name: str = "Transform"):
    """Print transformation matrix in a readable format."""
    print(f"\n{name}:")
    print("=" * 60)
    
    # Extract rotation and translation
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    
    # Convert to quaternion
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
    
    # Convert to euler angles (in degrees)
    rot = R.from_matrix(rotation)
    euler = rot.as_euler('xyz', degrees=True)
    
    print(f"Translation (tx, ty, tz):")
    print(f"  {translation[0]:12.6f} {translation[1]:12.6f} {translation[2]:12.6f}")
    
    print(f"\nRotation (quaternion qx, qy, qz, qw):")
    print(f"  {quaternion[0]:12.6f} {quaternion[1]:12.6f} {quaternion[2]:12.6f} {quaternion[3]:12.6f}")
    
    print(f"\nRotation (Euler XYZ in degrees):")
    print(f"  Roll:  {euler[0]:12.6f}°")
    print(f"  Pitch: {euler[1]:12.6f}°")
    print(f"  Yaw:   {euler[2]:12.6f}°")
    
    print(f"\n4x4 Matrix:")
    for row in transform:
        print(f"  {row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}")
    
    # Compute distance and rotation angle
    distance = np.linalg.norm(translation)
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    
    print(f"\nDistance: {distance:.6f} m")
    print(f"Rotation angle: {np.degrees(angle):.6f}°")
    print("=" * 60)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate poses from a TUM trajectory file and compute the "
            "relative transform between two specified timestamps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tum_file",
        type=Path,
        help="Input TUM trajectory file (timestamp tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "time_a",
        type=float,
        help="First timestamp",
    )
    parser.add_argument(
        "time_b",
        type=float,
        help="Second timestamp",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional: Save relative transform to file (4x4 matrix format)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    
    # Validate input
    if not args.tum_file.exists():
        print(f"Error: TUM file not found: {args.tum_file}", file=sys.stderr)
        sys.exit(1)
    
    # Load trajectory
    print(f"Loading trajectory from: {args.tum_file}")
    timestamps, positions, quaternions = load_tum_trajectory(args.tum_file)
    print(f"Loaded {len(timestamps)} poses")
    print(f"Time range: [{timestamps[0]:.6f}, {timestamps[-1]:.6f}]")
    
    # Interpolate poses
    print(f"\nInterpolating pose at time_a = {args.time_a:.6f}")
    try:
        pose_a = interpolate_pose(timestamps, positions, quaternions, args.time_a)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Interpolating pose at time_b = {args.time_b:.6f}")
    try:
        pose_b = interpolate_pose(timestamps, positions, quaternions, args.time_b)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Compute relative transform
    relative_transform = compute_relative_transform(pose_a, pose_b)
    
    # Print results
    print_transform(pose_a, f"Pose at time_a ({args.time_a:.6f})")
    print_transform(pose_b, f"Pose at time_b ({args.time_b:.6f})")
    print_transform(relative_transform, f"Relative Transform (a → b)")
    
    # Save to file if requested
    if args.output:
        np.savetxt(args.output, relative_transform, fmt='%.9f')
        print(f"\nSaved relative transform to: {args.output}")


if __name__ == "__main__":
    main()
