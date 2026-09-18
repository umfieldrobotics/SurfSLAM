#!/usr/bin/env python3
"""
Stereo SLAM Visualization for Paper Figures

Generates 3D visualizations showing:
1. Estimated trajectory from TUM format file
2. Stereo image pairs floating in 3D at specified timestamps
3. Depth point clouds from stereo pairs using DEFOM estimator

Usage:
    python slam_visualization.py \
        --trajectory trajectory.txt \
        --sequence_config config.yaml \
        --keyframes 0.0 5.0 10.0 \
        --start_time 0.0 \
        --end_time 30.0 \
        --output figure.html
"""

import argparse
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
import cv2


# Set up paths
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "surfslam"))

from surfslam.common.dataset_paths import load_config, resolve_config_path
from examples.tbnms.tbnms_calibration import TBNMSCalibration
from examples.utils.pose_utils import load_calibration
from common.settings import Settings
from mapping.stereo.stereo_estimation_defom import DefomStereoEstimator
from examples.utils.h5_log_reader import H5LogReader
from common.sensors import StereoImage

# ============================================================================
# TUM Trajectory Loading
# ============================================================================

@dataclass
class Pose:
    timestamp: float
    position: np.ndarray  # [x, y, z]
    quaternion: np.ndarray  # [qx, qy, qz, qw]
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix from quaternion."""
        r = Rotation.from_quat(self.quaternion)  # scipy uses [x, y, z, w]
        return r.as_matrix()
    
    @property
    def transform_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        return T


def load_tum_trajectory(path: str) -> List[Pose]:
    """
    Load trajectory from TUM format file.
    Format: timestamp tx ty tz qx qy qz qw
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            poses.append(Pose(
                timestamp=timestamp,
                position=np.array([tx, ty, tz]),
                quaternion=np.array([qx, qy, qz, qw])
            ))
    
    return sorted(poses, key=lambda p: p.timestamp)


def filter_trajectory_by_time(poses: List[Pose], 
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None) -> List[Pose]:
    """Filter poses to a time range."""
    if start_time is None and end_time is None:
        return poses
    
    filtered = []
    for p in poses:
        if start_time is not None and p.timestamp < start_time:
            continue
        if end_time is not None and p.timestamp > end_time:
            continue
        filtered.append(p)
    return filtered


def interpolate_pose(poses: List[Pose], timestamp: float) -> Optional[Pose]:
    """Linearly interpolate pose at a given timestamp."""
    if not poses:
        return None
    
    before, after = None, None
    for i, p in enumerate(poses):
        if p.timestamp <= timestamp:
            before = p
        if p.timestamp >= timestamp and after is None:
            after = p
            break
    
    if before is None and after is None:
        return None
    if before is None:
        return after
    if after is None:
        return before
    if before.timestamp == after.timestamp:
        return before
    
    t = (timestamp - before.timestamp) / (after.timestamp - before.timestamp)
    position = (1 - t) * before.position + t * after.position
    
    r_before = Rotation.from_quat(before.quaternion)
    r_after = Rotation.from_quat(after.quaternion)
    r_interp = Rotation.from_quat(
        Rotation.from_rotvec(t * (r_after * r_before.inv()).as_rotvec()).as_quat()
    ) * r_before
    
    return Pose(
        timestamp=timestamp,
        position=position,
        quaternion=r_interp.as_quat()
    )


# ============================================================================
# Point Cloud Utilities
# ============================================================================

def depth_to_pointcloud(depth: np.ndarray,
                        left_image: np.ndarray,
                        fx: float,
                        fy: float,
                        cx: float,
                        cy: float,
                        max_depth: float = 20.0,
                        min_depth: float = 0.1,
                        subsample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth map to colored 3D point cloud.
    
    Args:
        depth: HxW depth map (meters)
        left_image: HxWx3 color image
        fx, fy: focal lengths in pixels
        cx, cy: principal point
        max_depth: maximum depth to include
        min_depth: minimum depth to include
        subsample: subsample factor for efficiency
        
    Returns:
        points: Nx3 array of 3D points in camera frame
        colors: Nx3 array of RGB colors [0-1]
    """
    H, W = depth.shape[:2]
    
    u = np.arange(0, W, subsample)
    v = np.arange(0, H, subsample)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    
    z = depth[v, u]
    
    # Filter by depth range
    valid = (z > min_depth) & (z < max_depth)
    u, v, z = u[valid], v[valid], z[valid]
    
    # Back-project to 3D (camera frame: x-right, y-down, z-forward)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.stack([x, y, z], axis=1)
    
    # Get colors
    colors = left_image[v, u].astype(np.float32) / 255.0
    
    return points, colors


def transform_points(points: np.ndarray, pose: Pose) -> np.ndarray:
    """Transform points from camera frame to world frame."""
    R = pose.rotation_matrix
    t = pose.position
    return (R @ points.T).T + t


def create_image_plane_vertices(pose: Pose, 
                                 width: float, 
                                 height: float,
                                 distance: float,
                                 lateral_offset: float = 0.0) -> np.ndarray:
    """Create 4 corner vertices for an image plane in world coordinates."""
    half_w = width / 2
    half_h = height / 2
    
    corners_cam = np.array([
        [-half_w + lateral_offset, -half_h, distance],
        [half_w + lateral_offset, -half_h, distance],
        [half_w + lateral_offset, half_h, distance],
        [-half_w + lateral_offset, half_h, distance],
    ])
    
    R = pose.rotation_matrix
    t = pose.position
    corners_world = (R @ corners_cam.T).T + t
    
    return corners_world


# ============================================================================
# Plotly Visualization Traces
# ============================================================================

def create_trajectory_trace(poses: List[Pose], 
                           name: str = "Trajectory",
                           color: str = 'blue',
                           width: int = 3) -> go.Scatter3d:
    """Create a 3D line trace for the trajectory."""
    positions = np.array([p.position for p in poses])
    
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines',
        name=name,
        line=dict(color=color, width=width),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    )


def create_camera_frustum_trace(pose: Pose,
                                 scale: float = 0.3,
                                 color: str = 'red',
                                 name: str = "Camera") -> List[go.Scatter3d]:
    """Create camera frustum visualization."""
    d = scale
    w = scale * 0.6
    h = scale * 0.4
    
    corners_cam = np.array([
        [0, 0, 0],
        [-w, -h, d],
        [w, -h, d],
        [w, h, d],
        [-w, h, d],
    ])
    
    R = pose.rotation_matrix
    t = pose.position
    corners_world = (R @ corners_cam.T).T + t
    
    traces = []
    
    for i in range(1, 5):
        traces.append(go.Scatter3d(
            x=[corners_world[0, 0], corners_world[i, 0]],
            y=[corners_world[0, 1], corners_world[i, 1]],
            z=[corners_world[0, 2], corners_world[i, 2]],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    far_indices = [1, 2, 3, 4, 1]
    traces.append(go.Scatter3d(
        x=[corners_world[i, 0] for i in far_indices],
        y=[corners_world[i, 1] for i in far_indices],
        z=[corners_world[i, 2] for i in far_indices],
        mode='lines',
        line=dict(color=color, width=2),
        name=name,
        hoverinfo='skip'
    ))
    
    return traces


def create_image_plane_trace(pose: Pose,
                              image: np.ndarray,
                              distance: float = 1.0,
                              lateral_offset: float = 0.0,
                              name: str = "Image",
                              opacity: float = 0.9) -> go.Surface:
    """Create a textured image plane in 3D using surface plot with RGB color."""
    H, W = image.shape[:2]
    aspect = H / W
    
    plane_width = 0.8
    plane_height = plane_width * aspect
    
    corners = create_image_plane_vertices(
        pose, plane_width, plane_height, distance, lateral_offset
    )
    
    # Resize image for efficiency
    target_width = 50
    target_height = int(target_width * aspect)
    img_small = cv2.resize(image, (target_width, target_height))
    
    # Convert BGR to RGB if needed (OpenCV loads as BGR)
    if len(img_small.shape) == 3 and img_small.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale - convert to RGB
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_GRAY2RGB)
    
    res = img_rgb.shape[0]
    res_w = img_rgb.shape[1]
    
    u = np.linspace(0, 1, res_w)
    v = np.linspace(0, 1, res)
    
    x_grid = np.zeros((res, res_w))
    y_grid = np.zeros((res, res_w))
    z_grid = np.zeros((res, res_w))
    
    for i, vi in enumerate(v):
        for j, uj in enumerate(u):
            top = corners[0] * (1 - uj) + corners[1] * uj
            bottom = corners[3] * (1 - uj) + corners[2] * uj
            pt = top * (1 - vi) + bottom * vi
            x_grid[i, j] = pt[0]
            y_grid[i, j] = pt[1]
            z_grid[i, j] = pt[2]
    
    # Create custom RGB colorscale from the image
    # Normalize image to 0-1 range
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Create a surface color array - Plotly uses surfacecolor with a colorscale
    # For true RGB, we need to convert RGB to a single value and use a custom colorscale
    # Alternative: use Image trace with Mesh3d, but Surface with facecolor is simpler
    
    # Convert RGB image to a list of RGB strings for each pixel
    # We'll create a custom approach using surfacecolor mapped to pixel indices
    # and a colorscale that maps those indices to colors
    
    # Flatten the image to create a unique color for each pixel position
    # Create intensity map (we'll use this for surfacecolor)
    # and override with custom colors using a discrete colorscale
    
    # Simpler approach: encode RGB into a single float and decode in colorscale
    # Pack RGB into a single value: R + G*256 + B*65536, then normalize
    r_channel = img_rgb[:, :, 0].astype(np.float32)
    g_channel = img_rgb[:, :, 1].astype(np.float32)
    b_channel = img_rgb[:, :, 2].astype(np.float32)
    
    # Create a unique index for each pixel
    n_colors = res * res_w
    pixel_indices = np.arange(n_colors).reshape(res, res_w).astype(np.float32)
    pixel_indices_normalized = pixel_indices / (n_colors - 1) if n_colors > 1 else pixel_indices
    
    # Build a custom colorscale that maps each normalized index to its RGB color
    colorscale = []
    for i in range(res):
        for j in range(res_w):
            idx = i * res_w + j
            norm_idx = idx / (n_colors - 1) if n_colors > 1 else 0
            r, g, b = img_rgb[i, j]
            color_str = f'rgb({r},{g},{b})'
            colorscale.append([norm_idx, color_str])
    
    # Ensure colorscale starts at 0 and ends at 1
    if colorscale:
        colorscale[0] = [0, colorscale[0][1]]
        colorscale[-1] = [1, colorscale[-1][1]]
    
    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=pixel_indices_normalized,
        colorscale=colorscale,
        showscale=False,
        name=name,
        opacity=opacity,
        hoverinfo='skip',
        cmin=0,
        cmax=1
    )


def create_pointcloud_trace(points: np.ndarray,
                            colors: np.ndarray,
                            name: str = "Point Cloud",
                            size: int = 2) -> go.Scatter3d:
    """Create a 3D scatter plot for point cloud."""
    color_strings = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' 
                     for c in colors]
    
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color_strings,
            opacity=0.8
        ),
        name=name,
        hoverinfo='skip'
    )


# ============================================================================
# DEFOM Stereo Processor
# ============================================================================

class StereoProcessor:
    """Wrapper for DEFOM stereo depth estimation."""
    
    def __init__(self, sequence_config_path: str, device: str = 'cuda:0'):
        """
        Initialize stereo processor from sequence config.
        
        Args:
            sequence_config_path: Path to YAML config file
            device: CUDA device string
        """
        import torch
        
        self.device = device
        self.torch = torch
        
        self.config = load_config(sequence_config_path)

        self.H5LogReader = H5LogReader

        baseline_settings_path = resolve_config_path(self.config['baseline'])
        print(f"Loading baseline settings from {baseline_settings_path}")
        
        self.settings = Settings.load_from_file(baseline_settings_path)
        if 'changes' in self.config:
            self.settings.augment(self.config['changes'])
        
        self.im_scale_factor = self.settings.system.image_scale_factor
        self.calibration = load_calibration(
            self.config["dataset_family"], 
            self.config["calibration"]
        )
        
        self.settings['calibration'] = self.calibration.to_dict(self.im_scale_factor)
        
        # Extract calibration parameters
        calib_dict = self.settings['calibration']
        stereo_calib = calib_dict.get('stereo', calib_dict)
        k = calib_dict['camera_intrinsic']['k']
        self.fx = k[0,0]
        self.fy = k[1,1]
        self.cx = k[0,2]
        self.cy = k[1,2]
        self.baseline = stereo_calib.get('baseline_m', 0.1)
        
        print(f"Calibration: fx={self.fx:.1f}, fy={self.fy:.1f}, "
              f"cx={self.cx:.1f}, cy={self.cy:.1f}, baseline={self.baseline:.4f}m")
        
        # Initialize stereo matcher (returns depth directly!)
        self.stereo_matcher = DefomStereoEstimator(
            self.settings.mapper.stereo_estimator,
            self.settings.calibration,
            device
        )
        
        # H5 keys
        self.stereo_key = self.settings.system.h5_keys.stereo
        self.barometer_key = self.settings.system.h5_keys.barometer
        self.dvl_key = self.settings.system.h5_keys.dvl
        self.imu_key = self.settings.system.h5_keys.imu
        
        self.log_reader = None
    
    def load_h5_data(self, h5_path: Optional[str] = None):
        """Load H5 data file."""
        if h5_path is None:
            h5_path = os.path.expanduser(self.config["dataset"])
        
        h5_path = Path(h5_path)
        self.log_reader = self.H5LogReader(
            h5_path, 
            self.imu_key, 
            self.dvl_key, 
            self.barometer_key, 
            self.stereo_key
        )
        print(f"Loaded H5 data with {len(self.log_reader.events())} events")
        
        stereo_count = sum(1 for e in self.log_reader.events() if e.type == "STEREO")
        print(f"Found {stereo_count} stereo frames")
    
    def get_stereo_timestamps(self) -> List[float]:
        """Get all stereo frame timestamps."""
        if self.log_reader is None:
            return []
        return [e.stamp for e in self.log_reader.events() if e.type == "STEREO"]
    
    def get_closest_stereo_frame(self, timestamp: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """Get the stereo frame closest to the given timestamp."""
        if self.log_reader is None:
            raise ValueError("H5 data not loaded")
        
        best_idx = None
        best_diff = float('inf')
        best_stamp = None
        
        for event in self.log_reader.events():
            if event.type != "STEREO":
                continue
            diff = abs(event.stamp - timestamp)
            if diff < best_diff:
                best_diff = diff
                best_idx = event.index
                best_stamp = event.stamp
        
        if best_idx is None:
            raise ValueError("No stereo frames found")
        
        ts, left, right = self.log_reader.stereo()[best_idx]
        
        # Rectify images
        left_rect, right_rect = self.calibration.rectify_images(
            left, right, self.im_scale_factor
        )
        
        return best_stamp, left_rect, right_rect
    
    def compute_depth(self, left: np.ndarray, right: np.ndarray, 
                      timestamp: float) -> np.ndarray:
        """
        Compute depth using DEFOM estimator.
        
        Note: DEFOM's infer() returns depth directly (not disparity).
        
        Args:
            left: Rectified left image (H, W, C)
            right: Rectified right image (H, W, C)
            timestamp: Frame timestamp
            
        Returns:
            Depth map (H, W) in meters
        """
        
        # Convert to torch tensors (C, H, W)
        left_tensor = self.torch.from_numpy(left).permute(2, 0, 1)
        right_tensor = self.torch.from_numpy(right).permute(2, 0, 1)
        
        stereo_img = StereoImage(left_tensor, right_tensor, timestamp)
        
        # DEFOM returns Image with depth (not disparity)
        result = self.stereo_matcher.infer(stereo_img)
        
        depth = result.image
        if depth.dim() == 3:
            depth = depth.squeeze(0)
        
        return depth.cpu().numpy()
    
    def get_pointcloud(self, timestamp: float, 
                       max_depth: float = 10.0,
                       min_depth: float = 0.1,
                       subsample: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get point cloud for a stereo frame.
        
        Returns:
            (points, colors, left_image, right_image)
        """
        actual_ts, left, right = self.get_closest_stereo_frame(timestamp)
        
        print(f"Computing depth for t={actual_ts:.3f} (requested {timestamp:.3f})")
        depth = self.compute_depth(left, right, actual_ts)
        
        points, colors = depth_to_pointcloud(
            depth, left,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            max_depth=max_depth,
            min_depth=min_depth,
            subsample=subsample
        )
        
        print(f"Generated {len(points)} points")
        
        return points, colors, left, right

# ============================================================================
# Main Visualization Builder
# ============================================================================

class StereoSLAMVisualizer:
    """Main class for building the visualization."""
    
    def __init__(self, 
                 trajectory_path: str,
                 sequence_config: str,
                 device: str = 'cuda:0'):
        """
        Initialize the visualizer.
        
        Args:
            trajectory_path: Path to TUM format trajectory file
            sequence_config: Path to sequence config YAML (for DEFOM)
            device: CUDA device for stereo inference
        """
        self.poses = load_tum_trajectory(trajectory_path)
        print(f"Loaded {len(self.poses)} poses from trajectory")
        
        if self.poses:
            print(f"Time range: {self.poses[0].timestamp:.2f} - {self.poses[-1].timestamp:.2f}")
        
        self.stereo_processor = None
        self.stereo_processor = StereoProcessor(sequence_config, device)

        
        self.traces = []
    
    def load_h5_data(self, h5_path: Optional[str] = None):
        """Load H5 data file."""
        self.stereo_processor.load_h5_data(h5_path)
    
    def add_trajectory(self,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       color: str = 'blue',
                       name: str = "Trajectory"):
        """Add trajectory to visualization."""
        poses = filter_trajectory_by_time(self.poses, start_time, end_time)
        if not poses:
            print("Warning: No poses in specified time range")
            return
        
        print(f"Adding trajectory with {len(poses)} poses")
        self.traces.append(create_trajectory_trace(poses, name=name, color=color))
    
    def add_keyframe_cameras(self,
                             timestamps: List[float],
                             scale: float = 0.3,
                             color: str = 'red'):
        """Add camera frustum visualizations at specified timestamps."""
        for ts in timestamps:
            pose = interpolate_pose(self.poses, ts)
            if pose is None:
                print(f"Warning: Could not interpolate pose at t={ts}")
                continue
            
            frustum_traces = create_camera_frustum_trace(
                pose, scale=scale, color=color, name=f"Camera t={ts:.2f}"
            )
            self.traces.extend(frustum_traces)
    
    def add_stereo_images(self,
                          timestamps: List[float],
                          distance: float = 1.0,
                          separation: float = 0.5):
        """Add stereo image pairs floating in 3D at specified timestamps."""
        for ts in timestamps:
            pose = interpolate_pose(self.poses, ts)
            if pose is None:
                print(f"Warning: Could not interpolate pose at t={ts}")
                continue
            
            try:
                actual_ts, left, right = self.stereo_processor.get_closest_stereo_frame(ts)
            except Exception as e:
                print(f"Warning: Could not get stereo frame at t={ts}: {e}")
                continue
            
            left_trace = create_image_plane_trace(
                pose, left, distance=distance, 
                lateral_offset=-separation/2,
                name=f"Left t={ts:.2f}"
            )
            self.traces.append(left_trace)
            
            right_trace = create_image_plane_trace(
                pose, right, distance=distance,
                lateral_offset=separation/2,
                name=f"Right t={ts:.2f}"
            )
            self.traces.append(right_trace)
    
    def add_depth_pointclouds(self,
                               timestamps: List[float],
                               max_depth: float = 10.0,
                               min_depth: float = 0.1,
                               subsample: int = 8,
                               point_size: int = 2):
        """Add depth point clouds from stereo pairs at specified timestamps."""
        for ts in timestamps:
            pose = interpolate_pose(self.poses, ts)
            if pose is None:
                print(f"Warning: Could not interpolate pose at t={ts}")
                continue
            
            try:
                points, colors, _, _ = self.stereo_processor.get_pointcloud(
                    ts, max_depth=max_depth, min_depth=min_depth, subsample=subsample
                )
            except Exception as e:
                print(f"Warning: Could not get point cloud at t={ts}: {e}")
                continue
            
            if len(points) == 0:
                print(f"Warning: No valid points at t={ts}")
                continue
            
            points_world = transform_points(points, pose)
            
            print(f"Adding {len(points_world)} points for t={ts:.2f}")
            
            pc_trace = create_pointcloud_trace(
                points_world, colors,
                name=f"Depth t={ts:.2f}",
                size=point_size
            )
            self.traces.append(pc_trace)
    
    def build_figure(self, title: str = "Stereo SLAM Visualization") -> go.Figure:
        """Build and return the Plotly figure."""
        fig = go.Figure(data=self.traces)
        
        all_points = []
        for trace in self.traces:
            if hasattr(trace, 'x') and trace.x is not None:
                if isinstance(trace.x, np.ndarray):
                    x = trace.x.flatten()
                    y = trace.y.flatten()
                    z = trace.z.flatten()
                else:
                    x = np.array(trace.x)
                    y = np.array(trace.y)
                    z = np.array(trace.z)
                pts = np.stack([x, y, z], axis=1)
                all_points.append(pts)
        
        if all_points:
            all_points = np.vstack(all_points)
            min_vals = np.min(all_points, axis=0)
            max_vals = np.max(all_points, axis=0)
            center = (min_vals + max_vals) / 2
            # Use the largest span for all axes to ensure equal scaling
            max_range = np.max(max_vals - min_vals) * 1.2 / 2
        else:
            center = np.zeros(3)
            max_range = 10.0

        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[center[0] - max_range, center[0] + max_range]),
            yaxis=dict(range=[center[1] - max_range, center[1] + max_range]),
            zaxis=dict(range=[center[2] - max_range, center[2] + max_range]),
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        )
        
        return fig
    
    def save(self, output_path: str, title: str = "Stereo SLAM Visualization"):
        """Build and save the visualization."""
        fig = self.build_figure(title=title)
        
        output_path = Path(output_path)
        if output_path.suffix == '.html':
            fig.write_html(str(output_path), include_plotlyjs='cdn')
            print(f"Saved interactive HTML to {output_path}")
        elif output_path.suffix == '.png':
            fig.write_image(str(output_path), width=1920, height=1080, scale=2)
            print(f"Saved PNG to {output_path}")
        elif output_path.suffix == '.pdf':
            fig.write_image(str(output_path), width=1920, height=1080)
            print(f"Saved PDF to {output_path}")
        else:
            output_html = output_path.with_suffix('.html')
            fig.write_html(str(output_html), include_plotlyjs='cdn')
            print(f"Saved interactive HTML to {output_html}")
        
        return fig


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D visualization for stereo SLAM paper figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With sequence config (uses DEFOM stereo estimator)
  python slam_visualization.py \\
      --trajectory traj.txt \\
      --sequence_config config.yaml \\
      --keyframes 0.0 5.0 10.0 \\
      --output viz.html
  
  # Basic trajectory only
  python slam_visualization.py \\
      --trajectory traj.txt \\
      --output viz.html
        """
    )
    
    parser.add_argument("--trajectory", "-t", required=True,
                        help="Path to TUM format trajectory file")
    parser.add_argument("--sequence_config", "-c", required=True,
                        help="Path to sequence config YAML (enables DEFOM stereo)")
    parser.add_argument("--h5_path", "-d",
                        help="Path to HDF5 data file (optional, uses config if not set)")
    
    parser.add_argument("--start_time", "-s", type=float, default=None,
                        help="Start timestamp for visualization")
    parser.add_argument("--end_time", "-e", type=float, default=None,
                        help="End timestamp for visualization")
    
    parser.add_argument("--keyframes", "-k", type=float, nargs="+", default=[],
                        help="Timestamps for keyframe visualization")
    
    parser.add_argument("--image_distance", type=float, default=1.0,
                        help="Distance to place image planes in front of camera")
    parser.add_argument("--image_separation", type=float, default=0.5,
                        help="Lateral separation between stereo images")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Maximum depth for point clouds")
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="Minimum depth for point clouds")
    parser.add_argument("--subsample", type=int, default=8,
                        help="Subsample factor for point clouds")
    parser.add_argument("--frustum_scale", type=float, default=0.3,
                        help="Scale for camera frustum visualization")
    parser.add_argument("--point_size", type=int, default=2,
                        help="Size of points in point cloud")
    
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device for stereo inference")
    
    parser.add_argument("--output", "-o", default="slam_visualization.html",
                        help="Output file path (.html, .png, or .pdf)")
    parser.add_argument("--title", default="Stereo SLAM Visualization",
                        help="Title for the figure")
    
    parser.add_argument("--no_trajectory", action="store_true",
                        help="Don't show trajectory")
    parser.add_argument("--no_images", action="store_true",
                        help="Don't show stereo images")
    parser.add_argument("--no_pointclouds", action="store_true",
                        help="Don't show depth point clouds")
    parser.add_argument("--no_cameras", action="store_true",
                        help="Don't show camera frustums")
    
    args = parser.parse_args()
    
    viz = StereoSLAMVisualizer(
        trajectory_path=args.trajectory,
        sequence_config=args.sequence_config,
        device=args.device
    )
    
    if args.h5_path or args.sequence_config:
        viz.load_h5_data(args.h5_path)
    
    if not args.no_trajectory:
        viz.add_trajectory(
            start_time=args.start_time,
            end_time=args.end_time,
            color='blue',
            name="Estimated Trajectory"
        )
    
    if args.keyframes:
        if not args.no_cameras:
            viz.add_keyframe_cameras(
                timestamps=args.keyframes,
                scale=args.frustum_scale,
                color='red'
            )
        
        if not args.no_images:
            viz.add_stereo_images(
                timestamps=args.keyframes,
                distance=args.image_distance,
                separation=args.image_separation
            )
        
        if not args.no_pointclouds:
            viz.add_depth_pointclouds(
                timestamps=args.keyframes,
                max_depth=args.max_depth,
                min_depth=args.min_depth,
                subsample=args.subsample,
                point_size=args.point_size
            )
    
    viz.save(args.output, title=args.title)
    
    print("\nVisualization complete!")
    print(f"Open {args.output} in a web browser to view the interactive 3D figure.")


if __name__ == "__main__":
    main()

    """
    python3 make_figs.py \
        -t /path/to/outputs/<run>/trajectory/surfslam.txt \
        -c ../cfg/tbnms/mono_long.yaml \
        -k 1748882197.135807037 1748882297.386443138 \
        -s 1748882197.135807037 \
        -e 1748882297.386443138
    """