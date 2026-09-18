import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import open3d as o3d

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/surfslam")


from surfslam.common.dataset_paths import load_config, resolve_config_path
from surfslam.common.settings import Settings
from surfslam.common.sensors import StereoImage
from surfslam.mapping.reconstruction.build_tsdf import NvBloxTSDFIntegrator
from surfslam.mapping.stereo.stereo_estimation_defom import DefomStereoEstimator
from examples.tbnms.tbnms_calibration import TBNMSCalibration
from examples.utils.h5_log_reader import H5LogReader
from examples.utils.pose_utils import load_calibration


def parse_args():
    parser = argparse.ArgumentParser("Build map offline from SurfSLAM output")
    parser.add_argument("configuration_path")
    parser.add_argument("surfslam_output", nargs="?", default=None)
    parser.add_argument(
        "--duration", help="How long to run for (in input data time, sec)", type=float, default=None)
    parser.add_argument("--output_mesh", type=str, default=None,
                        help="Path to save output mesh (default: None)")
    parser.add_argument("--maxrange", type=float, default=None,
                        help="if set, overrides ray_range[1]")
    parser.add_argument("--max_time_diff", type=float, default=0.05,
                        help="Maximum time difference for stereo frame matching (default: 0.05s)")
    return parser.parse_args()


def find_closest_stereo_index(target_timestamp, stereo_timestamps, stereo_indices, max_time_diff=0.005):
    """Find the stereo frame index closest to the target timestamp."""
    # Vectorized computation of time differences
    time_diffs = np.abs(stereo_timestamps - target_timestamp)
    
    # Find the closest match
    min_idx = np.argmin(time_diffs)
    
    # Check if within tolerance
    if time_diffs[min_idx] > max_time_diff:
        return None
    
    return stereo_indices[min_idx]


def reject_depth_edges(depth, depth_jump_thresh):
    """
    depth: (H, W) torch tensor
    returns: cleaned depth (invalid set to 0)
    """
    dzdx = torch.abs(depth[:, 1:] - depth[:, :-1])
    dzdy = torch.abs(depth[1:, :] - depth[:-1, :])

    dzdx = torch.nn.functional.pad(dzdx, (0, 1))
    dzdy = torch.nn.functional.pad(dzdy, (0, 0, 0, 1))

    edge_mask = (dzdx > depth_jump_thresh) | (dzdy > depth_jump_thresh)
    depth_clean = depth.clone()
    depth_clean[edge_mask] = 0.0
    return depth_clean


def joint_bilateral_depth(depth, color, sigma_spatial=5, sigma_depth=0.05):
    """
    depth: (H, W) torch tensor
    color: (3, H, W) torch tensor, uint8 or float
    """
    depth_np = depth.cpu().numpy().astype(np.float32)
    color_np = color.permute(1, 2, 0).cpu().numpy()

    depth_filtered = cv2.ximgproc.jointBilateralFilter(
        guide=color_np,
        src=depth_np,
        d=-1,
        sigmaColor=sigma_depth,
        sigmaSpace=sigma_spatial,
    )

    return torch.from_numpy(depth_filtered).to(depth.device)

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.configuration_path)

    # Load settings
    baseline_settings_path = resolve_config_path(config['baseline'])
    settings = Settings.load_from_file(baseline_settings_path)

    if "changes" in config and config["changes"] is not None:
        settings.augment(config["changes"])
    
    # Override maxrange if specified
    if args.maxrange is not None:
        settings.mapper.stereo.ray_range[1] = args.maxrange
    
    im_scale_factor = settings.system.image_scale_factor
    
    # Load calibration
    calibration: TBNMSCalibration = load_calibration(
        config["dataset_family"], 
        config["calibration"]
    )
    
    # Get calibration dict for intrinsics
    calib_dict = calibration.to_dict(im_scale_factor)
    camera_intrinsics = calib_dict["camera_intrinsic"]["k"]  # torch tensor (3, 3)
    
    # Initialize stereo depth estimator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stereo_estimator = DefomStereoEstimator(
        settings.mapper.stereo_estimator,
        calib_dict,
        device
    )
    mapper = NvBloxTSDFIntegrator(settings)
    
    # Load SurfSLAM trajectory output
    if args.surfslam_output is None:
        raise ValueError("surfslam_output argument is required")
    
    try:
        surfslam_output = np.loadtxt(args.surfslam_output)
    except Exception as e:
        raise ValueError(f"Could not load SurfSLAM output from {args.surfslam_output}: {e}")

    timestamps = surfslam_output[:, 0]
    positions = surfslam_output[:, 1:4]
    orientations = surfslam_output[:, 4:8]  # quaternions (qx, qy, qz, qw)
    
    # Load H5 dataset
    h5_path = Path(os.path.expanduser(config["dataset"]))
    stereo_key = settings.system.h5_keys.stereo
    barometer_key = settings.system.h5_keys.barometer
    dvl_key = settings.system.h5_keys.dvl
    imu_key = settings.system.h5_keys.imu
    
    log_reader = H5LogReader(h5_path, imu_key, dvl_key, barometer_key, stereo_key)
    
    # Build list of stereo events for faster lookup
    stereo_events = [event for event in log_reader.events() if event.type == "STEREO"]
    
    # Pre-compute stereo timestamps and indices for vectorized lookup
    # Access timestamps directly from events (already cached during indexing)
    stereo_timestamps = np.array([event.stamp for event in stereo_events])
    stereo_indices = np.array([event.index for event in stereo_events])
    
    # Process each pose and integrate corresponding depth frame
    integrated_count = 0
    coords = []
    for i in tqdm(range(len(timestamps)), desc="Integrating frames"):
        timestamp = timestamps[i]
        
        # Check duration constraint
        if args.duration is not None and timestamp > args.duration:
            break
        
        # Find closest stereo frame
        stereo_idx = find_closest_stereo_index(
            timestamp, 
            stereo_timestamps,
            stereo_indices,
            max_time_diff=args.max_time_diff
        )
        
        if stereo_idx is None:
            continue
        
        # Load stereo images
        stereo_ts, left, right = log_reader.stereo()[stereo_idx]
        
        # Rectify images
        leftRect, rightRect = calibration.rectify_images(left, right, im_scale_factor)
        
        # Prepare stereo image for inference
        left_torch = torch.from_numpy(leftRect).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        right_torch = torch.from_numpy(rightRect).permute(2, 0, 1)
        stereo_img = StereoImage(left_torch, right_torch, stereo_ts)
        
        # Run stereo depth inference
        depth_image_obj = stereo_estimator.infer(stereo_img)
        depth_tensor = depth_image_obj.image  # (1, H, W)
        
        # Build camera pose from position and orientation
        position = positions[i]
        quat_xyzw = orientations[i]  # (qx, qy, qz, qw)
        
        # Convert quaternion to rotation matrix
        rot = R.from_quat(quat_xyzw)  # scipy uses (x,y,z,w) format
        rot_matrix = rot.as_matrix()
        
        # Build 4x4 transformation matrix
        camera_pose = torch.eye(4, dtype=torch.float32)
        camera_pose[:3, :3] = torch.from_numpy(rot_matrix).float()
        camera_pose[:3, 3] = torch.from_numpy(position).float()
        
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        coord.transform(camera_pose.cpu().numpy())
        coords.append(coord)
        
        # Prepare color image (left rectified)
        color_image = torch.from_numpy(leftRect).permute(2, 0, 1)  # (C,H,W)
        
        depth_tensor_cleaned = reject_depth_edges(depth_tensor.squeeze(), depth_jump_thresh=0.03)
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].imshow(depth_tensor_cleaned.cpu().numpy())
        # axs[1].imshow(depth_tensor.squeeze().cpu().numpy())
        # plt.savefig(f"depth_frame_{i}.png")
        # exit()
        mapper.integrate_frame(
            depth_tensor_cleaned.squeeze(),
            camera_intrinsics,
            camera_pose,
            color_image
        )
        
        integrated_count += 1
    
    print(f"\nIntegrated {integrated_count} frames into the map.")
    
    # Save mesh if output path specified
    if args.output_mesh is not None:
        output_path = Path(args.output_mesh)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving mesh to {output_path}...")
        mapper.save_mesh(str(output_path))
        print("Done!")
    else:
        print("No output mesh path specified. Skipping mesh save.")

    output_mesh = mapper.get_output_mesh()
    import trimesh
    mesh = trimesh.Trimesh(vertices=output_mesh.vertices().cpu().numpy(), faces=output_mesh.triangles().cpu().numpy())
    o3d_mesh = output_mesh.to_open3d()
    print(f"Final mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    o3d.visualization.draw_geometries([o3d_mesh, *coords])
    
if __name__ == "__main__":
    main()
    