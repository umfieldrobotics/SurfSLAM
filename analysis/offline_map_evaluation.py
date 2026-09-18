#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d  # type: ignore
import point_cloud_utils as pcu
import torch
import tqdm
from kornia.geometry.depth import depth_to_3d_v2
from scipy.spatial.transform import Rotation as R
from kaolin.metrics.pointcloud import chamfer_distance, sided_distance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "surfslam"))

from surfslam.common.dataset_paths import load_config, resolve_config_path
from examples.utils.h5_log_reader import H5LogReader
from examples.utils.pose_utils import load_calibration
from surfslam.mapping.stereo.stereo_estimation_defom import DefomStereoEstimator
from surfslam.common.settings import Settings
from surfslam.common.sensors import StereoImage
from surfslam.mapping.pointcloud_registration.pointcloud_cleaner import GPUPointCloudCleaner

ATM_PRESSURE = 986.3999633789062


class PointCloudCleanerSettings:
    def __init__(self, enable_sor=False, sor_neighbors=20, sor_std=2.0,
                 enable_ror=True, ror_radius=0.05, ror_min_pts=10,
                 target_point_count=50000, max_depth_m=10.0):
        self.enable_sor = enable_sor
        self.sor_neighbors = sor_neighbors
        self.sor_std = sor_std
        self.enable_ror = enable_ror
        self.ror_radius = ror_radius
        self.ror_min_pts = ror_min_pts
        self.target_point_count = target_point_count
        self.max_depth_m = max_depth_m


def load_tum_poses(tum_path):
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


def load_alignment_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    T = np.eye(4)
    T[:3, :3] = np.array(data['rotation']) * data.get('scale', 1.0)
    T[:3, 3] = np.array(data['translation'])

    return T, Path(data['trajectory_path'])


def find_closest_stereo_frame(target_ts, stereo_ts, stereo_events, log_reader, max_diff=0.005):
    if len(stereo_ts) == 0:
        return None, None, None

    time_diffs = np.abs(stereo_ts - target_ts)
    min_idx = np.argmin(time_diffs)

    if time_diffs[min_idx] > max_diff:
        return None, None, None

    event = stereo_events[min_idx]
    timestamp = stereo_ts[min_idx]
    _, left_img, right_img = log_reader.stereo()[event.index]
    return min_idx, timestamp, (left_img, right_img)


def find_closest_baro_reading(stereo_ts, baro_readings):
    if not baro_readings:
        return None

    baro_ts = np.array([b['timestamp'] for b in baro_readings])
    time_diffs = np.abs(baro_ts - stereo_ts)
    min_idx = np.argmin(time_diffs)

    return baro_readings[min_idx]


def find_closest_imu_reading(stereo_ts, imu_readings):
    if not imu_readings:
        return None

    imu_ts = np.array([imu['timestamp'] for imu in imu_readings])
    time_diffs = np.abs(imu_ts - stereo_ts)
    min_idx = np.argmin(time_diffs)

    return imu_readings[min_idx]


def estimate_water_surface(baro_data, imu_data, cam_pose, surface_threshold=0.5):
    if baro_data is None or imu_data is None:
        return None

    RHO_WATER = 997.0
    G = 9.81

    depth = (baro_data['pressure'] - ATM_PRESSURE) * 100 / (RHO_WATER * G)

    accel_imu = np.array(
        [imu_data['accel_x'], imu_data['accel_y'], imu_data['accel_z']])
    gravity_imu = accel_imu / np.linalg.norm(accel_imu)

    R_imu_to_cam = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    gravity_cam = R_imu_to_cam @ gravity_imu

    R_cam = cam_pose[:3, :3]
    gravity_world = R_cam @ gravity_cam

    surface_normal_world = -gravity_world / np.linalg.norm(gravity_world)

    cam_pos = cam_pose[:3, 3]
    surface_point = cam_pos + depth * gravity_world

    return {
        'point': surface_point,
        'normal': surface_normal_world,
        'depth': depth,
        'threshold': surface_threshold
    }


def estimate_water_surface_plane_in_camera(
    baro_data,
    imu_data,
    T_world_cam,
    surface_threshold=0.5,
    pressure_units="hpa",
    rho_water_kg_m3=997.0,
    g_m_s2=9.81,
    R_cam_imu=None,
):
    if baro_data is None or imu_data is None:
        return None

    pressure_meas = float(baro_data["pressure"])
    pressure_atm = float(ATM_PRESSURE)  # must match baro units

    if pressure_units.lower() == "hpa":
        pressure_delta_pa = (pressure_meas - pressure_atm) * 100.0
    elif pressure_units.lower() == "pa":
        pressure_delta_pa = (pressure_meas - pressure_atm)
    else:
        raise ValueError("pressure_units must be 'hpa' or 'pa'.")

    depth_m = pressure_delta_pa / (rho_water_kg_m3 * g_m_s2)
    if not np.isfinite(depth_m):
        return None
    depth_m = float(max(0.0, depth_m))

    accel_imu = np.array(
        [imu_data["accel_x"], imu_data["accel_y"], imu_data["accel_z"]],
        dtype=np.float64,
    )
    accel_norm = float(np.linalg.norm(accel_imu))
    if accel_norm < 1e-9 or not np.isfinite(accel_norm):
        return None

    accel_dir_imu = accel_imu / accel_norm

    gravity_dir_imu = -accel_dir_imu

    if R_cam_imu is None:
        # TODO: onur get the extrinsics from the config file
        R_cam_imu = np.array(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    else:
        R_cam_imu = np.asarray(R_cam_imu, dtype=np.float64)
        if R_cam_imu.shape != (3, 3):
            raise ValueError("R_cam_imu must be 3x3.")

    gravity_dir_cam = R_cam_imu @ gravity_dir_imu
    gravity_dir_cam /= np.linalg.norm(gravity_dir_cam)

    T_world_cam = np.asarray(T_world_cam, dtype=np.float64)
    if T_world_cam.shape != (4, 4):
        raise ValueError("T_world_cam must be 4x4.")

    R_world_cam = T_world_cam[:3, :3]
    p_world_cam = T_world_cam[:3, 3]

    gravity_dir_world = R_world_cam @ gravity_dir_cam
    gravity_dir_world /= np.linalg.norm(gravity_dir_world)

    n_world = -gravity_dir_world
    n_world /= np.linalg.norm(n_world)

    if float(np.dot(n_world, gravity_dir_world)) > 0.0:
        n_world = -n_world
    p0_world = p_world_cam + depth_m * n_world
    signed_height_check = float(
        np.dot(p0_world - p_world_cam, n_world))  # should be +depth_m

    R_cam_world = R_world_cam.T
    n_cam = R_cam_world @ n_world
    n_cam /= np.linalg.norm(n_cam)

    p0_cam = depth_m * n_cam
    d_cam = -float(np.dot(n_cam, p0_cam))

    return {
        "point": p0_world,
        "normal": n_world,
        "depth": depth_m,
        "threshold": float(surface_threshold),
        "signed_height_check": signed_height_check,
        "plane_normal_cam": n_cam,
        "plane_offset_d_cam": d_cam,
        "point_on_plane_cam": p0_cam,
    }


def filter_points_above_water(pts3d_cam, cam_pose, water_surface):
    if water_surface is None or len(pts3d_cam) == 0:
        return pts3d_cam

    pts3d_world = (cam_pose[:3, :3] @ pts3d_cam.T).T + cam_pose[:3, 3]

    surface_point = water_surface['point']
    surface_normal = water_surface['normal']
    threshold = water_surface['threshold']

    distances = np.dot(pts3d_world - surface_point, surface_normal)

    valid_mask = distances < -threshold

    return pts3d_cam[valid_mask], pts3d_cam[~valid_mask]


def create_water_surface_mesh(water_surface, size=5.0):
    if water_surface is None:
        return None

    point = water_surface['point']
    normal = water_surface['normal']

    u = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    vertices = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertex = point + size * (i * u + j * v)
            vertices.append(vertex)

    vertices = np.array(vertices)
    triangles = np.array([[0, 1, 2], [1, 3, 2]])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color([0.3, 0.6, 0.9])
    mesh.compute_vertex_normals()

    return mesh


def reject_depth_edges(depth, threshold=0.03):
    dzdx = torch.abs(depth[:, 1:] - depth[:, :-1])
    dzdy = torch.abs(depth[1:, :] - depth[:-1, :])
    dzdx = torch.nn.functional.pad(dzdx, (0, 1))
    dzdy = torch.nn.functional.pad(dzdy, (0, 0, 0, 1))
    edge_mask = (dzdx > threshold) | (dzdy > threshold)
    depth_clean = depth.clone()
    depth_clean[edge_mask] = 0.0
    return depth_clean


def prepare_torch_image(img):
    if img.ndim != 3:
        raise ValueError("Expected HxWxC image")
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def compute_point_cloud_metrics(pred_pts: torch.Tensor, gt_pts: torch.Tensor):
    chamfer_dist, _ = _chamfer(pred_pts, gt_pts)
    accuracy = _sided_distance(pred_pts, gt_pts)
    completeness = _sided_distance(gt_pts, pred_pts)
    hausdorff = _hausdorff(pred_pts, gt_pts)

    return {
        'chamfer': chamfer_dist,
        'accuracy': accuracy.mean(),
        'completeness': completeness.mean(),
        'hausdorff': hausdorff
    }


def _hausdorff(src: torch.Tensor, dst: torch.Tensor) -> float:
    """Computes the Hausdorff distance between two point clouds."""
    src_np = src.squeeze(0).cpu().numpy()
    dst_np = dst.squeeze(0).cpu().numpy()
    hausdorff_dist = pcu.hausdorff_distance(src_np, dst_np)
    return hausdorff_dist


def _chamfer(src: torch.Tensor, dst: torch.Tensor) -> Tuple[float, float]:
    pcd_cd = pcu.chamfer_distance(
        src.squeeze(0).cpu().numpy(),
        dst.squeeze(0).cpu().numpy(),
    )
    kaolin_cd = float(chamfer_distance(src, dst, squared=False).item())
    kaolin_cd_l2 = float(chamfer_distance(src, dst, squared=True).item())
    # check if the two methods give similar results
    if not np.isclose(pcd_cd, kaolin_cd, rtol=1e-5, atol=1e-5):
        logging.warning(
            "Point Cloud Utils and Kaolin give different Chamfer distances: "
            "PCD: %.6f, Kaolin: %.6f",
            pcd_cd,
            kaolin_cd,
        )
    return kaolin_cd, kaolin_cd_l2


def _sided_distance(src: torch.Tensor, dst: torch.Tensor) -> np.ndarray:
    dist, _ = sided_distance(src, dst)
    return dist.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_config")
    parser.add_argument("--alignment_json", type=Path, required=True)
    parser.add_argument("--gt_pointcloud", type=Path, required=True)
    parser.add_argument("--gt_file", type=Path, default=None)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="outputs/pointcloud_eval")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--max_time_diff", type=float, default=0.005)
    parser.add_argument("--enable_cleaning", action="store_true")
    parser.add_argument("--sor_neighbors", type=int, default=20)
    parser.add_argument("--sor_std", type=float, default=2.0)
    parser.add_argument("--ror_radius", type=float, default=0.5)
    parser.add_argument("--ror_min_pts", type=int, default=10)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_frames", action="store_true",
                        help="Visualize each frame with water surface and depth points")
    parser.add_argument("--water_surface_threshold", type=float, default=0.5,
                        help="Threshold below water surface to filter points (m)")
    args = parser.parse_args()

    pc_cleaner = None
    if args.enable_cleaning:
        settings = PointCloudCleanerSettings(
            sor_neighbors=args.sor_neighbors,
            sor_std=args.sor_std,
            ror_radius=args.ror_radius,
            ror_min_pts=args.ror_min_pts
        )
        pc_cleaner = GPUPointCloudCleaner(settings, device=args.device)
        print(f"Cleaning enabled: SOR(n={args.sor_neighbors}, std={args.sor_std}), "
              f"ROR(r={args.ror_radius}m, min={args.ror_min_pts})")

    config = load_config(args.sequence_config)

    baseline_path = resolve_config_path(config['baseline'])
    print(f"Loading settings from {baseline_path}")

    settings = Settings.load_from_file(baseline_path)
    if 'changes' in config:
        settings.augment(config['changes'])

    calibration = load_calibration(
        config["dataset_family"], config["calibration"])
    settings['calibration'] = calibration.to_dict(args.scale)

    stereo_matcher = DefomStereoEstimator(
        settings.mapper.stereo_estimator,  # type: ignore
        settings.calibration,
        args.device
    )

    h5_path = Path(os.path.expanduser(config["dataset"]))
    h5_keys = settings.system.h5_keys
    log_reader = H5LogReader(h5_path, h5_keys.imu,
                             h5_keys.dvl, h5_keys.barometer, h5_keys.stereo)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    T_align, traj_file = load_alignment_json(args.alignment_json)
    print(f"Alignment: {args.alignment_json}")
    print(f"Trajectory: {traj_file}")
    print(f"Transform:\n{T_align}")

    traj_poses = load_tum_poses(traj_file)
    print(f"Loaded {len(traj_poses)} trajectory poses")

    gt_pcd = o3d.io.read_point_cloud(str(args.gt_pointcloud))
    print(f"Loaded GT pointcloud: {len(gt_pcd.points)} points")

    gt_poses = None
    if args.gt_file is not None:
        gt_poses = load_tum_poses(args.gt_file)
        print(f"Loaded {len(gt_poses)} GT poses")

    stereo_events = [e for e in log_reader.events() if e.type == "STEREO"]
    if not stereo_events:
        raise RuntimeError("No stereo events in HDF5")
    stereo_times = np.array([e.stamp for e in stereo_events])
    print(f"Found {len(stereo_times)} stereo frames")

    baro_events = [e for e in log_reader.events() if e.type == "PRESSURE"]
    baro_readings = []
    if baro_events:
        for event in baro_events:
            baro_data = log_reader.pressure()[event.index]  # type: ignore
            baro_readings.append(
                {'timestamp': baro_data.timestamp, 'pressure': baro_data.pressure})
        print(f"Logged {len(baro_readings)} barometer readings")

    imu_events = [e for e in log_reader.events() if e.type == "IMU"]
    imu_readings = []
    if imu_events:
        for event in tqdm.tqdm(imu_events):
            imu_data = log_reader.imu()[event.index]  # type: ignore
            imu_readings.append({
                'timestamp': imu_data.timestamp,
                'accel_x': imu_data.ax,
                'accel_y': imu_data.ay,
                'accel_z': imu_data.az
            })
        print(f"Logged {len(imu_readings)} IMU readings")

    K = np.array(calibration.to_dict(args.scale)['camera_intrinsic']['k'])

    print("\n=== Processing Frames ===")
    accumulated_pcd = o3d.geometry.PointCloud()
    max_frames = args.max_frames if args.max_frames > 0 else len(traj_poses)
    processed = 0
    coords = []
    gt_coords = []
    stereo_baro_pairs = []

    for _, (traj_ts, traj_pose) in tqdm.tqdm(enumerate(traj_poses), total=len(traj_poses)):
        if processed >= max_frames:
            break

        _, stereo_ts, images = find_closest_stereo_frame(
            traj_ts, stereo_times, stereo_events, log_reader, args.max_time_diff
        )
        if images is None:
            continue

        left_img, right_img = images
        left_rect, right_rect = calibration.rectify_images(
            left_img, right_img, args.scale)

        closest_baro = find_closest_baro_reading(stereo_ts, baro_readings)
        closest_imu = find_closest_imu_reading(stereo_ts, imu_readings)

        if closest_baro is not None:
            stereo_baro_pairs.append({
                'stereo_timestamp': float(stereo_ts),
                'baro_timestamp': float(closest_baro['timestamp']),
                'pressure': float(closest_baro['pressure']),
                'time_diff': float(abs(stereo_ts - closest_baro['timestamp']))
            })

        # water_surface = estimate_water_surface(
        #     closest_baro, closest_imu, traj_pose, args.water_surface_threshold
        # )
        water_surface = estimate_water_surface_plane_in_camera(
            closest_baro, closest_imu, traj_pose,
            pressure_units="hpa",
            surface_threshold=args.water_surface_threshold,
        )

        stereo_img = StereoImage(
            prepare_torch_image(left_rect),
            prepare_torch_image(right_rect),
            stereo_ts
        )
        depth_result = stereo_matcher.infer(stereo_img)
        depth = depth_result.image.squeeze(
            0).detach().cpu().numpy().astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        depth_tensor = torch.from_numpy(depth)
        depth_tensor[depth_tensor >= args.max_depth] = 0.0
        depth_clean = reject_depth_edges(depth_tensor.squeeze())

        pts3d = depth_to_3d_v2(
            depth_clean.unsqueeze(0).unsqueeze(0),
            torch.from_numpy(K).unsqueeze(
                0) if K.ndim == 2 else torch.from_numpy(K)
        ).reshape(-1, 3)

        valid_mask = depth_clean.flatten() > 0
        pts3d_valid = pts3d[valid_mask]

        if isinstance(pts3d_valid, torch.Tensor):
            pts3d_valid = pts3d_valid.cpu().numpy()
        if pts3d_valid.shape[0] == 0:
            continue
        pts3d_valid, pts3d_above_water = filter_points_above_water(
            pts3d_valid, traj_pose, water_surface)

        if pc_cleaner is not None:
            if len(pts3d_valid) == 0:
                continue
            pts3d_clean, _ = pc_cleaner.clean(
                torch.from_numpy(pts3d_valid).to(args.device))
            if isinstance(pts3d_clean, torch.Tensor):
                pts3d_clean = pts3d_clean.cpu().numpy()
        else:
            pts3d_clean = pts3d_valid

        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(pts3d_clean)

        if args.visualize_frames and pts3d_above_water.shape[0] > 0:
            vis_geometries = []

            frame_pcd_world = o3d.geometry.PointCloud()
            frame_pcd_world.points = o3d.utility.Vector3dVector(pts3d_clean)
            frame_pcd_world.transform(traj_pose)
            frame_pcd_world.paint_uniform_color([0.0, 1.0, 0.0])
            # estimate normals
            frame_pcd_world.estimate_normals()
            vis_geometries.append(frame_pcd_world)

            frame_pcd_above_water = o3d.geometry.PointCloud()
            frame_pcd_above_water.points = o3d.utility.Vector3dVector(
                pts3d_above_water)
            frame_pcd_above_water.transform(traj_pose)
            frame_pcd_above_water.paint_uniform_color([1.0, 0.0, 0.0])
            frame_pcd_above_water.estimate_normals()
            vis_geometries.append(frame_pcd_above_water)

            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0)
            cam_frame.transform(traj_pose)
            vis_geometries.append(cam_frame)

            water_mesh = create_water_surface_mesh(water_surface, size=10.0)
            if water_mesh is not None:
                vis_geometries.append(water_mesh)

            o3d.visualization.draw_geometries(
                vis_geometries,
                window_name=f"Frame {processed}: Pose + Water Surface + Depth Points",
                width=1280, height=720
            )

        frame_pcd.transform(traj_pose)
        # o3d.io.write_point_cloud(
        #     str(output_dir / f"frame_{processed:06d}_pcd.ply"), frame_pcd)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord.transform(traj_pose).transform(T_align)
        coords.append(coord)

        if gt_poses is not None:
            closest_gt = min(gt_poses, key=lambda x: abs(x[0] - traj_ts))
            if abs(closest_gt[0] - traj_ts) < 0.1:
                gt_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5)
                gt_coord.transform(
                    closest_gt[1]).paint_uniform_color([0, 1, 0])
                gt_coords.append(gt_coord)

        accumulated_pcd += frame_pcd
        processed += 1

    print(
        f"\nAccumulated {processed} frames: {len(accumulated_pcd.points)} points")
    print(f"Downsampling with voxel size {args.voxel_size}m...")

    pcd_downsampled = accumulated_pcd.voxel_down_sample(
        voxel_size=args.voxel_size)
    print(f"After downsampling: {len(pcd_downsampled.points)} points")

    o3d.io.write_point_cloud(
        str(output_dir / "trajectory_pointcloud.ply"), pcd_downsampled)

    pcd_aligned = pcd_downsampled.transform(T_align)
    o3d.io.write_point_cloud(
        str(output_dir / "trajectory_pointcloud_aligned.ply"), accumulated_pcd.transform(T_align))

    if baro_readings:
        with open(output_dir / "barometer_readings.json", 'w') as f:
            json.dump(baro_readings, f, indent=2)

    if stereo_baro_pairs:
        with open(output_dir / "stereo_baro_pairs.json", 'w') as f:
            json.dump(stereo_baro_pairs, f, indent=2)
        print(f"Saved {len(stereo_baro_pairs)} stereo-barometer pairs")

    if args.visualize:
        o3d.visualization.draw_geometries(  # type: ignore
            [pcd_downsampled, gt_pcd, *coords, *gt_coords],
            window_name="Point Cloud Evaluation"
        )

    print("\n=== Computing Metrics ===")
    pred_pts = torch.from_numpy(np.asarray(
        pcd_aligned.points, dtype=np.float32)).unsqueeze(0).cuda()
    gt_pts = torch.from_numpy(np.asarray(
        gt_pcd.points, dtype=np.float32)).unsqueeze(0).cuda()

    metrics = compute_point_cloud_metrics(
        pred_pts=pred_pts, gt_pts=gt_pts)

    print(f"\nMetrics:")
    print(f"  Chamfer:     {metrics['chamfer']:.6f} m")
    print(f"  Accuracy:    {metrics['accuracy']:.6f} m")
    print(f"  Completeness:{metrics['completeness']:.6f} m")
    print(f"  Hausdorff:   {metrics['hausdorff']:.6f} m")

    with open(output_dir / "evaluation_results.txt", "w") as f:
        f.write(f"Point Cloud Evaluation Results\n")
        f.write(f"==============================\n\n")
        f.write(f"Alignment JSON:    {args.alignment_json}\n")
        f.write(f"Trajectory:        {traj_file}\n")
        f.write(f"GT pointcloud:     {args.gt_pointcloud}\n")
        f.write(f"Frames processed:  {processed}\n")
        f.write(f"Voxel size:        {args.voxel_size} m\n")
        f.write(f"Max depth:         {args.max_depth} m\n")
        f.write(f"Baro readings:     {len(baro_readings)}\n\n")
        f.write(f"Pred points:       {len(pcd_aligned.points)}\n")
        f.write(f"GT points:         {len(gt_pcd.points)}\n\n")
        f.write(f"Metrics:\n")
        f.write(f"--------\n")
        f.write(f"Chamfer:           {metrics['chamfer']:.6f} m\n")
        f.write(f"Accuracy:          {metrics['accuracy']:.6f} m\n")
        f.write(f"Completeness:      {metrics['completeness']:.6f} m\n")
        f.write(f"Hausdorff:         {metrics['hausdorff']:.6f} m\n")

    print(f"\nResults saved to {output_dir / 'evaluation_results.txt'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
