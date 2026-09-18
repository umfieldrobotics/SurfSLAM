#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import open3d as o3d  # type: ignore
import torch
import tqdm
import warnings
import glob
from kornia.geometry.depth import depth_to_3d_v2
import matplotlib.pyplot as plt

import yaml
from scipy.spatial.transform import Rotation

from io_utils import (SCENE_DIRS, load_alignment_json, load_tum_poses, find_closest_stereo_frame,
                      find_closest_baro_reading, find_closest_imu_reading, prepare_torch_image,
                      get_pointcloud)
from pc_utils import compute_point_cloud_metrics
from cleanup_utils import (create_water_surface_mesh, filter_points_above_water,
                           estimate_water_surface_plane_in_camera, reject_depth_edges)
from valid_space_utils import filter_pointcloud_to_valid_space

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "surfslam"))

from examples.utils.h5_log_reader import H5LogReader
from examples.utils.pose_utils import load_calibration
from surfslam.mapping.stereo.stereo_estimation_defom import DefomStereoEstimator
from surfslam.common.dataset_paths import get_dataset_root, load_config, resolve_config_path
from surfslam.common.settings import Settings
from surfslam.common.sensors import StereoImage
from surfslam.mapping.pointcloud_registration.pointcloud_cleaner import GPUPointCloudCleaner

warnings.filterwarnings(
    "ignore",
    message=r".*autocast.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)

_GT_SCENES = SCENE_DIRS

#: Transform from the released trajectory frame into the photogrammetry model frame.
#: See model_frame.yaml for why the two disagree and where these numbers come from.
_MODEL_FRAME_FILE = Path(__file__).resolve().parent / "model_frame.yaml"


def gt_pointcloud_path(experiment_name: str) -> str:
    scene = _GT_SCENES[experiment_name]
    return str(get_dataset_root("suds_slam") / scene / "ground_truth" / "reconstruction.ply")


def hull_file_path(experiment_name: str, hull_dir) -> str:
    """Convex hull bounding the space the ground-truth model actually covers.

    Not part of the DeepBlue release, so the directory holding the <scene>_clean_hull.npz
    files has to be passed in with --hull_dir when --use_hull_filtering is on.
    """
    scene = _GT_SCENES[experiment_name]
    return str(Path(hull_dir) / f"{scene}_clean_hull.npz")


def load_model_frame_transforms() -> dict:
    """Per-sequence 4x4 transform: released trajectory frame -> photogrammetry model frame."""
    with open(_MODEL_FRAME_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    transforms = {}
    for name, entry in cfg["scenes"].items():
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(entry["quaternion_xyzw"]).as_matrix()
        T[:3, 3] = entry["translation"]
        transforms[name] = T

    # Result trees predating 8970b4b still say monohansett_start.
    for alias, canonical in [("monohansett_start", "monohansett_engine")]:
        if canonical in transforms:
            transforms.setdefault(alias, transforms[canonical])
    return transforms


def model_frame_alignment(alignment_json, transforms: dict):
    """Load a trajectory alignment and lift it into the photogrammetry model frame.

    evaluate_trajectories.py aligns onto the release's trajectory.tum, so the transform it
    saves lands the estimate in the trajectory frame. The ground-truth point cloud lives in
    the model frame, one fixed transform away.
    """
    T_align, traj_file = load_alignment_json(alignment_json)
    sequence = json.load(open(alignment_json))["sequence"]
    if sequence not in transforms:
        raise KeyError(
            f"No model-frame transform for sequence {sequence!r} in {_MODEL_FRAME_FILE}")
    return transforms[sequence] @ T_align, traj_file


class PointCloudCleanerSettings:
    def __init__(self, enable_sor=True, sor_neighbors=10, sor_std=1.0,
                 enable_ror=True, ror_radius=0.1, ror_min_pts=4,
                 target_point_count=75000, max_depth_m=10.0):
        self.enable_sor = enable_sor
        self.sor_neighbors = sor_neighbors
        self.sor_std = sor_std
        self.enable_ror = enable_ror
        self.ror_radius = ror_radius
        self.ror_min_pts = ror_min_pts
        self.target_point_count = target_point_count
        self.max_depth_m = max_depth_m


def evaluate_others(args, ours:bool = False):
    test_number = args.alignment_json.name.split("_")[1]
    T_align, traj_file = model_frame_alignment(
        args.alignment_json, args.model_frame_transforms)
    pred_pcd = get_pointcloud(args.method, traj_file, ours=ours, scene=args.scene_dir,
                              turtlmap_maps_root=args.turtlmap_maps,
                              voxel_size=args.voxel_size)
    if pred_pcd is None:
        return None
    pred_pcd.transform(T_align)
    pred_pcd_downsampled = o3d.geometry.PointCloud.voxel_down_sample(
        pred_pcd, voxel_size=args.voxel_size)

    gt_pcd = o3d.io.read_point_cloud(str(args.gt_pointcloud))
    gt_pcd_downsampled = o3d.geometry.PointCloud.voxel_down_sample(
        gt_pcd, voxel_size=args.voxel_size)

    # Apply valid space filtering if enabled
    if args.use_hull_filtering and hasattr(args, 'hull_file') and args.hull_file is not None:
        print("\nApplying valid space filtering...")
        pred_pts_np = np.asarray(pred_pcd_downsampled.points, dtype=np.float32)
        pred_colors_np = np.asarray(pred_pcd_downsampled.colors) if pred_pcd_downsampled.has_colors() else None
        
        gt_pts_np = np.asarray(gt_pcd_downsampled.points, dtype=np.float32)
        gt_colors_np = np.asarray(gt_pcd_downsampled.colors) if gt_pcd_downsampled.has_colors() else None
        
        # Filter predicted points
        pred_pts_filtered, pred_colors_filtered = filter_pointcloud_to_valid_space(
            pred_pts_np, args.hull_file, pred_colors_np
        )
        
        # Filter GT points
        gt_pts_filtered, gt_colors_filtered = filter_pointcloud_to_valid_space(
            gt_pts_np, args.hull_file, gt_colors_np
        )
        
        pred_pts = torch.from_numpy(pred_pts_filtered).unsqueeze(0).cuda()
        gt_pts = torch.from_numpy(gt_pts_filtered).unsqueeze(0).cuda()
    else:
        pred_pts = torch.from_numpy(np.asarray(
            pred_pcd_downsampled.points, dtype=np.float32)).unsqueeze(0).cuda()
        gt_pts = torch.from_numpy(np.asarray(
            gt_pcd_downsampled.points, dtype=np.float32)).unsqueeze(0).cuda()
    
    metrics, acc, comp = compute_point_cloud_metrics(
        pred_pts=pred_pts, gt_pts=gt_pts)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # TURTLMap scores two maps per trial (its depths and ours) into the same directory.
    tag = f"{test_number}_our_depths" if ours else test_number

    acc_cmap = plt.get_cmap("jet")(acc.squeeze())[:, :3]
    comp_cmap = plt.get_cmap("jet")(comp.squeeze())[:, :3]
    acc_pcd = o3d.geometry.PointCloud()
    acc_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pred_pcd_downsampled.points))
    acc_pcd.colors = o3d.utility.Vector3dVector(acc_cmap)
    comp_pcd = o3d.geometry.PointCloud()
    comp_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_pcd_downsampled.points))
    comp_pcd.colors = o3d.utility.Vector3dVector(comp_cmap)
    o3d.io.write_point_cloud(
        str(output_dir / f"accuracy_colored_{tag}.ply"), acc_pcd)
    o3d.io.write_point_cloud(
        str(output_dir / f"completeness_colored_{tag}.ply"), comp_pcd)
    return metrics


def evaluate_surfslam(args):

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
    test_number = args.alignment_json.name.split("_")[1]
    T_align, traj_file = model_frame_alignment(
        args.alignment_json, args.model_frame_transforms)

    traj_poses = load_tum_poses(traj_file)

    gt_pcd = o3d.io.read_point_cloud(str(args.gt_pointcloud))
    gt_pcd_downsampled = o3d.geometry.PointCloud.voxel_down_sample(
        gt_pcd, voxel_size=args.voxel_size)

    stereo_events = [e for e in log_reader.events() if e.type == "STEREO"]
    if not stereo_events:
        raise RuntimeError("No stereo events in HDF5")
    stereo_times = np.array([e.stamp for e in stereo_events])

    baro_events = [e for e in log_reader.events() if e.type == "PRESSURE"]
    baro_readings = []
    if baro_events:
        for event in baro_events:
            baro_data = log_reader.pressure()[event.index]  # type: ignore
            baro_readings.append(
                {'timestamp': baro_data.timestamp, 'pressure': baro_data.pressure})

    imu_events = [e for e in log_reader.events() if e.type == "IMU"]
    imu_times = np.array([e.stamp for e in imu_events])

    K = np.array(calibration.to_dict(args.scale)['camera_intrinsic']['k'])

    accumulated_points = []
    accumulated_colors = []
    max_frames = args.max_frames if args.max_frames > 0 else len(traj_poses)
    processed = 0
    coords = []
    gt_coords = []
    stereo_baro_pairs = []
    stereo_data = log_reader.stereo()
    traj_times = np.array([ts for ts, _ in traj_poses])
    traj_monotonic = np.all(np.diff(traj_times) >= 0)
    stereo_idx = 0

    for _, traj_tuple in tqdm.tqdm(enumerate(traj_poses), total=len(traj_poses)):
        traj_ts, traj_pose = traj_tuple
        if processed >= max_frames:
            break

        if not traj_monotonic:
            _, stereo_ts, images = find_closest_stereo_frame(
                traj_ts, stereo_times, stereo_events, log_reader, args.max_time_diff
            )
            if images is None:
                continue
        else:
            while stereo_idx + 1 < len(stereo_times) and stereo_times[stereo_idx + 1] <= traj_ts:
                stereo_idx += 1

            candidate_idxs = [stereo_idx]
            if stereo_idx + 1 < len(stereo_times):
                candidate_idxs.append(stereo_idx + 1)

            candidate_diffs = [abs(stereo_times[i] - traj_ts)
                               for i in candidate_idxs]
            min_offset = int(np.argmin(candidate_diffs))
            min_idx = candidate_idxs[min_offset]

            if candidate_diffs[min_offset] > args.max_time_diff:
                continue

            event = stereo_events[min_idx]
            stereo_ts = stereo_times[min_idx]
            _, left_img, right_img = stereo_data[event.index]
            images = (left_img, right_img)

        left_img, right_img = images
        left_rect, right_rect = calibration.rectify_images(
            left_img, right_img, args.scale)

        closest_baro = find_closest_baro_reading(stereo_ts, baro_readings)
        closest_imu = find_closest_imu_reading(
            stereo_ts, imu_events, imu_times, log_reader)

        if closest_baro is not None:
            stereo_baro_pairs.append({
                'stereo_timestamp': float(stereo_ts),
                'baro_timestamp': float(closest_baro['timestamp']),
                'pressure': float(closest_baro['pressure']),
                'time_diff': float(abs(stereo_ts - closest_baro['timestamp']))
            })

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

        if pc_cleaner is not None:
            if len(pts3d) == 0:
                continue
            pts3d_clean, color_clean = pc_cleaner.clean(
                pts3d.cuda(), torch.from_numpy(left_rect).reshape(-1, 3).cuda())
            if isinstance(pts3d_clean, torch.Tensor):
                pts3d_clean = pts3d_clean.cpu().numpy()
        else:
            pts3d_clean = pts3d

        valid_mask = pts3d_clean[:, 2].flatten() > 0
        pts3d_valid = pts3d_clean[valid_mask]
        color_valid = color_clean[valid_mask]

        if isinstance(pts3d_valid, torch.Tensor):
            pts3d_valid = pts3d_valid.cpu().numpy()
        if pts3d_valid.shape[0] == 0:
            continue
        below_water_mask = filter_points_above_water(
            pts3d_valid, traj_pose, water_surface)

        pts3d_valid = pts3d_valid[below_water_mask]
        color_valid = color_valid[below_water_mask]

        pts3d_valid = np.ascontiguousarray(pts3d_valid, dtype=np.float64)

        pts3d_valid_world = (traj_pose[:3, :3] @
                             pts3d_valid.T).T + traj_pose[:3, 3]
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord.transform(traj_pose).transform(T_align)
        coords.append(coord)
        accumulated_points.append(pts3d_valid_world)
        accumulated_colors.append(color_valid.cpu().numpy())
        processed += 1

    accumulated_pcd = o3d.geometry.PointCloud()
    if accumulated_points:
        all_points = np.vstack(accumulated_points)
        all_colors = np.vstack(accumulated_colors) / 255.0
        # bgr->rgb
        all_colors = all_colors[:, [2, 1, 0]]
        accumulated_pcd.points = o3d.utility.Vector3dVector(all_points)
        accumulated_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    pcd_downsampled = accumulated_pcd.voxel_down_sample(
        voxel_size=args.voxel_size)

    pcd_aligned = pcd_downsampled.transform(T_align)
    o3d.io.write_point_cloud(
        str(args.output_dir / "trajectory_pointcloud_aligned.ply"), accumulated_pcd.transform(T_align))
    print(f'Wrote to {args.output_dir / "trajectory_pointcloud_aligned.ply"}')

    if args.visualize:
        o3d.visualization.draw_geometries(  # type: ignore
            [pcd_downsampled, gt_pcd, *coords, *gt_coords],
            window_name="Point Cloud Evaluation"
        )

    # Apply valid space filtering if enabled
    if args.use_hull_filtering and hasattr(args, 'hull_file') and args.hull_file is not None:
        print("\nApplying valid space filtering...")
        pcd_pts_np = np.asarray(pcd_aligned.points, dtype=np.float32)
        pcd_colors_np = np.asarray(pcd_aligned.colors) if pcd_aligned.has_colors() else None
        
        gt_pts_np = np.asarray(gt_pcd_downsampled.points, dtype=np.float32)
        gt_colors_np = np.asarray(gt_pcd_downsampled.colors) if gt_pcd_downsampled.has_colors() else None
        
        # Filter predicted points
        pcd_pts_filtered, pcd_colors_filtered = filter_pointcloud_to_valid_space(
            pcd_pts_np, args.hull_file, pcd_colors_np
        )
        
        # Filter GT points
        gt_pts_filtered, gt_colors_filtered = filter_pointcloud_to_valid_space(
            gt_pts_np, args.hull_file, gt_colors_np
        )
        
        pred_pts = torch.from_numpy(pcd_pts_filtered).unsqueeze(0).cuda()
        gt_pts = torch.from_numpy(gt_pts_filtered).unsqueeze(0).cuda()
    else:
        pred_pts = torch.from_numpy(np.asarray(
            pcd_aligned.points, dtype=np.float32)).unsqueeze(0).cuda()
        gt_pts = torch.from_numpy(np.asarray(
            gt_pcd_downsampled.points, dtype=np.float32)).unsqueeze(0).cuda()

    o3d.io.write_point_cloud(
        str(args.output_dir / f"gt_pointcloud_downsampled_{test_number}.ply"), gt_pcd_downsampled)
    metrics, acc, comp = compute_point_cloud_metrics(
        pred_pts=pred_pts, gt_pts=gt_pts)

    acc_cmap = plt.get_cmap("jet")(acc.squeeze())[:, :3]
    comp_cmap = plt.get_cmap("jet")(comp.squeeze())[:, :3]
    acc_pcd = o3d.geometry.PointCloud()
    acc_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_aligned.points))
    acc_pcd.colors = o3d.utility.Vector3dVector(acc_cmap)
    comp_pcd = o3d.geometry.PointCloud()
    comp_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_pcd_downsampled.points))
    comp_pcd.colors = o3d.utility.Vector3dVector(comp_cmap)
    o3d.io.write_point_cloud(
        str(args.output_dir / f"accuracy_colored_{test_number}.ply"), acc_pcd)
    o3d.io.write_point_cloud(
        str(args.output_dir / f"completeness_colored_{test_number}.ply"), comp_pcd)
    return metrics


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_results_dir", type=Path)
    parser.add_argument("--cfg_dir", type=Path, default=PROJECT_ROOT / "cfg")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir_base", type=Path,
                        default=PROJECT_ROOT / "eval_results" / "map_comparison" / "maps")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=10.0)
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
    parser.add_argument("--use_hull_filtering", action="store_true",
                        help="Restrict both clouds to the space the ground-truth model covers")
    parser.add_argument("--hull_dir", type=Path, default=None,
                        help="Directory holding <scene>_clean_hull.npz. Required with "
                             "--use_hull_filtering; the hulls are not in the release.")
    parser.add_argument("--turtlmap_maps", type=Path, default=None,
                        help="Root of the TURTLMap map tree (baseline_depths/ and our_depths/). "
                             "Defaults to ${slam_results}/turtlmap_maps.")
    parser.add_argument("--methods", nargs="+", default=None, metavar="NAME",
                        help="Only score these methods (directory names under alignments/). "
                             "Scoring one map takes minutes, so use this to re-run just the "
                             "methods whose runs changed rather than the whole comparison.")
    parser.add_argument("--reaccumulate_ours", action="store_true",
                        help="Re-run stereo and re-accumulate our map from the trajectory "
                             "instead of scoring the reconstructed_map.ply the run saved.")
    args = parser.parse_args()
    if args.use_hull_filtering and args.hull_dir is None:
        parser.error("--use_hull_filtering needs --hull_dir")
    return args


def record(metrics, label, scene_name, exp_number, output_dir, all_results):
    """Append one scored map to the per-method csv and the overall list."""
    metrics = dict(metrics)
    metrics["method"] = label
    metrics["scene"] = scene_name
    metrics["exp_number"] = exp_number

    csv_path = output_dir / "evaluation_metrics.csv"
    pd.DataFrame([metrics]).to_csv(
        csv_path, mode="a" if csv_path.exists() else "w",
        header=not csv_path.exists(), index=False)
    all_results.append(metrics)


def main():
    args = parse_args()

    os.makedirs(args.output_dir_base, exist_ok=True)
    args.model_frame_transforms = load_model_frame_transforms()

    if "alignments" not in str(args.eval_results_dir):
        eval_dir = Path(args.eval_results_dir / "alignments")
    else:
        eval_dir = args.eval_results_dir

    # folders under eval dir are the method names
    method_dirs = sorted(d for d in eval_dir.iterdir() if d.is_dir())
    if args.methods:
        wanted = set(args.methods)
        unknown = wanted - {d.name for d in method_dirs}
        if unknown:
            raise SystemExit(
                f"No alignments for {sorted(unknown)} in {eval_dir}. "
                f"Available: {sorted(d.name for d in method_dirs)}")
        method_dirs = [d for d in method_dirs if d.name in wanted]

    all_results = []
    skipped_no_map = 0
    for method in method_dirs:
        print(f"\n=== Evaluating Method: {method.name} ===")
        args.method = method.name
        if "orbslam3" in args.method.lower():
            print("ORB-SLAM3 is sparse-only, no map to evaluate. Skipping.")
            continue

        scene_dirs = [s for s in method.iterdir() if s.is_dir()]
        if len(scene_dirs) == 0:
            raise RuntimeError(f"No scene directories found in {method}")

        for scene in scene_dirs:
            print(f"\n--- Scene: {scene.name} ---")
            if scene.name not in _GT_SCENES:
                print(f"No GT pointcloud for scene {scene.name}, skipping...")
                continue

            args.scene_dir = _GT_SCENES[scene.name]
            args.gt_pointcloud = gt_pointcloud_path(scene.name)
            args.hull_file = (hull_file_path(scene.name, args.hull_dir)
                              if args.use_hull_filtering else None)
            args.output_dir = args.output_dir_base / scene.name / method.name
            os.makedirs(args.output_dir, exist_ok=True)

            # record() appends, so clear any previous pass over this (method, scene)
            # instead of stacking a second set of rows on top of it.
            stale_csv = args.output_dir / "evaluation_metrics.csv"
            if stale_csv.exists():
                stale_csv.unlink()

            traj_jsons = sorted(os.listdir(scene))
            for traj_json in tqdm.tqdm(traj_jsons, total=len(traj_jsons)):
                args.alignment_json = scene / traj_json
                exp_number = traj_json.split(".json")[0]

                # TURTLMap contributes two columns off the same trajectory: its own depths
                # and ours, fused into separate maps under turtlmap_maps/.
                if "turtlmap" in args.method.lower():
                    for ours, label in [(False, method.name),
                                        (True, f"{method.name}_our_depths")]:
                        metrics = evaluate_others(args, ours=ours)
                        if metrics is None:
                            # The mapper is deterministic, so only trial_0 was mapped.
                            skipped_no_map += 1
                            continue
                        record(metrics, label, scene.name, exp_number,
                               args.output_dir, all_results)
                    continue

                if args.reaccumulate_ours and (
                        "surfslam" in args.method.lower() or "ours" in args.method.lower()):
                    args.sequence_config = str(args.cfg_dir / "tbnms" / f"{scene.name}.yaml")
                    if not os.path.exists(args.sequence_config):
                        raise Exception(
                            f"Sequence config {args.sequence_config} does not exist")
                    metrics = evaluate_surfslam(args)
                else:
                    metrics = evaluate_others(args)

                if metrics:
                    record(metrics, method.name, scene.name, exp_number,
                           args.output_dir, all_results)
                else:
                    print(f"[FAILED] {args.alignment_json}")

    if skipped_no_map:
        print(f"\nSkipped {skipped_no_map} trial(s) with no saved map "
              f"(expected: TURTLMap is deterministic, only trial_0 was mapped).")
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        overall_csv_path = args.output_dir_base / "overall_evaluation_metrics.csv"

        # A --methods run only re-scores part of the comparison; keep the rows it didn't
        # touch so the roll-up stays complete. Match on method *and* variant label, since
        # turtlmap contributes turtlmap and turtlmap_our_depths.
        if overall_csv_path.exists():
            previous = pd.read_csv(overall_csv_path)
            rescored = set(all_results_df["method"])
            kept = previous[~previous["method"].isin(rescored)]
            if len(kept):
                print(f"Keeping {len(kept)} row(s) for methods not re-scored: "
                      f"{sorted(set(kept['method']))}")
            all_results_df = pd.concat([kept, all_results_df], ignore_index=True)

        all_results_df.to_csv(overall_csv_path, index=False)
        print(f"\nOverall evaluation metrics saved to {overall_csv_path}")


if __name__ == "__main__":
    main()
