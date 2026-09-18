import glob
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import tqdm
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[2]))
from surfslam.common.dataset_paths import get_dataset_root

#: Sequence name -> scene directory in the release and in the map result trees.
#: monohansett_start is the pre-8970b4b name for monohansett_engine; result trees written
#: before that rename still use it, so both spellings resolve.
SCENE_DIRS = {
    "monohansett_long": "long",
    "monohansett_boiler": "boiler",
    "monohansett_engine": "engine",
    "monohansett_start": "engine",
}


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

    insert_idx = np.searchsorted(stereo_ts, target_ts)
    if insert_idx == 0:
        candidate_idxs = [0]
    elif insert_idx >= len(stereo_ts):
        candidate_idxs = [len(stereo_ts) - 1]
    else:
        candidate_idxs = [insert_idx - 1, insert_idx]

    candidate_diffs = [abs(stereo_ts[i] - target_ts) for i in candidate_idxs]
    min_offset = int(np.argmin(candidate_diffs))
    min_idx = candidate_idxs[min_offset]

    if candidate_diffs[min_offset] > max_diff:
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


def find_closest_imu_reading(stereo_ts, imu_events, imu_times, log_reader):
    if len(imu_events) == 0:
        return None

    time_diffs = np.abs(imu_times - stereo_ts)
    min_idx = np.argmin(time_diffs)

    imu_data = log_reader.imu()[imu_events[min_idx].index]  # type: ignore
    return {
        'timestamp': imu_data.timestamp,
        'accel_x': imu_data.ax,
        'accel_y': imu_data.ay,
        'accel_z': imu_data.az
    }


def prepare_torch_image(img):
    if img.ndim != 3:
        raise ValueError("Expected HxWxC image")
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def default_turtlmap_maps_root() -> Path:
    return get_dataset_root("slam_results") / "turtlmap_maps"


def svin2_dense_dir(traj_file: Path) -> Path:
    """Directory of SVIn2's dense-mapping pass for the trial that wrote ``traj_file``.

    Dense mapping ran separately from the trajectory evaluation but keyed its output by the
    same <scene>/<trial> directory names, so the trajectory path locates it.
    """
    parts = traj_file.parts                       # .../<scene>/<trial>/<name>.txt
    trial, scene_dir = parts[-2], parts[-3]
    return get_dataset_root("baseline_results") / "svin2_dense_mapping_v2" / "results" / scene_dir / trial


def read_svin2_map(traj_file: Path, voxel_size=None) -> o3d.geometry.PointCloud:
    """Assemble SVIn2's dense map from its per-keyframe clouds.

    Prefers ``point_clouds/``, the per-keyframe binary PLYs the dense-mapping pass wrote,
    over the single fused ASCII PLY beside them. The fused files for the `long` sequence are
    damaged: each runs to its full declared length but the tail past ~6.2 GB is a hole of
    NUL bytes, so Open3D's reader stops partway (at vertex 149,950,366 of a declared
    268,816,290 for trial_1) and silently returns a little over half the map. The
    per-keyframe clouds are intact -- they sum to slightly *more* than the fused header
    claims, by exactly one keyframe, which is the fuser's own off-by-one -- and they are
    already in world frame, so concatenating them reproduces the fused map.

    Accumulating 270M points outright would cost ~13 GB, so clouds are downsampled in
    batches. The grid is half the caller's voxel size, leaving the authoritative
    downsampling in evaluate_others unchanged.
    """
    dense_dir = svin2_dense_dir(traj_file)
    cloud_dir = dense_dir / "point_clouds"
    keyframe_plys = sorted(cloud_dir.glob("*.ply")) if cloud_dir.is_dir() else []

    if not keyframe_plys:
        parts = traj_file.parts
        fused = dense_dir / f"{parts[-3]}_{parts[-2]}.ply"
        print(f"warning: no per-keyframe clouds in {cloud_dir}, falling back to {fused.name}. "
              f"Check it is not one of the holed fused maps.")
        return o3d.io.read_point_cloud(str(fused))

    grid = (voxel_size / 2.0) if voxel_size else None
    accumulated = o3d.geometry.PointCloud()
    batch = o3d.geometry.PointCloud()

    for i, ply in enumerate(tqdm.tqdm(keyframe_plys, desc="svin2 keyframes", leave=False), 1):
        batch += o3d.io.read_point_cloud(str(ply))
        if i % 200 == 0 or i == len(keyframe_plys):
            accumulated += batch
            batch = o3d.geometry.PointCloud()
            if grid:
                accumulated = accumulated.voxel_down_sample(voxel_size=grid)

    return accumulated


def turtlmap_map_path(traj_file: Path, scene: str, ours: bool = False, maps_root=None):
    """Locate the saved TURTLMap reconstruction for a trial, or None if there isn't one.

    The re-run parks maps outside the trial trees, in a flat per-scene layout:

        <maps_root>/baseline_depths/<scene>/<run>_trial<N>.ply   (baseline DEFOM ViT-S)
        <maps_root>/our_depths/<scene>/<run>_trial<N>.ply        (our DEFOM ViT-S)

    Both variants use the same TURTLMap trajectory, so the pair isolates the depth
    estimator. A third tree, zed_depths/, holds TURTLMap's own ZED SDK depths; it is not
    scored -- the paper's baseline column is the DEFOM one, which shares our architecture.

    where <run> is the timestamped run directory that holds the trajectory. The mapper is
    deterministic, so only trial_0 was mapped; every other trial returns None and is
    skipped by the caller.
    """
    parts = traj_file.parts                       # .../<run>/<trial>/trajectory/surfslam.txt
    trial, run = parts[-3], parts[-4]
    variant = "our_depths" if ours else "baseline_depths"
    root = Path(maps_root) if maps_root else default_turtlmap_maps_root()
    scene_dir = root / variant / scene
    if not scene_dir.is_dir():
        return None

    # trial_0 -> trial0 in the map filenames.
    candidate = scene_dir / f"{run}_{trial.replace('_', '')}.ply"
    if candidate.exists():
        return candidate

    # Fall back to the only map in the directory, so a differently-named fused map still
    # resolves -- but only for trial_0, the one the deterministic mapper was run for.
    plys = sorted(p for p in scene_dir.glob("*.ply") if p.is_file())
    if len(plys) == 1 and trial in ("trial_0", "trial0"):
        return plys[0]
    return None


def get_pointcloud(method: str, traj_file: Path, ours: bool = False, scene: str = "",
                   turtlmap_maps_root=None, voxel_size=None) -> o3d.geometry.PointCloud:
    if "mast3r" in method.lower():
        if "_all_frames" in traj_file.stem:
            traj_file_root = traj_file.parent / traj_file.stem.replace("_all_frames", "")
            traj_file = str(traj_file_root) + ".txt"
        ply_file = Path(str(traj_file).replace(".txt", ".ply"))
        pred_pcd = o3d.io.read_point_cloud(str(ply_file))
    elif method.lower() == "droidslam":
        ply_file = Path(str(traj_file).replace("trajectory.txt", "points.ply"))
        pred_pcd = o3d.io.read_point_cloud(str(ply_file))
    elif method.lower() == "vggt_slam":
        submaps_dir = Path(str(traj_file).replace("trajectory.txt", "submaps"))
        if not submaps_dir.exists() or not submaps_dir.is_dir():
            raise FileNotFoundError(
                f"Submaps directory not found: {submaps_dir}")
        ply_files = sorted(glob.glob(str(submaps_dir / "submap_*.pcd")))
        if len(ply_files) == 0:
            raise FileNotFoundError(
                f"No submap PLY files found in: {submaps_dir}")
        pred_pcd = o3d.geometry.PointCloud()
        print(f"Found {len(ply_files)} submap PLY files")
        for pf in tqdm.tqdm(ply_files):
            submap_pcd = o3d.io.read_point_cloud(pf)
            pred_pcd += submap_pcd
    elif method.lower() == "svin2":
        pred_pcd = read_svin2_map(traj_file, voxel_size=voxel_size)
    elif "turtlmap" in method.lower():
        ply_file = turtlmap_map_path(traj_file, scene, ours=ours, maps_root=turtlmap_maps_root)
        if ply_file is None:
            return None
        pred_pcd = o3d.io.read_point_cloud(str(ply_file))
    elif "surfslam" in method.lower() or "ours" in method.lower():
        # Every trial saves its own map next to the trajectory, so there is nothing to
        # re-accumulate: score the map the run actually produced.
        ply_file = traj_file.parent.parent / "reconstructed_map.ply"
        if not ply_file.exists():
            raise FileNotFoundError(f"No reconstructed_map.ply beside {traj_file}")
        pred_pcd = o3d.io.read_point_cloud(str(ply_file))
    elif method.lower() == "orbslam3":
        print("ORB-SLAM3 evaluation not implemented yet")
        pred_pcd = None
    else:
        raise ValueError(f"Don't know where {method} keeps its maps")
    return pred_pcd


# def compute_aggregated_stats(df: pd.DataFrame, config: )
