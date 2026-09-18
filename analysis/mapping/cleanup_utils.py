import numpy as np
import open3d as o3d
import torch
# @TODO: bu da hardcoded. ponyo cfg'den oku
ATM_PRESSURE = 986.3999633789062


def reject_depth_edges(depth, threshold=0.03):
    dzdx = torch.abs(depth[:, 1:] - depth[:, :-1])
    dzdy = torch.abs(depth[1:, :] - depth[:-1, :])
    dzdx = torch.nn.functional.pad(dzdx, (0, 1))
    dzdy = torch.nn.functional.pad(dzdy, (0, 0, 0, 1))
    edge_mask = (dzdx > threshold) | (dzdy > threshold)
    depth_clean = depth.clone()
    depth_clean[edge_mask] = 0.0
    return depth_clean



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
        np.dot(p0_world - p_world_cam, n_world))

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

    return valid_mask


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