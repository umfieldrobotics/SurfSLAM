#!/usr/bin/env python3
"""
Valid Space Filtering Utilities
================================
Utilities for filtering point clouds to a valid space defined by a convex hull
and gravity direction.
"""

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
from typing import Tuple


def load_convex_hull_data(hull_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load convex hull vertices and gravity direction from file.
    
    Args:
        hull_file: Path to npz file containing hull data
        
    Returns:
        Tuple of (hull_vertices, gravity_direction)
    """
    data = np.load(hull_file)
    hull_vertices = data['hull_vertices']
    gravity_direction = data['gravity_direction']
    
    print(f"Loaded convex hull with {len(hull_vertices)} vertices")
    print(f"Gravity direction: [{gravity_direction[0]:.4f}, {gravity_direction[1]:.4f}, {gravity_direction[2]:.4f}]")
    
    return hull_vertices, gravity_direction


def save_convex_hull_data(hull_vertices_3d: np.ndarray, gravity_dir: np.ndarray, output_path: str):
    """
    Save convex hull vertices and gravity direction to a file.
    
    Args:
        hull_vertices_3d: Kx3 array of hull vertices in 3D world coordinates
        gravity_dir: 3D unit vector representing gravity direction
        output_path: Path to save the data (npz format)
    """
    np.savez(
        output_path,
        hull_vertices=hull_vertices_3d,
        gravity_direction=gravity_dir
    )
    print(f"Saved convex hull data to: {output_path}")


def project_points_to_2d_plane(points_3d: np.ndarray, gravity_dir: np.ndarray) -> np.ndarray:
    """
    Project 3D points onto 2D plane perpendicular to gravity direction.
    
    Args:
        points_3d: Nx3 array of 3D points
        gravity_dir: 3D unit vector representing gravity direction
        
    Returns:
        Nx2 array of 2D projected points
    """
    # Normalize gravity direction
    gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
    
    # Create orthonormal basis for the plane perpendicular to gravity
    # Find an arbitrary vector not parallel to gravity
    if abs(gravity_dir[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])
    
    # First basis vector (in plane)
    u = arbitrary - np.dot(arbitrary, gravity_dir) * gravity_dir
    u = u / np.linalg.norm(u)
    
    # Second basis vector (in plane, orthogonal to u)
    v = np.cross(gravity_dir, u)
    v = v / np.linalg.norm(v)
    
    # Project 3D points onto 2D plane using basis vectors
    points_2d = np.column_stack([
        np.dot(points_3d, u),
        np.dot(points_3d, v)
    ])
    
    return points_2d


def filter_points_by_convex_hull(
    points_3d: np.ndarray,
    hull_vertices_3d: np.ndarray,
    gravity_dir: np.ndarray
) -> np.ndarray:
    """
    Filter 3D points to only include those inside the convex hull when projected
    onto the plane perpendicular to gravity.
    
    The valid space is defined as any point that, when projected along the gravity
    direction onto the plane perpendicular to gravity, falls within the 2D convex
    hull boundary.
    
    Args:
        points_3d: Nx3 array of 3D points to filter
        hull_vertices_3d: Kx3 array of convex hull vertices in 3D
        gravity_dir: 3D unit vector representing gravity direction
        
    Returns:
        Boolean mask of length N indicating which points are inside the hull
    """
    # Normalize gravity direction
    gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
    
    # Project both the points and hull vertices to 2D
    points_2d = project_points_to_2d_plane(points_3d, gravity_dir)
    hull_2d = project_points_to_2d_plane(hull_vertices_3d, gravity_dir)
    
    # Create a matplotlib Path object from the hull vertices
    # The hull vertices should already be in order from ConvexHull
    hull_path = MplPath(hull_2d)
    
    # Check which points are inside the hull
    inside_mask = hull_path.contains_points(points_2d)
    
    return inside_mask


def filter_pointcloud_to_valid_space(
    points_3d: np.ndarray,
    hull_file: str,
    colors: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter a point cloud to only include points within the valid space defined
    by a convex hull and gravity direction.
    
    Args:
        points_3d: Nx3 array of 3D points
        hull_file: Path to npz file containing hull vertices and gravity direction
        colors: Optional Nx3 array of RGB colors
        
    Returns:
        Tuple of (filtered_points, filtered_colors)
        - filtered_points: Mx3 array of points inside the valid space
        - filtered_colors: Mx3 array of colors (or None if colors not provided)
    """
    hull_vertices, gravity_dir = load_convex_hull_data(hull_file)
    valid_mask = filter_points_by_convex_hull(points_3d, hull_vertices, gravity_dir)
    filtered_points = points_3d[valid_mask]
    
    filtered_colors = None
    if colors is not None:
        filtered_colors = colors[valid_mask]
    
    num_total = len(points_3d)
    num_valid = len(filtered_points)
    percentage = 100.0 * num_valid / num_total if num_total > 0 else 0.0
    
    print(f"Filtered {num_total} points to {num_valid} valid points ({percentage:.1f}%)")
    
    return filtered_points, filtered_colors
