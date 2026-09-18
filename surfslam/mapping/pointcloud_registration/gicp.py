import small_gicp
import numpy as np



def compute_covariance_from_hessian(H, scale_factor=1.0, epsilon=1e-6):
    """
    Compute covariance matrix from the Hessian matrix.

    For least-squares optimization, cov ≈ scale * H^{-1}
    The Hessian from GICP is in the tangent space (6D: rotation, translation).

    Args:
        H: 6x6 Hessian matrix from GICP optimization (in [rot, trans] order)
        scale_factor: Optional scaling factor (e.g., based on residual variance)

    Returns:
        cov: 6x6 covariance matrix in tangent space [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
    """
    try:
        # Add small regularization for numerical stability
        H_reg = H + np.eye(6) * epsilon
        cov_rot_trans = scale_factor * np.linalg.inv(H_reg)

        # Reorder from [rot, trans] to [trans, rot] to match code convention
        # Original: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        # Target:   [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
        reorder_indices = [3, 4, 5, 0, 1, 2]
        cov = cov_rot_trans[np.ix_(reorder_indices, reorder_indices)]

        return cov
    except np.linalg.LinAlgError:
        print("Warning: Hessian is singular, returning large covariance")
        return np.eye(6) * 1e6


def run_gicp(source_points,
             target_points,
             initial_transform,
             max_correspondence_distance=0.1,
             max_iterations=100,
             downsampling_resolution=None,
             num_threads=4):
    """
    Run GICP using small_gicp library (returns Hessian for covariance).

    Args:
        source_points: Nx3 numpy array (source/current frame)
        target_points: Nx3 numpy array (target/previous frame)
        initial_transform: 4x4 numpy array (initial transformation estimate)
        max_correspondence_distance: Maximum correspondence distance, or list of distances
            for coarse-to-fine refinement (each result initializes the next)
        max_iterations: Maximum iterations
        downsampling_resolution: Optional voxel size for downsampling (None = no downsampling)
        num_threads: Number of threads to use

    Returns:
        result_dict: Dictionary containing transform, Hessian, covariance, and metrics
    """
    # Convert single distance to list for uniform handling
    if isinstance(max_correspondence_distance, (int, float)):
        distance_thresholds = [max_correspondence_distance]
    else:
        distance_thresholds = list(max_correspondence_distance)

    # Preprocess point clouds (only once)
    if downsampling_resolution is not None:
        downsampling_resolution = min(downsampling_resolution, min(distance_thresholds))
        target, target_tree = small_gicp.preprocess_points(
            target_points.astype(np.float64),
            downsampling_resolution=downsampling_resolution,
            num_threads=num_threads
        )
        source, source_tree = small_gicp.preprocess_points(
            source_points.astype(np.float64),
            downsampling_resolution=downsampling_resolution,
            num_threads=num_threads
        )
    else:
        # Manual preprocessing without downsampling
        target = small_gicp.PointCloud(target_points.astype(np.float64))
        source = small_gicp.PointCloud(source_points.astype(np.float64))
        target_tree = small_gicp.KdTree(target)
        source_tree = small_gicp.KdTree(source)
        small_gicp.estimate_covariances(target, target_tree)
        small_gicp.estimate_covariances(source, source_tree)

    # Run GICP with coarse-to-fine distance thresholds
    current_transform = initial_transform.astype(np.float64)
    total_iterations = 0

    for dist_threshold in distance_thresholds:
        result = small_gicp.align(
            target, source, target_tree,
            init_T_target_source=current_transform,
            max_correspondence_distance=dist_threshold,
            num_threads=num_threads,
            registration_type='GICP'
        )

        current_transform = result.T_target_source
        total_iterations += result.iterations

    # Extract results from final pass
    T = result.T_target_source
    H = result.H  # 6x6 Hessian matrix

    # Compute covariance from Hessian
    # For GICP, the Hessian approximates the Fisher information matrix
    # Covariance ≈ H^{-1} (or scaled version based on residual variance)
    cov = compute_covariance_from_hessian(H)

    # Compute fitness and RMSE (approximate from small_gicp results)
    num_inliers = result.num_inliers
    total_points = source.size()
    fitness = num_inliers / total_points if total_points > 0 else 0

    # Estimate RMSE from final error
    inlier_rmse = np.sqrt(result.error / max(num_inliers, 1)) if num_inliers > 0 else float('inf')

    return {
        'transformation': T,
        'hessian': H,
        'covariance': cov,
        'fitness': fitness,
        'inlier_rmse': inlier_rmse,
        'num_inliers': num_inliers,
        'converged': result.converged,
        'iterations': total_iterations,
        'final_error': result.error
    }