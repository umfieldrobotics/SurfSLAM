import pickle
import torch
import numpy as np
from common.frame import Frame
from common.sensors import Image
from common.settings import Settings
from kornia.geometry.depth import depth_to_3d_v2


class FeatureAlignmentRegistration:
    def __init__(self, settings: Settings, calibration: Settings) -> None:
        self._settings = settings
        self._K = calibration.camera_intrinsic.k
        self._calibration = calibration

    def start(self):
        """Initialize the registration solver (no-op for this implementation)."""
        pass

    def solve(
        self,
        prev_frame: Frame,
        prev_mkpts: torch.Tensor,
        curr_frame: Frame,
        curr_mkpts: torch.Tensor,
        debug_file=None,
        threshold=None):
        """
        Solve for the transformation between two frames using feature alignment with RANSAC.

        Args:
            prev_frame: Target/reference frame
            prev_mkpts: Matched keypoints in prev_frame (N, 2)
            curr_frame: Source/current frame
            curr_mkpts: Matched keypoints in curr_frame (N, 2)
            debug_file: Optional file path for debug visualization

        Returns:
            tuple: (estimated_transform, inlier_mask, pct_inlier)
        """
        
        prev_depth = prev_frame.depth_image.image.squeeze()
        curr_depth = curr_frame.depth_image.image.squeeze()

        prev_uv = prev_mkpts.int()
        curr_uv = curr_mkpts.int()

        prev_depth_valid = prev_depth[prev_uv[:,1], prev_uv[:,0]] > 1e-3
        curr_depth_valid = curr_depth[curr_uv[:,1], curr_uv[:,0]] > 1e-3
        valid_mask = (prev_depth_valid & curr_depth_valid)
        valid_indices = valid_mask.flatten().nonzero()
        
        if valid_mask.sum() == 0:
            # No valid matches
            return np.eye(4), np.zeros(len(prev_mkpts), dtype=bool)

        N_valid_pts = len(valid_indices)
        good_prev_keypoints = prev_mkpts.cuda()[valid_mask]
        good_curr_keypoints = curr_mkpts.cuda()[valid_mask]

        # Build 3D point clouds from keypoints
        prev_cloud = self._get_compute_points(prev_frame, good_prev_keypoints)
        curr_cloud = self._get_compute_points(curr_frame, good_curr_keypoints)

        # RANSAC parameters
        n_ransac = min(self._settings.N_points, prev_cloud.shape[0])
        ransac_threshold = self._settings.threshold if threshold is None else threshold
        ransac_its = self._settings.n_iterations

        # generates a N_iterations x N_pts matirx of the chosen samples
        all_samples = torch.randperm(N_valid_pts * ransac_its, device='cuda:0').reshape(ransac_its, N_valid_pts)[:, :n_ransac] % N_valid_pts

        # N_iterations, n_ransac, 3
        all_prev_pts = prev_cloud[all_samples]
        all_curr_pts = curr_cloud[all_samples]
        transforms = self._estimate_transform(all_prev_pts, all_curr_pts)
        all_inliers = self._compute_inliers(prev_cloud, curr_cloud, transforms, ransac_threshold)
        all_n_inliers = all_inliers.sum(1)
        best_candidate = all_n_inliers.argmax()
        n_inliers = all_n_inliers[best_candidate]
        selected_inliers = all_inliers[best_candidate]

        # Check minimum inliers requirement
        if n_inliers < self._settings.min_inliers:
            return np.eye(4), np.zeros(len(prev_mkpts), dtype=bool)

        # Refine transformation using all inliers
        final_prev_pts = prev_cloud[selected_inliers]
        final_curr_pts = curr_cloud[selected_inliers]

        # check wall thickness
        prev_wall_thickness = self._compute_wall_thickness(final_prev_pts)
        curr_wall_thickness = self._compute_wall_thickness(final_curr_pts)

        if prev_wall_thickness is None or curr_wall_thickness is None \
           or (prev_wall_thickness < self._settings.min_wall_thickness \
           and curr_wall_thickness < self._settings.min_wall_thickness):
            
            return np.eye(4), np.zeros(len(prev_mkpts), dtype=bool)

        T_prev_to_curr = self._estimate_transform(final_prev_pts, final_curr_pts)
        
        # Convert back to full-size inlier mask (including invalid depth matches)
        full_inlier_mask = np.zeros(len(prev_mkpts), dtype=bool)
        for local_idx, global_idx in enumerate(valid_indices):
            if selected_inliers[local_idx]:
                full_inlier_mask[global_idx] = True

        pct_inlier = full_inlier_mask.mean() if len(prev_mkpts) > 0 else 0.0

        # Convert to numpy
        estimated_transform = T_prev_to_curr.detach().cpu().numpy()


        # Debug visualization if requested
        if debug_file:
            import open3d as o3d
            import matplotlib.pyplot as plt

            # 2D correspondences visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Prev frame
            ax = axes[0]
            if hasattr(prev_frame, 'color_image') and prev_frame.color_image is not None:
                prev_img = prev_frame.color_image.image
                if isinstance(prev_img, torch.Tensor):
                    prev_img = prev_img.detach().cpu().numpy()
                if prev_img.ndim == 3 and prev_img.shape[0] == 3:
                    prev_img = np.transpose(prev_img, (1, 2, 0))
                ax.imshow(prev_img)
            else:
                prev_depth_vis = prev_frame.depth_image.image.squeeze().detach().cpu().numpy()
                ax.imshow(prev_depth_vis, cmap='viridis')

            # Plot all matches in red
            ax.scatter(good_prev_keypoints[:, 0].cpu().numpy(),
                      good_prev_keypoints[:, 1].cpu().numpy(),
                      c='red', s=30, alpha=0.5, label='All matches')
            # Plot inliers in green
            inlier_prev_kpts = good_prev_keypoints[selected_inliers]
            ax.scatter(inlier_prev_kpts[:, 0].cpu().numpy(),
                      inlier_prev_kpts[:, 1].cpu().numpy(),
                      c='lime', s=50, marker='x', linewidths=2, label='Inliers')
            ax.set_title(f'Prev Frame: {selected_inliers.sum()}/{len(good_prev_keypoints)} inliers')
            ax.legend()
            ax.axis('off')

            # Curr frame
            ax = axes[1]
            if hasattr(curr_frame, 'color_image') and curr_frame.color_image is not None:
                curr_img = curr_frame.color_image.image
                if isinstance(curr_img, torch.Tensor):
                    curr_img = curr_img.detach().cpu().numpy()
                if curr_img.ndim == 3 and curr_img.shape[0] == 3:
                    curr_img = np.transpose(curr_img, (1, 2, 0))
                ax.imshow(curr_img)
            else:
                curr_depth_vis = curr_frame.depth_image.image.squeeze().detach().cpu().numpy()
                ax.imshow(curr_depth_vis, cmap='viridis')

            ax.scatter(good_curr_keypoints[:, 0].cpu().numpy(),
                      good_curr_keypoints[:, 1].cpu().numpy(),
                      c='red', s=30, alpha=0.5, label='All matches')
            inlier_curr_kpts = good_curr_keypoints[selected_inliers]
            ax.scatter(inlier_curr_kpts[:, 0].cpu().numpy(),
                      inlier_curr_kpts[:, 1].cpu().numpy(),
                      c='lime', s=50, marker='x', linewidths=2, label='Inliers')
            ax.set_title(f'Curr Frame: {pct_inlier*100:.1f}% inlier rate')
            ax.legend()
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(debug_file.replace('.pkl', '_2d_matches.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # 3D correspondences visualization
            fig = plt.figure(figsize=(18, 6))

            prev_3d = prev_cloud.detach().cpu().numpy()
            curr_3d = curr_cloud.detach().cpu().numpy()
            curr_3d_transformed = (estimated_transform @ np.hstack([curr_3d, np.ones((curr_3d.shape[0], 1))]).T).T[:, :3]
            inlier_mask_np = selected_inliers.cpu().numpy()

            # Before alignment
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(prev_3d[:, 0], prev_3d[:, 1], prev_3d[:, 2],
                       c='blue', s=30, alpha=0.6, label='Prev')
            ax1.scatter(curr_3d[:, 0], curr_3d[:, 1], curr_3d[:, 2],
                       c='red', s=30, alpha=0.6, label='Curr')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('Before Alignment')
            ax1.legend()

            # After alignment
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(prev_3d[:, 0], prev_3d[:, 1], prev_3d[:, 2],
                       c='blue', s=30, alpha=0.6, label='Prev')
            ax2.scatter(curr_3d_transformed[:, 0], curr_3d_transformed[:, 1], curr_3d_transformed[:, 2],
                       c='orange', s=30, alpha=0.6, label='Curr (aligned)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title('After Alignment')
            ax2.legend()

            # Inlier correspondences with lines
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.scatter(prev_3d[inlier_mask_np, 0], prev_3d[inlier_mask_np, 1], prev_3d[inlier_mask_np, 2],
                       c='blue', s=40, alpha=0.8, label='Prev inliers')
            ax3.scatter(curr_3d_transformed[inlier_mask_np, 0], curr_3d_transformed[inlier_mask_np, 1],
                       curr_3d_transformed[inlier_mask_np, 2],
                       c='orange', s=40, alpha=0.8, label='Curr inliers')

            for i in np.where(inlier_mask_np)[0]:
                ax3.plot([prev_3d[i, 0], curr_3d_transformed[i, 0]],
                        [prev_3d[i, 1], curr_3d_transformed[i, 1]],
                        [prev_3d[i, 2], curr_3d_transformed[i, 2]],
                        'g-', alpha=0.4, linewidth=1)

            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.set_title(f'Inliers Only ({selected_inliers.sum()} pairs)')
            ax3.legend()

            plt.tight_layout()
            plt.savefig(debug_file.replace('.pkl', '_3d_matches.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved 2D matches to {debug_file.replace('.pkl', '_2d_matches.png')}")
            print(f"Saved 3D matches to {debug_file.replace('.pkl', '_3d_matches.png')}")

            # Original full point cloud visualization
            target_full_pcd = self._frame_to_pcd(prev_frame)
            source_full_pcd = self._frame_to_pcd(curr_frame)
            pcd_target = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(
                    target_full_pcd.detach().cpu().numpy()
                )
            )
            # voxel downsample for faster visualization
            pcd_target = pcd_target.voxel_down_sample(voxel_size=0.05)
            pcd_target.paint_uniform_color([0.0, 1.0, 0.0])
            source_numpy = source_full_pcd.detach().cpu().numpy()
            pcd_source = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(source_numpy.copy())
            )
            pcd_source = pcd_source.voxel_down_sample(voxel_size=0.05)
            pcd_source.paint_uniform_color([1.0, 0.0, 1.0])
            pcd_source_persistent = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(source_numpy.copy())
            )
            pcd_source_persistent = pcd_source_persistent.voxel_down_sample(voxel_size=0.05)
            pcd_source_persistent.paint_uniform_color([1.0, 0.0, 0.0])
            pcd_source.transform(estimated_transform)

            # estimate normals for better visualization
            pcd_target.estimate_normals()
            pcd_source.estimate_normals()
            pcd_source_persistent.estimate_normals()

            data = {
                'target': np.asarray(pcd_target.points),
                'source': np.asarray(pcd_source.points),
                'source_persistent': np.asarray(pcd_source_persistent.points),
                'features_prev': prev_3d,
                'features_curr': curr_3d_transformed,
                'inlier_mask': inlier_mask_np,
                'transform': estimated_transform,
                "target_time": prev_frame.get_time(),
                "source_time": curr_frame.get_time(),
                'prev_uv': prev_mkpts.cpu().numpy(),
                'curr_uv': curr_mkpts.cpu().numpy(),
                'valid_depth_mask': valid_mask.cpu().numpy(),
                'valid_indices': valid_indices.cpu().numpy().flatten(),
                'inlier_mask_among_valid': selected_inliers.cpu().numpy(),
                'full_inlier_mask': full_inlier_mask,
                'n_total_matches': len(prev_mkpts),
                'n_valid_depth': valid_mask.sum().item(),
                'n_inliers': full_inlier_mask.sum(),
                'pct_inlier': pct_inlier,
            }

            with open(debug_file, 'wb+') as f:
                pickle.dump(data, f)

            print(f"Saved full point cloud data to {debug_file}")

        return estimated_transform, full_inlier_mask

    def _get_compute_points(self, frame: Frame, mkpts: torch.Tensor):
        """Convert 2D keypoints to 3D points using depth map."""
        depth_frame = frame.depth_image
        if depth_frame is None:
            raise ValueError("Depth image is required for registration")
        depth_tensor = depth_frame.image

        device = mkpts.device if isinstance(mkpts, torch.Tensor) else torch.device("cpu")

        if isinstance(depth_tensor, np.ndarray):
            depth_tensor = torch.from_numpy(depth_tensor).to(device)
        elif isinstance(depth_tensor, torch.Tensor):
            depth_tensor = depth_tensor.to(device)
        else:
            raise TypeError("Depth image must be a torch.Tensor or numpy.ndarray")

        if depth_tensor.dim() == 3 and depth_tensor.shape[0] == 1:
            depth_tensor = depth_tensor[0]
        elif depth_tensor.dim() == 3 and depth_tensor.shape[0] != 1:
            depth_tensor = depth_tensor.squeeze(0)
        elif depth_tensor.dim() == 2:
            pass
        else:
            raise ValueError(
                "Unsupported depth tensor shape: {}".format(depth_tensor.shape)
            )

        if not isinstance(self._K, torch.Tensor):
            K = torch.as_tensor(self._K, device=device, dtype=depth_tensor.dtype)
        else:
            K = self._K.to(device=device, dtype=depth_tensor.dtype)

        return self._keypoints_to_3d(mkpts.to(device), depth_tensor, K)

    def _compute_wall_thickness(self, pts_3d: torch.Tensor) -> float:
        if len(pts_3d) < 3:
            return None
        
        centroid = torch.mean(pts_3d, dim=0)
        centered = pts_3d - centroid

        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        normal = Vt.transpose(0,1)[:, -1:]
        distances = centered @ normal
        return distances.max() - distances.min()


    def _keypoints_to_3d(
        self, keypoints: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor
    ):
        """Convert 2D keypoints to 3D points using depth map."""
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        image_height, image_width = depth.shape
        keypoints_xy = keypoints
        pixel_x_int = torch.round(keypoints[:, 0]).int()
        pixel_y_int = torch.round(keypoints[:, 1]).int()
        valid_mask = (
            (pixel_x_int >= 0)
            & (pixel_x_int < image_width)
            & (pixel_y_int >= 0)
            & (pixel_y_int < image_height)
        )
        # Indexing requires long indices; clamp to valid range to avoid indexing errors
        pixel_x_int_valid = pixel_x_int[valid_mask]
        pixel_y_int_valid = pixel_y_int[valid_mask]
        pixel_x_float_valid = keypoints_xy[valid_mask, 0]
        pixel_y_float_valid = keypoints_xy[valid_mask, 1]

        depth_values = depth[pixel_y_int_valid, pixel_x_int_valid]
        depth_valid_mask = (depth_values > 1e-3) & torch.isfinite(depth_values)
        final_x = pixel_x_float_valid[depth_valid_mask]
        final_y = pixel_y_float_valid[depth_valid_mask]
        final_depth = depth_values[depth_valid_mask].float()
        depth_x = (final_x - cx) * final_depth / fx
        depth_y = (final_y - cy) * final_depth / fy
        points_cam = torch.stack([depth_x, depth_y, final_depth], dim=-1)

        return points_cam[valid_mask]

    def _estimate_transform(self, ref_pts: torch.Tensor, frame_pts: torch.Tensor):
        """
        Estimate rigid transformation using SVD (Orthogonal Procrustes problem).

        Args:
            ref_pts: Reference points (B?, N, 3)
            frame_pts: Frame points (B?, N, 3)

        Returns:
            (B?,4,4) transformation matrix
        """
        # force it up to 3D (B, N, 3)
        ref_pts = ref_pts.view(-1, ref_pts.shape[-2], ref_pts.shape[-1])
        frame_pts = frame_pts.view(-1, frame_pts.shape[-2], frame_pts.shape[-1])

        B = ref_pts.shape[0]
        
        ref_center = torch.mean(ref_pts, dim=-2, keepdim=True) # B x 1 x 3
        frame_center = torch.mean(frame_pts, dim=-2, keepdim=True) # B x 1 x 3

        ref_pts_centered = ref_pts - ref_center
        frame_pts_centered = frame_pts - frame_center

        cov = ref_pts_centered.transpose(-2, -1) @ frame_pts_centered

        U, _, Vt = torch.linalg.svd(cov)

        R: torch.Tensor = U @ Vt
        reflection_mask = torch.linalg.det(R) < 0
        # Correct for reflection
        # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        S = torch.eye(3, device=R.device, dtype=R.dtype).repeat(B, 1, 1)
        S[reflection_mask, 2, 2] = -1
        R = U @ S @ Vt

        t = ref_center.flatten(1) - (R @ frame_center.transpose(-2, -1)).flatten(1)

        transform = torch.cat((R, t.view(-1, 3, 1)), dim=-1)
        h_row = torch.tensor([[0, 0, 0, 1]], device=R.device, dtype=R.dtype).repeat(B, 1, 1)
        T = torch.cat((transform, h_row), dim=-2)
        return T.squeeze()

    def _compute_inliers(
        self,
        ref_pts: torch.Tensor,
        frame_pts: torch.Tensor,
        T: torch.Tensor,
        threshold: float
    ):
        """
        Compute inlier mask based on transformation and distance threshold.

        Args:
            ref_pts: Reference points (N, 3)
            frame_pts: Frame points (N, 3)
            T: B?, 4x4 transformation matrix
            threshold: Distance threshold for inliers

        Returns:
            B?xN Boolean tensor indicating inliers
        """ 
        # force to 1, N, 3
        ref_pts = ref_pts.view(-1, ref_pts.shape[-2], ref_pts.shape[-1])
        frame_pts = frame_pts.view(-1, frame_pts.shape[-2], frame_pts.shape[-1])
        N = ref_pts.shape[1]

        T = T.view(-1, 4, 4)

        h_row = torch.ones(ref_pts.shape[:-1], device=ref_pts.device).reshape(1,1,N) # 1xNx1
        frame_points_homog = torch.cat((frame_pts.transpose(-2, -1), h_row), dim=1)
        transformed_frame_pts = (T @ frame_points_homog)[:, :3].transpose(-2, -1)

        dists = (ref_pts - transformed_frame_pts).norm(dim=-1)
      
        return dists <= threshold

    def _frame_to_pcd(self, frame: Frame):
        """Convert frame depth image to full point cloud for visualization."""
        depth_frame: Image | None = frame.depth_image
        if depth_frame is None:
            raise ValueError("Depth image is required for visualization")
        depth_tensor = depth_frame.image

        depth_tensor[depth_tensor > 10.0] = 0.0  # filter out large depths

        device = (
            depth_tensor.device
            if isinstance(depth_tensor, torch.Tensor)
            else torch.device("cpu")
        )

        if not isinstance(self._K, torch.Tensor):
            K = torch.as_tensor(self._K, device=device, dtype=depth_tensor.dtype)
        else:
            K = self._K.to(device=device, dtype=depth_tensor.dtype)

        pcd = depth_to_3d_v2(depth_tensor.unsqueeze(0), K.unsqueeze(0))  # (1, H, W, 3)
        pcd = pcd.view(-1, 3)  # (H*W, 3)

        return pcd
