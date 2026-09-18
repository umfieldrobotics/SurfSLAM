import torch
from pytorch3d.ops import knn_points, ball_query
import math

class GPUPointCloudCleaner:
    def __init__(
        self,
        settings,
        device='cuda:0'
    ):
        self.sor_neighbors = settings.sor_neighbors
        self.sor_std = settings.sor_std
        self.ror_radius = settings.ror_radius
        self.ror_min_pts = settings.ror_min_pts
        self.device = torch.device(device)
        self.enable_sor = settings.enable_sor
        self.enable_ror = settings.enable_ror
        self.target_point_count = settings.target_point_count
        self.max_depth_m = settings.max_depth_m

    def clean(self, points, colors=None):
        downsample_factor = math.ceil(points.shape[0] / self.target_point_count)
        points = points[::downsample_factor]
        depth_max_mask = points[:, 2] < self.max_depth_m
        zero_depth_mask = points[:, 2] > 0.0
        depth_max_mask = depth_max_mask & zero_depth_mask
        points = points[depth_max_mask]
        if colors is not None:
            colors = colors[::downsample_factor]
            colors = colors[depth_max_mask]

        pts = points.to(dtype=torch.float32, device=self.device).unsqueeze(0)

        # ---- SOR ----
        if self.enable_sor:
            mask_sor = self._sor(pts)

            pts = pts[:, mask_sor]
            if colors is not None:
                cols = colors.to(device=self.device)[mask_sor]
            else:
                cols = None
            if pts.shape[1] == 0:
                empty_pts = torch.empty((0, 3), dtype=points.dtype, device="cpu")
                empty_cols = torch.empty((0, colors.shape[1]), dtype=colors.dtype, device="cpu") if colors is not None else None
                return empty_pts, empty_cols
        else:
            cols = colors.to(device=self.device) if colors is not None else None

        # ---- ROR ----
        if self.enable_ror:
            mask_ror = self._ror(pts)
            pts = pts[:, mask_ror]
            if cols is not None:
                cols = cols[mask_ror]

        clean_pts = pts[0].cpu()
        clean_cols = cols.cpu() if colors is not None else None

        return clean_pts, clean_cols


    def _sor(self, pts):
        k = self.sor_neighbors + 1
        dists_sq, _, _ = knn_points(pts, pts, K=k)
        mean_dists = torch.sqrt(dists_sq[0, :, 1:]).mean(dim=1)
        thresh = mean_dists.mean() + self.sor_std * mean_dists.std()
        return mean_dists < thresh

    def _ror(self, pts):
        _, idx, _ = ball_query(
            pts, pts,
            K=self.ror_min_pts + 10,
            radius=self.ror_radius,
            return_nn=False
        )
        counts = (idx[0] >= 0).sum(dim=1)
        neighbor_counts = counts - 1
        return neighbor_counts >= self.ror_min_pts
