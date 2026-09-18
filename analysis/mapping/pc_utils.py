import torch
import point_cloud_utils as pcu
from kaolin.metrics.pointcloud import chamfer_distance, sided_distance, f_score
import logging
import numpy as np
from typing import Tuple


def _precision_recall(gt_points, pred_points, radius=0.1):
    pred_distances = torch.sqrt(sided_distance(gt_points, pred_points)[0])
    gt_distances = torch.sqrt(sided_distance(pred_points, gt_points)[0])
    fn = torch.sum(pred_distances > radius, dim=1).type(gt_points.dtype)
    fp = torch.sum(gt_distances > radius, dim=1).type(gt_points.dtype)
    tp = (gt_distances.shape[1] - fp).type(gt_points.dtype)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision, recall


def _fscore(gt_points, pred_points, radius=0.1):
    return f_score(gt_points, pred_points, radius=radius)


def _hausdorff(src: torch.Tensor, dst: torch.Tensor):
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
    return torch.sqrt(dist).cpu().numpy()


def compute_point_cloud_metrics(pred_pts: torch.Tensor, gt_pts: torch.Tensor):
    chamfer_dist, chamfer_dist_l2 = _chamfer(pred_pts, gt_pts)
    accuracy = _sided_distance(pred_pts, gt_pts)
    completeness = _sided_distance(gt_pts, pred_pts)
    hausdorff = _hausdorff(pred_pts, gt_pts)
    precision, recall = _precision_recall(gt_pts, pred_pts, radius=0.1)
    fscore = _fscore(gt_pts, pred_pts, radius=0.1)

    return {
        'chamfer': chamfer_dist,
        'accuracy': accuracy.mean(),
        'completeness': completeness.mean(),
        'hausdorff': hausdorff,
        'precision': precision.item(),
        'recall': recall.item(),
        'fscore': fscore.item(),
    }, accuracy, completeness
