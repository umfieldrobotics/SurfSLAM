#!/usr/bin/env python3

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import cv2
import numpy as np
import torch
import tqdm
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "surfslam"))

from surfslam.mapping.place_recognition.image_descriptor import ImageDescriptorManager
from surfslam.mapping.descriptors.matcher import SuperGlueMatcher
from surfslam.common.settings import Settings
from examples.tbnms.tbnms_calibration import TBNMSCalibration

# Calibration loading (same as compute_netvlad_vocab.py)
from examples.utils.pose_utils import load_calibration

# SuperGlue imports
from models.matching import Matching
from models.utils import make_matching_plot

torch.set_grad_enabled(False)


class StereoProcessor:
    """Handles stereo rectification and undistortion."""
    
    def __init__(self, calibration, scale: float = 0.5):
        """
        Initialize with a calibration object (e.g., TBNMSCalibration).
        
        Args:
            calibration: Calibration object with rectify_images method
            scale: Image scale factor
        """
        self.calibration: TBNMSCalibration = calibration
        self.scale = scale
        
        self.map1_left = None
        self.map2_left = None
        self.newK = None
        self.new_size = None
        self._initialized = False
        
    def initialize(self, img_shape: Tuple[int, int]):
        """Compute stereo rectification maps from calibration."""
        h, w = img_shape[:2]
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        self.new_size = (new_w, new_h)
        
        # Get calibration parameters from the calibration object
        K0 = self.calibration.K0
        D0 = self.calibration.D0
        K1 = self.calibration.K1
        D1 = self.calibration.D1
        R = self.calibration.R
        T = self.calibration.t
        
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K0, D0, K1, D1, (w, h), R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=self.new_size
        )
        
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, self.new_size, cv2.CV_32FC1
        )
        self.newK = P1[:3, :3].copy()
        self._initialized = True
        
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Apply undistortion using precomputed maps."""
        if not self._initialized:
            raise RuntimeError("StereoProcessor not initialized. Call initialize() first.")
        return cv2.remap(frame, self.map1_left, self.map2_left, 
                        interpolation=cv2.INTER_LINEAR)
    
    def undistort_to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Undistort and convert to grayscale."""
        undistorted = self.undistort(frame)
        return cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    def rectify_images(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair using calibration object's method.
        This delegates to the calibration object for consistency with compute_netvlad_vocab.py.
        """
        return self.calibration.rectify_images(left, right, self.scale)


# =============================================================================
# Video frame extraction utilities
# =============================================================================
class VideoFrameExtractor:
    """Handles video frame extraction."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def extract_frame(self, t_sec: float) -> Optional[np.ndarray]:
        """Extract a frame at a given timestamp."""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = self.cap.read()
        return frame if ok else None
    
    def close(self):
        """Release video capture."""
        self.cap.release()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# RANSAC filtering
# =============================================================================
def filter_matches_essential_ransac(
    mkpts0: np.ndarray, 
    mkpts1: np.ndarray, 
    K: np.ndarray, 
    mconf: np.ndarray,
    ransac_threshold: float = 1.0, 
    confidence: float = 0.999
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Filter matches using Essential Matrix RANSAC."""
    if len(mkpts0) < 5:
        return mkpts0, mkpts1, mconf, None, np.ones(len(mkpts0), dtype=bool)
    
    E, mask = cv2.findEssentialMat(
        mkpts0, mkpts1, cameraMatrix=K, method=cv2.RANSAC,
        prob=confidence, threshold=ransac_threshold
    )
    
    if mask is None:
        return mkpts0, mkpts1, mconf, None, np.ones(len(mkpts0), dtype=bool)
    
    inlier_mask = mask.ravel().astype(bool)
    return (mkpts0[inlier_mask], mkpts1[inlier_mask], 
            mconf[inlier_mask], E, inlier_mask)


# =============================================================================
# Main video matcher class
# =============================================================================
class VideoMatcher:
    """
    Two-phase video frame matcher using VLAD global descriptors
    and SuperGlue local matching.
    """
    
    def __init__(self,
                 vocab_path: str,
                 device: str = 'cuda:0',
                 superglue_config: Dict = None):
        self.device = torch.device(device)
        
        # Load pre-fitted VLAD vocabulary
        print(f"Loading VLAD vocabulary from: {vocab_path}")
        self.descriptor = ImageDescriptorManager.from_file(
            vocab_path, device=self.device
        )
        print(f"  Clusters: {self.descriptor.n_clusters}")
        print(f"  Embedding dim: {self.descriptor.embedding_dim}")
        
        # Initialize SuperGlue matcher
        if superglue_config is None:
            superglue_config = {
                "superpoint": {
                    "nms_radius": 4,
                    "keypoint_threshold": 0.005,
                    "max_keypoints": 1024,
                },
                "superglue": {
                    "weights": "outdoor",
                    "sinkhorn_iterations": 20,
                    "match_threshold": 0.1,
                },
            }
        self.matching = Matching(superglue_config).eval().to(self.device)
        print("SuperGlue initialized")
        
    def _frame_to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        """Convert grayscale frame to tensor."""
        tensor = torch.from_numpy(gray).float().to(self.device) / 255.0
        return tensor[None, None]
    
    def extract_embedding(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract VLAD embedding from grayscale image."""
        tensor = self._frame_to_tensor(gray)
        
        with torch.no_grad():
            pred = self.matching.superpoint({"image": tensor})
            
        descriptors = pred["descriptors"][0].cpu().numpy().T
        scores = pred["scores"][0].cpu().numpy()
        
        embedding = self.descriptor.aggregate(descriptors, scores)
        return embedding, descriptors
    
    def compute_global_similarities(
        self,
        video: VideoFrameExtractor,
        stereo: StereoProcessor,
        ref_embedding: np.ndarray,
        timestamps: np.ndarray,
        show_progress: bool = True
    ) -> List[Dict]:
        """Compute global similarities for all candidate timestamps."""
        candidates = []
        
        iterator = tqdm.tqdm(timestamps, desc="Global embeddings") if show_progress else timestamps
        
        for t in iterator:
            frame = video.extract_frame(t)
            if frame is None:
                continue
                
            gray = stereo.undistort_to_gray(frame)
            embedding, _ = self.extract_embedding(gray)
            similarity = self.descriptor.compute_similarity(ref_embedding, embedding)
            
            candidates.append({
                "timestamp": t,
                "embedding": embedding,
                "similarity": similarity
            })
            
        return candidates
    
    def run_local_matching(
        self,
        video: VideoFrameExtractor,
        stereo: StereoProcessor,
        ref_gray: np.ndarray,
        ref_tensor: torch.Tensor,
        candidates: List[Dict],
        ransac_threshold: float = 1.0,
        ransac_confidence: float = 0.999,
        show_progress: bool = True
    ) -> List[Dict]:
        """Run SuperGlue local matching on selected candidates."""
        results = []
        
        iterator = tqdm.tqdm(candidates, desc="Local matching") if show_progress else candidates
        
        for c in iterator:
            t = c["timestamp"]
            
            frame = video.extract_frame(t)
            if frame is None:
                continue
                
            tgt_gray = stereo.undistort_to_gray(frame)
            tgt_tensor = self._frame_to_tensor(tgt_gray)
            
            # Run SuperGlue matching
            pred = self.matching({"image0": ref_tensor, "image1": tgt_tensor})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            
            k0 = pred["keypoints0"]
            k1 = pred["keypoints1"]
            m = pred["matches0"]
            conf = pred["matching_scores0"]
            
            valid = m > -1
            mk0 = k0[valid]
            mk1 = k1[m[valid]]
            mconf = conf[valid]
            
            raw_match_count = len(mk0)
            
            # Essential Matrix RANSAC filtering
            if len(mk0) >= 5:
                mk0, mk1, mconf, E, inlier_mask = filter_matches_essential_ransac(
                    mk0, mk1, stereo.newK, mconf,
                    ransac_threshold=ransac_threshold,
                    confidence=ransac_confidence
                )
                filtered_match_count = len(mk0)
            else:
                filtered_match_count = raw_match_count
                
            results.append({
                "timestamp": t,
                "global_similarity": c["similarity"],
                "raw_matches": raw_match_count,
                "filtered_matches": filtered_match_count,
                "mean_match_conf": float(np.mean(mconf)) if len(mconf) > 0 else 0.0,
                "keypoints0": k0,
                "keypoints1": k1,
                "matched_kpts0": mk0,
                "matched_kpts1": mk1,
                "match_conf": mconf,
                "gray_image": tgt_gray
            })
            
        return results


# =============================================================================
# Visualization utilities
# =============================================================================
def _save_summary_grid(
    ref_gray: np.ndarray,
    results: List[Dict],
    ref_time: float,
    output_path: Path,
    max_results: int = 9
):
    """Save a summary grid showing reference and top matches."""
    n_results = min(len(results), max_results)
    n_cols = 3
    n_rows = 1 + (n_results + n_cols - 1) // n_cols  # 1 row for ref + rows for results
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    # Turn off all axes first
    for ax in axes:
        ax.axis('off')
    
    # Reference image in first position
    axes[0].imshow(ref_gray, cmap='gray')
    axes[0].set_title(f"REFERENCE\nt={ref_time:.1f}s", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Leave positions 1,2 empty in first row (or use for stats)
    axes[1].text(0.5, 0.5, f"Top {n_results} Matches\nby filtered match count",
                 ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
    
    # Results in remaining positions
    for i, r in enumerate(results[:max_results]):
        ax = axes[i + n_cols]  # Start from second row
        ax.imshow(r["gray_image"], cmap='gray')
        
        # Color code by match quality
        if r["filtered_matches"] >= 50:
            title_color = 'green'
        elif r["filtered_matches"] >= 20:
            title_color = 'orange'
        else:
            title_color = 'red'
            
        ax.set_title(
            f"Rank {i+1}: t={r['timestamp']:.1f}s\n"
            f"Matches: {r['filtered_matches']} | Sim: {r['global_similarity']:.3f}",
            fontsize=9, color=title_color
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved summary grid: {output_path.name}")


# =============================================================================
# Main function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Video frame matching with pre-fitted VLAD vocabulary"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--vocab", required=True, help="Path to VLAD vocabulary (.npz)")
    parser.add_argument("--calibration", required=True, help="Path to calibration file (JSON)")
    parser.add_argument("--dataset-family", default="tbnms", 
                        help="Dataset family for calibration loading (default: tbnms)")
    parser.add_argument("--output", default="video_matches", help="Output directory")
    parser.add_argument("--ref-time", type=float, required=True, 
                        help="Reference frame timestamp (seconds)")
    parser.add_argument("--search-start", type=float, default=0, 
                        help="Search range start (seconds)")
    parser.add_argument("--search-end", type=float, default=None, 
                        help="Search range end (seconds, default: video end)")
    parser.add_argument("--sample-step", type=float, default=2.5, 
                        help="Time step between candidate frames (seconds)")
    parser.add_argument("--top-k", type=int, default=10, 
                        help="Number of top candidates for local matching")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device")
    parser.add_argument("--scale", type=float, default=0.5, 
                        help="Image scale factor")
    parser.add_argument("--ransac-threshold", type=float, default=0.5,
                        help="RANSAC inlier threshold")
    parser.add_argument("--min-matches-viz", type=int, default=5,
                        help="Minimum filtered matches to save visualization")
    args = parser.parse_args()
    
    # Setup output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load calibration (same as compute_netvlad_vocab.py)
    print(f"Loading calibration from: {args.calibration}")
    print(f"  Dataset family: {args.dataset_family}")
    calibration: TBNMSCalibration = load_calibration(args.dataset_family, args.calibration)
    
    # Initialize video extractor
    with VideoFrameExtractor(args.video) as video:
        print(f"Video: {video.width}x{video.height}, {video.duration:.1f}s @ {video.fps:.1f}fps")
        
        # Determine search range
        search_end = args.search_end if args.search_end else video.duration
        
        # Initialize stereo processor with calibration
        stereo = StereoProcessor(calibration, scale=args.scale)
        stereo.initialize((video.height, video.width))
        print(f"Processed image size: {stereo.new_size[0]}x{stereo.new_size[1]}")
        
        # Initialize matcher with pre-fitted vocabulary
        matcher = VideoMatcher(args.vocab, device=args.device)
        
        # Extract reference frame
        print(f"\nExtracting reference frame at t={args.ref_time:.1f}s")
        ref_frame = video.extract_frame(args.ref_time)
        if ref_frame is None:
            raise RuntimeError("Failed to extract reference frame")
            
        ref_gray = stereo.undistort_to_gray(ref_frame)
        ref_embedding, _ = matcher.extract_embedding(ref_gray)
        ref_tensor = matcher._frame_to_tensor(ref_gray)
        print(f"Reference embedding shape: {ref_embedding.shape}")
        
        # =======================================================================
        # PHASE 1: Compute global embeddings for all candidates
        # =======================================================================
        print("\n" + "="*80)
        print("PHASE 1: Computing global embeddings for all candidates")
        print("="*80)
        
        sample_times = np.arange(args.search_start, search_end, args.sample_step)
        print(f"Processing {len(sample_times)} candidate frames...")
        
        candidates = matcher.compute_global_similarities(
            video, stereo, ref_embedding, sample_times
        )
        
        # =======================================================================
        # PHASE 2: Select top-k candidates by global similarity
        # =======================================================================
        print("\n" + "="*80)
        print(f"PHASE 2: Selecting top {args.top_k} candidates by global similarity")
        print("="*80)
        
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_k_candidates = candidates[:args.top_k]
        
        print(f"\nTop {args.top_k} candidates by global similarity:")
        print(f"{'Rank':>4} | {'Time':>8} | {'Similarity':>10}")
        print("-" * 30)
        for i, c in enumerate(top_k_candidates):
            print(f"{i+1:4d} | {c['timestamp']:8.1f}s | {c['similarity']:.4f}")
        
        # =======================================================================
        # PHASE 3: Run SuperGlue local matching on top-k
        # =======================================================================
        print("\n" + "="*80)
        print(f"PHASE 3: Running SuperGlue local matching on top {args.top_k} candidates")
        print("="*80)
        
        results = matcher.run_local_matching(
            video, stereo, ref_gray, ref_tensor, top_k_candidates,
            ransac_threshold=args.ransac_threshold
        )
        
        # Sort by filtered matches
        results.sort(key=lambda x: x["filtered_matches"], reverse=True)
        
        # =======================================================================
        # Print final results
        # =======================================================================
        print("\n" + "="*80)
        print("FINAL RESULTS (sorted by filtered matches)")
        print("="*80)
        print(f"{'Rank':>4} | {'Time':>8} | {'Global Sim':>10} | {'Raw':>6} | {'Filtered':>8} | {'Conf':>6}")
        print("-" * 60)
        
        for i, r in enumerate(results):
            print(f"{i+1:4d} | {r['timestamp']:8.1f}s | {r['global_similarity']:.4f} | "
                  f"{r['raw_matches']:6d} | {r['filtered_matches']:8d} | {r['mean_match_conf']:.4f}")
        
        # =======================================================================
        # Save match visualizations
        # =======================================================================
        print("\n" + "="*80)
        print("Saving match visualizations")
        print("="*80)
        
        viz_dir = out_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for i, r in enumerate(results):
            if r["filtered_matches"] >= args.min_matches_viz:
                viz_path = viz_dir / f"match_rank{i+1:02d}_t{int(r['timestamp'])}s.png"
                
                # Color matches by confidence
                if len(r["match_conf"]) > 0:
                    colors = cm.jet(r["match_conf"])[:, :3]
                else:
                    colors = np.array([])
                
                title_lines = [
                    f"Rank {i+1}: t={r['timestamp']:.1f}s (ref: t={args.ref_time:.1f}s)",
                    f"VLAD Sim: {r['global_similarity']:.4f} | Matches: {r['raw_matches']}->{r['filtered_matches']} | Conf: {r['mean_match_conf']:.3f}"
                ]
                
                make_matching_plot(
                    ref_gray, r["gray_image"],
                    r["keypoints0"], r["keypoints1"],
                    r["matched_kpts0"], r["matched_kpts1"],
                    colors, title_lines, str(viz_path),
                    False, False, False, "Matches", [],
                    no_lines=True
                )
                saved_count += 1
                print(f"  Saved: {viz_path.name}")
        
        print(f"\nSaved {saved_count} visualizations to: {viz_dir}/")
        
        # Save a summary grid image showing top matches
        if len(results) > 0:
            _save_summary_grid(ref_gray, results, args.ref_time, viz_dir / "summary_grid.png")
        
        # =======================================================================
        # Summary statistics
        # =======================================================================
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if results:
            best = results[0]
            print(f"Best match: t={best['timestamp']:.1f}s with {best['filtered_matches']} filtered matches")
            print(f"  Global similarity: {best['global_similarity']:.4f}")
            
            # Correlation between global similarity and local matches
            sims = np.array([r["global_similarity"] for r in results])
            matches = np.array([r["filtered_matches"] for r in results])
            if len(results) >= 3:
                corr = np.corrcoef(sims, matches)[0, 1]
                print(f"\nCorrelation (global sim vs local matches): {corr:.4f}")
        
        # Save results to CSV
        csv_path = out_dir / "results.csv"
        with open(csv_path, 'w') as f:
            f.write("rank,timestamp,global_similarity,raw_matches,filtered_matches,mean_conf\n")
            for i, r in enumerate(results):
                f.write(f"{i+1},{r['timestamp']},{r['global_similarity']:.4f},"
                       f"{r['raw_matches']},{r['filtered_matches']},{r['mean_match_conf']:.4f}\n")
        print(f"\nResults saved to: {csv_path}")
        print(f"Visualizations saved to: {viz_dir}/")


if __name__ == "__main__":
    main()


"""
Example usage:

python3 video_match.py \
    --video /path/to/video.mp4 \
    --vocab ./vlad_vocab.npz \
    --calibration ../../cfg/calibration/calib_vi_optimized_surfslam.yaml \
    --dataset-family tbnms \
    --ref-time 609 \
    --search-start 300 \
    --search-end 500

"""