import os, sys
import cv2
import numpy as np
import torch
from common.frame import Frame
import kornia

code_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(code_dir, os.pardir, os.pardir, os.pardir))
superglue_path = os.path.join(project_root, "submodules", "SuperGluePretrainedNetwork")
sys.path.append(superglue_path)
from common.settings import Settings
from models.matching import Matching
from mapping.place_recognition.image_descriptor import ImageDescriptorManager


class SuperGlueMatcher:
    def __init__(self,
                 settings: Settings,
                 calibration: Settings,
                 img_descriptor_manager: ImageDescriptorManager,
                 device="cuda:0"):
        self.device = device

        self.histogram_equalize = settings.histogram_equalize
        keypoint_config = settings.keypoints
        self.matcher = Matching(keypoint_config).eval().to(device)

        # this is a "cache", but size is not limited.
        self._feature_cache = {}

        self._img_descriptor_manager = img_descriptor_manager

        self._frame_descriptors = None
        self._frame_ids = None
        self._frame_times = None

        self._calibration = calibration
        self._settings = settings

    def extract_features(self, img_gray: torch.Tensor):
        tensor0 = img_gray.float().to(self.device)
        return self.matcher.superpoint({"image": tensor0})

    def equalize(self, img: torch.Tensor):
        if img.dim() == 2:
            img_b = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3:
            img_b = img.unsqueeze(0)
        else:
            img_b = img

        # Normalize to [0, 1]
        if not img_b.is_floating_point():
            img_b = img_b.float() / 255.0

        lab = kornia.color.rgb_to_lab(img_b)

        # Split channels
        L = lab[:, 0:1, :, :] / 100
        ab = lab[:, 1:3, :, :]

        # Apply CLAHE only on luminance
        L_eq = kornia.enhance.equalize_clahe(L,
            clip_limit=0.03,
            grid_size=(8, 8))

        # Recombine and convert back to RGB
        lab_eq = torch.cat([L_eq*100, ab], dim=1)
        img_eq = kornia.color.lab_to_rgb(lab_eq).clamp(0.0, 1.0)

        return img_eq.squeeze()

    def prepare_frame(self, frame: Frame):
        img = frame.stereo_image.left_image
            
        img_tensor = self.equalize(img)
        img_gray = kornia.color.rgb_to_grayscale(img_tensor.unsqueeze(0))
        
        if frame.depth_image is not None:
            valid_mask = (frame.depth_image.image > 0.05).squeeze()
            img_gray[..., ~valid_mask] = 0

        # Normalize to [0, 1] and ensure shape is (1, 1, H, W)
        return img_gray.float().squeeze()[None, None]
    
    def prepare_frames(self, frame0: Frame, frame1: Frame):
        return self.prepare_frame(frame0), self.prepare_frame(frame1)

    def register_frame(self, frame: Frame):
        img_tensor = self.prepare_frame(frame)
        self._get_or_extract_features(frame, img_tensor)

    def _get_or_extract_features(self, frame: Frame, img_tensor: torch.Tensor):
        """
        Get features from cache or extract and cache them.

        Args:
            frame: Frame object with ._id attribute
            img_tensor: Preprocessed grayscale tensor (1, 1, H, W)

        Returns:
            dict with keys: 'keypoints', 'scores', 'descriptors', 'image_embedding'
        """
        frame_id = frame._id

        if frame_id in self._feature_cache:
            return self._feature_cache[frame_id]

        with torch.no_grad():
            pred = self.matcher.superpoint({"image": img_tensor.to(self.device)})

        keypoints = pred['keypoints'][0].cpu().numpy()
        scores = pred['scores'][0].cpu().numpy()
        descriptors = pred['descriptors'][0].cpu().numpy()

        desc_for_vlad = pred['descriptors'][0].T
        image_embedding = self._img_descriptor_manager.aggregate(desc_for_vlad, pred['scores'][0])

        cache_entry = {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'image_embedding': image_embedding,
            'frame': frame
        }

        self._feature_cache[frame_id] = cache_entry

        new_frame_id = torch.as_tensor(frame_id).int()[None]
        new_frame_time = torch.as_tensor(frame.get_time(), dtype=torch.float64)[None]
        if self._frame_ids is None:
            self._frame_ids = new_frame_id
            self._frame_descriptors = image_embedding[None]
            self._frame_times = new_frame_time
        else:
            self._frame_ids = torch.cat((self._frame_ids, new_frame_id))
            self._frame_descriptors = torch.vstack((self._frame_descriptors, image_embedding[None]))
            self._frame_times = torch.cat((self._frame_times, new_frame_time))
        
        return cache_entry

    def _filter_matches_ransac(self,
                               mkpts0: np.ndarray,
                               mkpts1: np.ndarray,
                               mconf: np.ndarray) -> tuple:
        """
        Filter matches using RANSAC on essential matrix.

        Args:
            mkpts0: Matched keypoints in frame0 (N, 2)
            mkpts1: Matched keypoints in frame1 (N, 2)
            mconf: Match confidence scores (N,)

        Returns:
            Tuple of (filtered_mkpts0, filtered_mkpts1, filtered_mconf, inlier_mask, ransac_success)
            where ransac_success is False if there were insufficient points or RANSAC failed
        """
        if len(mkpts0) < 8:  # Need at least 8 points for essential matrix
            # Not enough points for RANSAC
            empty_mask = np.zeros(len(mkpts0), dtype=bool)
            return mkpts0[:0], mkpts1[:0], mconf[:0], empty_mask, False

        # Get camera intrinsic matrix
        K = np.array(self._calibration.camera_intrinsic.k, dtype=np.float64)

        # Use OpenCV's RANSAC implementation
        E, inlier_mask = cv2.findEssentialMat(
            mkpts0,
            mkpts1,
            K,
            method=cv2.RANSAC,
            prob=self._settings.ransac.prob,
            threshold=self._settings.ransac.threshold_px
        )

        if inlier_mask is None or E is None:
            # RANSAC failed
            empty_mask = np.zeros(len(mkpts0), dtype=bool)
            return mkpts0, mkpts1, mconf, empty_mask, False

        inlier_mask = inlier_mask.ravel().astype(bool)
        return mkpts0[inlier_mask], mkpts1[inlier_mask], mconf[inlier_mask], inlier_mask, True

    def match(self, frame0: Frame, frame1: Frame, use_ransac: bool = True, debug_path=False):
        """Match two frames using cached SuperPoint features + SuperGlue.

        Args:
            frame0: First frame
            frame1: Second frame
            use_ransac: If True, filter matches using RANSAC on essential matrix
            debug: If True, visualize matches before and after RANSAC

        Returns:
            mkpts0, mkpts1, mconf, kpts0_np, kpts1_np
        """
        tensor0_gray, tensor1_gray = self.prepare_frames(frame0, frame1)

        features0 = self._get_or_extract_features(frame0, tensor0_gray)
        features1 = self._get_or_extract_features(frame1, tensor1_gray)

        kpts0 = torch.from_numpy(features0['keypoints']).unsqueeze(0).to(self.device)
        kpts1 = torch.from_numpy(features1['keypoints']).unsqueeze(0).to(self.device)
        scores0 = torch.from_numpy(features0['scores']).unsqueeze(0).to(self.device)
        scores1 = torch.from_numpy(features1['scores']).unsqueeze(0).to(self.device)
        desc0 = torch.from_numpy(features0['descriptors']).unsqueeze(0).to(self.device)
        desc1 = torch.from_numpy(features1['descriptors']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.matcher({
                'keypoints0': kpts0,
                'keypoints1': kpts1,
                'scores0': scores0,
                'scores1': scores1,
                'descriptors0': desc0,
                'descriptors1': desc1,
                'image0': tensor0_gray.to(self.device),
                'image1': tensor1_gray.to(self.device),
            })

        pred_np = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0_np = features0['keypoints']
        kpts1_np = features1['keypoints']
        matches = pred_np["matches0"]
        conf = pred_np["matching_scores0"]

        valid = matches > -1
        mkpts0 = kpts0_np[valid]
        mkpts1 = kpts1_np[matches[valid]]
        mconf = conf[valid]

        if use_ransac and self._calibration is not None:
            mkpts0_filtered, mkpts1_filtered, mconf_filtered, ransac_inliers, ransac_success = self._filter_matches_ransac(
                mkpts0, mkpts1, mconf
            )

            # Visualize matches after RANSAC if debug is enabled
            if debug_path:
                self.visualize_matches(frame0, frame1, mkpts0, mkpts1, mconf,
                                       passfail=ransac_success, inlier_stats=None,
                                       out_path=debug_path, inliers=ransac_inliers)

            return mkpts0_filtered, mkpts1_filtered, mconf_filtered, kpts0_np, kpts1_np, ransac_inliers

        return mkpts0, mkpts1, mconf, kpts0_np, kpts1_np
    
    def get_loop_candidates(self, frame: Frame, debug=False, n_candidates=3):
        
        if self._frame_ids is None:
            return None

        if frame.get_id() not in self._feature_cache:
            img_gray = self.prepare_frame(frame)
            feature_info = self._get_or_extract_features(frame, img_gray)
        else:
            feature_info = self._feature_cache[frame.get_id()]

        img_descriptor = feature_info['image_embedding']

        frame_mask = (self._frame_ids != frame.get_id()) & (frame.get_time() - self._frame_times >= self._settings.loop_closure.min_time_diff_sec)
        
        if not frame_mask.any():
            return None

        target_descriptors = self._frame_descriptors[frame_mask]
        target_ids = self._frame_ids[frame_mask]

        similarities_all: torch.Tensor = self._img_descriptor_manager.compute_similarity_batch(
            img_descriptor, target_descriptors)

        similarities, matches = torch.topk(similarities_all, k=min(n_candidates, frame_mask.sum()))

        sim_thresh = self._settings.loop_closure.similarity_threshold
        if sim_thresh is not None:
            mask: torch.BoolTensor = similarities >= sim_thresh
            if not mask.any():
                print(
                    f"[INFO] No loop closure candidates exceed similarity threshold of {sim_thresh}."
                )
                return None
            matches = matches[mask]
            similarities = similarities[mask]

        valid_candidates = []

        for candidate in matches:
            candidate_id = target_ids[candidate]
            candidate_frame: Frame = self._feature_cache[candidate_id.item()]['frame']

            mkpts0, mkpts1, mconf, _, _, inliers = self.match(frame, candidate_frame, True)
            
            ransac_pass = inliers.sum() >= self._settings.loop_closure.ransac_min_inliers_ignore_ratio or \
                          (inliers.sum() >= self._settings.loop_closure.ransac_min_inliers and \
                          inliers.mean() >= self._settings.loop_closure.ransac_min_inlier_ratio)
            out_path = os.path.join(self._settings.log_directory, "loop_candidates", 
                                    f"{frame.get_id()}_matched_{candidate_frame.get_id()}.png")

            if ransac_pass:
                valid_candidates.append((candidate_frame.get_id(), mkpts0, mkpts1, mconf, inliers))

                if self._settings.loop_closure.stop_at_first_candidate:
                    if debug:
                        self.visualize_matches(frame, candidate_frame, mkpts0, mkpts1, mconf, 
                                               passfail=ransac_pass, inlier_stats=(inliers.sum(), len(inliers)),
                                               out_path=out_path)

                    return valid_candidates
 
            if debug:
                self.visualize_matches(frame, candidate_frame, mkpts0, mkpts1, mconf, 
                                       passfail=ransac_pass, inlier_stats=(inliers.sum(), len(inliers)),
                                       out_path=out_path)

        return valid_candidates if len(valid_candidates) > 0 else None
    

    def _tensor_to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().cpu()
        if arr.dim() == 3:
            arr = arr.permute(1, 2, 0).numpy()
        else:
            arr = arr.numpy()

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    def visualize_matches(
        self,
        frame0: Frame,
        frame1: Frame,
        mkpts0,
        mkpts1,
        mconf,
        passfail=None,
        inlier_stats=None,
        out_path=None,
        inliers=None,
    ):

        img0 = self._tensor_to_bgr(frame0.stereo_image.left_image)
        img1 = self._tensor_to_bgr(frame1.stereo_image.left_image)

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
        canvas[:h0, :w0] = img0
        canvas[:h1, w0:] = img1

        # Add timestamps at top-center of each image
        ts0 = float(frame0.get_time())
        ts1 = float(frame1.get_time())

        timestamp_text0 = f"t={ts0:.3f}s"
        timestamp_text1 = f"t={ts1:.3f}s"

        # Get text sizes
        text_size0, _ = cv2.getTextSize(timestamp_text0, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_size1, _ = cv2.getTextSize(timestamp_text1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Calculate centered positions (top-center of each image)
        x0 = (w0 - text_size0[0]) // 2
        x1 = w0 + (w1 - text_size1[0]) // 2
        y_pos = 25

        # Draw background rectangles for timestamps
        cv2.rectangle(canvas, (x0 - 5, 5), (x0 + text_size0[0] + 5, y_pos + text_size0[1]), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x1 - 5, 5), (x1 + text_size1[0] + 5, y_pos + text_size1[1]), (255, 255, 255), -1)

        # Draw timestamp text
        cv2.putText(canvas, timestamp_text0, (x0, y_pos + text_size0[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, timestamp_text1, (x1, y_pos + text_size1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Alpha-blend helper
        def draw_line_alpha(img, p0, p1, color, alpha, thickness):
            overlay = img.copy()
            cv2.line(overlay, p0, p1, color, thickness)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        def draw_circle_alpha(img, p, radius, color, alpha):
            overlay = img.copy()
            cv2.circle(overlay, p, radius, color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        for idx, (pt0, pt1, c) in enumerate(zip(mkpts0, mkpts1, mconf)):
            if inliers is not None:
                is_inlier = inliers[idx]
                if is_inlier:
                    color = (0, int(255 * c), int(255 * (1 - c)))
                    alpha = 1.0
                    thickness = 1
                    radius = 3
                else:
                    color = (0, 0, 0)
                    alpha = 0.20
                    thickness = 1
                    radius = 2
            else:
                color = (0, int(255 * c), int(255 * (1 - c)))
                alpha = 1.0
                thickness = 1
                radius = 3

            p0 = (int(pt0[0]), int(pt0[1]))
            p1 = (int(pt1[0]) + w0, int(pt1[1]))

            if alpha == 1.0:
                cv2.line(canvas, p0, p1, color, thickness)
                cv2.circle(canvas, p0, radius, color, -1)
                cv2.circle(canvas, p1, radius, color, -1)
            else:
                draw_line_alpha(canvas, p0, p1, color, alpha, thickness)
                draw_circle_alpha(canvas, p0, radius, color, alpha)
                draw_circle_alpha(canvas, p1, radius, color, alpha)

        if passfail is not None:
            text = "PASS" if passfail else "FAIL"
            color = (0, 255, 0) if passfail else (0, 0, 255)  # Green for PASS, Red for FAIL

            # Get text size for background rectangle
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            text_w, text_h = text_size

            # Draw white background rectangle
            cv2.rectangle(canvas, (5, 5), (15 + text_w, 45 + text_h), (255, 255, 255), -1)

            # Draw text on top
            cv2.putText(canvas, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        if inlier_stats is not None or inliers is not None:
            if inliers is None:
                num_inliers, num_total = inlier_stats
            else:
                num_inliers = inliers.sum()
                num_total = len(inliers)

            inlier_ratio = num_inliers / num_total

            # Display inliers info below the PASS/FAIL text
            inlier_text = f"Inliers: {num_inliers}/{num_total} ({inlier_ratio:.1%})"

            # Get text size for background rectangle
            text_size, _ = cv2.getTextSize(inlier_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_w, text_h = text_size

            # Position below PASS/FAIL text (if it exists, otherwise at top)
            y_offset = 80 if passfail is not None else 10

            # Draw white background rectangle
            cv2.rectangle(canvas, (5, y_offset), (15 + text_w, y_offset + text_h + 10), (255, 255, 255), -1)

            # Draw text on top in blue
            cv2.putText(canvas, inlier_text, (10, y_offset + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)


        if out_path is not None:
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_path)

            cv2.imwrite(fname, canvas)
