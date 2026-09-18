import os
import numpy as np
import cv2
import torch
import yaml

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'surfslam'))

from common.dataset_paths import REPO_ROOT
from common.pose import Pose

class TBNMSCalibration:
    def __init__(self, calibration_path: str) -> None:
        calibration_path = os.path.expanduser(calibration_path)
        calibration_path = calibration_path.replace('PROJECT_ROOT', str(REPO_ROOT))

        if calibration_path.endswith('.json'):
            raise ValueError(
                "JSON calibration files are deprecated; point 'calibration:' at "
                "cfg/calibration/calib_vi_optimized_surfslam.yaml")

        with open(calibration_path, 'r') as calib_f:
            calib_data = yaml.safe_load(calib_f)

        W, H = calib_data['left_cam_params']['image_dimension']
        self.size = (W, H)

        def load_calib(cam_data):
            fx, fy = cam_data['focal_length']
            cx, cy = cam_data['principal_point']
            k1, k2, p1, p2 = cam_data['distortion_coefficients']
            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            D = np.array([k1, k2, p1, p2], dtype=np.float64)
            return K, D

        self.K0, self.D0 = load_calib(calib_data['left_cam_params'])
        self.K1, self.D1 = load_calib(calib_data['right_cam_params'])

        if 'stereo_calibration' not in calib_data:
            raise ValueError(
                f"{calibration_path} has no 'stereo_calibration' block; the "
                "canonical calibration file is cfg/calibration/calib_vi_optimized_surfslam.yaml")
        T = np.array(calib_data['stereo_calibration']['T_left_to_right'], dtype=np.float64)
        self.R = T[:3, :3]
        self.t = T[:3, 3]
        self.baseline = float(np.linalg.norm(self.t))

        # Lazy rectification cache
        self._cached_scale = None
        self._map1_l = None
        self._map2_l = None
        self._map1_r = None
        self._map2_r = None
        self._K_new = None
        self._fx = None

    def _ensure_rectified(self, scale: float):
        if self._cached_scale is not None and abs(scale - self._cached_scale) > 1e-9:
            raise ValueError("Rectification already computed with different scale factor")

        if self._cached_scale is not None:
            return

        self._cached_scale = scale
        new_size = (int(self.size[0] * scale), int(self.size[1] * scale))

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K0, self.D0, self.K1, self.D1, self.size, self.R, self.t.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
            newImageSize=new_size
        )

        self._K_new = P1[:3, :3].astype(np.float32)

        self._fx = float(self._K_new[0, 0])
        
        self._map1_l, self._map2_l = cv2.initUndistortRectifyMap(
            self.K0, self.D0, R1, P1, new_size, cv2.CV_32FC1
        )
        self._map1_r, self._map2_r = cv2.initUndistortRectifyMap(
            self.K1, self.D1, R2, P2, new_size, cv2.CV_32FC1
        )

    def rectify_images(self, imgL, imgR, im_scale_factor: float):
        self._ensure_rectified(im_scale_factor)
        rectL = cv2.remap(imgL, self._map1_l, self._map2_l, interpolation=cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self._map1_r, self._map2_r, interpolation=cv2.INTER_LINEAR)
        return rectL, rectR

    def to_dict(self, im_scale_factor: float):
        self._ensure_rectified(im_scale_factor)

        cam_intrinsic = {}
        cam_intrinsic["k"] = torch.from_numpy(self._K_new.copy()).float()
        cam_intrinsic["distortion"] = None
        cam_intrinsic["width"] = int(self.size[0] * im_scale_factor)
        cam_intrinsic["height"] = int(self.size[1] * im_scale_factor)

        result = {}
        result["camera_intrinsic"] = cam_intrinsic
        result["stereo"] = {
            "baseline_m": float(self.baseline),
            "fx": float(self._fx)
        }
        result['imu_to_camera'] = Pose().to_settings()
        return result
