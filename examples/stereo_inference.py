from pathlib import Path
import os, sys
import torch
import argparse
import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/surfslam")

from examples.tbnms.tbnms_calibration import TBNMSCalibration
from examples.utils.pose_utils import load_calibration
from surfslam.common.sensors import StereoImage
from surfslam.common.dataset_paths import load_config, resolve_config_path
from surfslam.common.settings import Settings
from surfslam.mapping.stereo.stereo_estimation_defom import DefomStereoEstimator

from examples.utils.h5_log_reader import H5LogReader


DEVICE = 'cuda:0'
IM_SCALE_FACTOR = 0.5


def apply_colormap(disparity: torch.Tensor) -> torch.Tensor:
    if disparity.dim() == 3:
        disparity = disparity.squeeze(0)
    
    disp_np = disparity.cpu().numpy()
    disp_np[disp_np > 10] = 0
    disp_normalized = cv2.normalize(disp_np, None, 0, 255, cv2.NORM_MINMAX)
    disp_uint8 = disp_normalized.astype(np.uint8)
    colored = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_TURBO)
    
    return torch.from_numpy(colored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_config")
    parser.add_argument("--output_dir", default="stereo_viz_output", 
                        help="Directory to save visualizations")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(PROJECT_ROOT) / "outputs" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.sequence_config)

    baseline_settings_path = resolve_config_path(config['baseline'])
    
    print(f"Loading baseline settings from {baseline_settings_path}")

    settings = Settings.load_from_file(baseline_settings_path)
    if 'changes' in config:
        settings.augment(config['changes'])

    calibration: TBNMSCalibration = load_calibration(config["dataset_family"], config["calibration"])

    settings['calibration'] = calibration.to_dict(IM_SCALE_FACTOR)

    stereo_matcher = DefomStereoEstimator(settings.mapper.stereo_estimator,
                                          settings.calibration,
                                          DEVICE)

    h5_path = Path(os.path.expanduser(config["dataset"]))
    stereo_key = settings.system.h5_keys.stereo
    barometer_key = settings.system.h5_keys.barometer
    dvl_key = settings.system.h5_keys.dvl
    imu_key = settings.system.h5_keys.imu

    log_reader = H5LogReader(h5_path, imu_key, dvl_key, barometer_key, stereo_key)

    frame_idx = 0
    for event in log_reader.events():
        if event.type != "STEREO":
            continue

        ts, left, right = log_reader.stereo()[event.index]
        
        def prep(img):
            torch_img = torch.from_numpy(img).permute(2, 0, 1)
            return torch_img
        
        leftRect, rightRect = calibration.rectify_images(left, right, IM_SCALE_FACTOR)
        left = prep(leftRect)
        right = prep(rightRect)
        stereo_img = StereoImage(left, right, ts)

        result = stereo_matcher.infer(stereo_img)
        
        # Apply colormap to 1-channel disparity output
        disparity_colored = apply_colormap(result.image)
        
        # Stack left image (H, W, 3) with colored disparity (H, W, 3)
        left_hwc = left.permute(1, 2, 0).cpu()
        viz = torch.hstack((left_hwc, disparity_colored))
        
        # Convert to numpy and save
        viz_np = viz.numpy().astype(np.uint8)
        output_path = output_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(output_path), viz_np)
        
        print(f"Saved {output_path}")
        frame_idx += 1

    print(f"\nSaved {frame_idx} frames to {output_dir}")